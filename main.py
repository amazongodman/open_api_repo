import os
import io
import logging
import secrets
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()
PASSWORD = os.getenv("PASSWORD")
if not PASSWORD:
    raise RuntimeError("PASSWORD environment variable is not set")

app = FastAPI()
security = HTTPBasic()

# CUDA利用可能性のチェックとデバイスの設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

# 静的ファイルのディレクトリを作成
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    # モデルの準備
    model = resnet50(pretrained=True)
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ImageNetのクラスラベル（簡略化のため上位20クラスのみ）
class_labels = [
    'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
    'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin',
    'bulbul', 'jay', 'magpie', 'chickadee'
]

def verify_password(credentials: HTTPBasicCredentials = Depends(security)):
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not correct_password:
        raise HTTPException(
            status_code=401,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "device": DEVICE.type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(credentials: HTTPBasicCredentials = Depends(verify_password)):
    """メインページの表示"""
    cuda_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU Mode"
    return f"""
    <html>
        <head>
            <title>画像分類アプリ</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .info {{ color: #666; margin-bottom: 20px; }}
                .upload-form {{ margin-top: 20px; }}
                .result {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>画像分類アプリ</h1>
            <div class="info">
                <p>実行環境: {cuda_info}</p>
            </div>
            <form action="/predict" method="post" enctype="multipart/form-data" class="upload-form">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="分類する">
            </form>
            <div id="result" class="result"></div>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(verify_password)
) -> Dict[str, Any]:
    """画像の分類を実行"""
    try:
        # 画像の読み込みと前処理
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # RGBモードでない場合は変換
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 前処理とデバイスへの転送
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 予測実行
        with torch.no_grad():
            try:
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # 確率の取得
                probability = probabilities[0][predicted].item() * 100
                
            except RuntimeError as e:
                logger.error(f"Error during inference: {e}")
                raise HTTPException(status_code=500, detail="Inference failed")

        predicted_class = class_labels[predicted.item()]
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{probability:.2f}%",
            "device_used": DEVICE.type
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """グローバルな例外ハンドラ"""
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)