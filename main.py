import os
import io
import logging
from typing import Dict, Any
from pathlib import Path
import secrets
import time
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()
PASSWORD = os.getenv("PASSWORD")
if not PASSWORD:
    raise RuntimeError("PASSWORD environment variable is not set")

# CUDA設定とデバイスの選択
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    # GPU メモリ情報
    logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB")

app = FastAPI(title="Image Classification API")
security = HTTPBasic()

# 静的ファイルのディレクトリを作成
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# モデルの初期化
try:
    logger.info("Loading model...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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

# ImageNetのクラスラベル（上位20クラス）
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

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """ヘルスチェックエンドポイント"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
            "memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB"
        }

    return {
        "status": "healthy",
        "device": DEVICE.type,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "model_loaded": True
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(credentials: HTTPBasicCredentials = Depends(verify_password)):
    """メインページの表示"""
    device_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU Mode"
    return f"""
    <html>
        <head>
            <title>画像分類アプリ</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .info {{ color: #666; margin-bottom: 20px; }}
                .upload-form {{ margin-top: 20px; }}
                .result {{ margin-top: 20px; }}
                .device-info {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>画像分類アプリ</h1>
            <div class="device-info">
                <p>実行環境: {device_info}</p>
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
        start_time = time.time()
        
        # 画像の読み込みと前処理
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # RGBモードでない場合は変換
        if image.mode != "RGB":
            image = image.convert("RGB")
            logger.info(f"Converted image from {image.mode} to RGB")
        
        # 前処理とデバイスへの転送
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 予測実行
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # 確率の取得
            probability = probabilities[0][predicted].item() * 100

        # 処理時間の計算
        process_time = time.time() - start_time
        
        # GPUメモリ使用状況の取得（GPUモードの場合）
        gpu_memory_info = {}
        if DEVICE.type == 'cuda':
            gpu_memory_info = {
                "allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
                "cached": f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB"
            }

        predicted_class = class_labels[predicted.item()]
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{probability:.2f}%",
            "device_used": DEVICE.type,
            "process_time": f"{process_time:.3f} seconds",
            "gpu_memory": gpu_memory_info if DEVICE.type == 'cuda' else None
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバルな例外ハンドラ"""
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "type": str(type(exc).__name__)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)