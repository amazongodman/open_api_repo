import os
import io
import json
import logging
import boto3
import uuid
from datetime import datetime
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
S3_BUCKET = os.getenv("S3_BUCKET")

if not PASSWORD:
    raise RuntimeError("PASSWORD environment variable is not set")
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is not set")

# S3クライアントの初期化
s3_client = boto3.client('s3')

# CUDA設定とデバイスの選択
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
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
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.to(DEVICE)
    model.eval()
    # ImageNet1Kクラスのラベルを取得
    class_labels = weights.meta["categories"]
    logger.info(f"Loaded {len(class_labels)} class labels")
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
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB"
        }

    return {
        "status": "healthy",
        "device": DEVICE.type,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "model_loaded": True
    }

@app.get("/labels")
async def get_labels() -> Dict[str, Any]:
    """クラスラベル一覧を返す"""
    return {
        "total_classes": len(class_labels),
        "labels": class_labels
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
                .result {{ margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
                .device-info {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
                .prediction {{ margin: 10px 0; padding: 10px; background-color: white; border-radius: 3px; }}
                .prediction-rank-1 {{ border-left: 4px solid #4CAF50; }}
                .prediction-rank-2 {{ border-left: 4px solid #2196F3; }}
                .prediction-rank-3 {{ border-left: 4px solid #9C27B0; }}
                .result-container {{ display: flex; gap: 20px; }}
                .image-container {{ flex: 0 0 300px; }}
                .image-container img {{ max-width: 100%; height: auto; border-radius: 5px; }}
                .predictions-container {{ flex: 1; }}
            </style>
            <script>
                async function submitForm(event) {{
                    event.preventDefault();
                    const formData = new FormData(event.target);
                    try {{
                        const response = await fetch('/predict', {{
                            method: 'POST',
                            body: formData,
                            headers: {{
                                'Authorization': 'Basic ' + btoa(':' + '{PASSWORD}')
                            }}
                        }});
                        const result = await response.json();
                        
                        let resultHtml = '<div class="result-container">';
                        
                        // 画像の表示
                        resultHtml += `
                            <div class="image-container">
                                <h3>アップロードされた画像:</h3>
                                <img src="${result.image_url}" alt="Uploaded image">
                            </div>
                            <div class="predictions-container">
                                <h3>分類結果:</h3>`;
                        
                        result.predictions.forEach((pred, index) => {{
                            resultHtml += `
                                <div class="prediction prediction-rank-${{index + 1}}">
                                    <strong>#${{index + 1}}:</strong> ${{pred.class}}<br>
                                    <span>確信度: ${{pred.probability}}</span>
                                </div>`;
                        }});
                        
                        resultHtml += `
                                <p>処理時間: ${{result.process_time}}</p>`;
                        if (result.gpu_memory) {{
                            resultHtml += `<p>GPU メモリ使用量: ${{result.gpu_memory.allocated}}</p>`;
                        }}
                        resultHtml += `<p>保存先: ${{result.s3_path}}</p>
                            </div>
                        </div>`;
                        
                        document.getElementById('result').innerHTML = resultHtml;
                    }} catch (error) {{
                        document.getElementById('result').innerHTML = `<p style="color: red;">エラーが発生しました: ${{error.message}}</p>`;
                    }}
                }}
            </script>
        </head>
        <body>
            <h1>画像分類アプリ</h1>
            <div class="device-info">
                <p>実行環境: {device_info}</p>
            </div>
            <form onsubmit="submitForm(event)" class="upload-form" enctype="multipart/form-data">
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

        # S3に画像を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        s3_key = f"uploads/{timestamp}_{unique_id}_{file.filename}"
        
        # 画像をメモリ上でバイト列に変換
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # S3にアップロード
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=img_byte_arr,
            ContentType=file.content_type
        )

        # S3の画像へのURLを生成（24時間有効）
        image_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=86400
        )
        
        # 前処理とデバイスへの転送
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 予測実行
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 上位3クラスの予測結果を取得
            top_probs, top_indices = torch.topk(probabilities[0], 3)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    "class": class_labels[idx.item()],
                    "probability": f"{prob.item()*100:.2f}%"
                })

        process_time = time.time() - start_time
        
        # GPUメモリ使用状況の取得（GPUモードの場合）
        gpu_memory_info = {}
        if DEVICE.type == 'cuda':
            gpu_memory_info = {
                "allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
                "cached": f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB"
            }

        # 予測結果もS3に保存
        result_data = {
            "filename": file.filename,
            "predictions": predictions,
            "process_time": f"{process_time:.3f} seconds",
            "timestamp": timestamp,
            "s3_image_path": s3_key
        }
        
        results_key = f"results/{timestamp}_{unique_id}_result.json"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=results_key,
            Body=json.dumps(result_data, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

        return {
            "filename": file.filename,
            "predictions": predictions,
            "process_time": f"{process_time:.3f} seconds",
            "device_used": DEVICE.type,
            "gpu_memory": gpu_memory_info if DEVICE.type == 'cuda' else None,
            "image_url": image_url,
            "s3_path": s3_key
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)