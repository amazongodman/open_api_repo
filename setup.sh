#!/bin/bash

# システムのアップデートとPython環境のセットアップ
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git nginx

# NVIDIAドライバとCUDAのインストール
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y nvidia-driver-525  # AWS Deep Learning AMIでは不要
sudo apt-get install -y nvidia-cuda-toolkit

# CUDAバージョンの確認
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "Detected CUDA Version: $CUDA_VERSION"

# プロジェクトディレクトリの作成
mkdir -p ~/image-classifier
cd ~/image-classifier

# Python仮想環境の作成とアクティベート
python3 -m venv venv
source venv/bin/activate

# CUDAバージョンに応じたPyTorchのインストール
MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)

if [ "$MAJOR_VERSION" = "11" ] && [ "$MINOR_VERSION" = "8" ]; then
    echo "Installing PyTorch for CUDA 11.8"
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
elif [ "$MAJOR_VERSION" = "12" ]; then
    echo "Installing PyTorch for CUDA 12.1"
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Warning: CUDA version $CUDA_VERSION might not be fully supported."
    echo "Supported versions are: CUDA 11.8 and CUDA 12.x"
    echo "Installing PyTorch for the closest supported CUDA version..."
    
    if [ "$MAJOR_VERSION" -lt "12" ]; then
        echo "Using CUDA 11.8 build"
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Using CUDA 12.1 build"
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi
fi

# 他の必要なPythonパッケージのインストール
pip install fastapi uvicorn python-multipart aiofiles python-jose[cryptography] passlib[bcrypt] python-dotenv

# GitHubからコードをクローン
git clone https://github.com/YOUR_GITHUB_REPO .

# .envファイルの作成
echo "SECRET_KEY=$(openssl rand -hex 32)" > .env
echo "PASSWORD=your_password_here" >> .env

# Nginxの設定
sudo bash -c 'cat > /etc/nginx/sites-available/image-classifier << EOL
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOL'

# Nginxの設定を有効化
sudo ln -sf /etc/nginx/sites-available/image-classifier /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

# サービスファイルの作成
sudo bash -c 'cat > /etc/systemd/system/image-classifier.service << EOL
[Unit]
Description=Image Classifier Web Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/image-classifier
Environment="PATH=/home/ubuntu/image-classifier/venv/bin"
ExecStart=/home/ubuntu/image-classifier/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
EOL'

# サービスの有効化と起動
sudo systemctl enable image-classifier
sudo systemctl start image-classifier

# CUDA環境の確認
echo "Checking CUDA installation..."
nvcc --version

# GPUの状態確認
echo "Checking NVIDIA GPU status..."
nvidia-smi

# PyTorchがGPUを認識しているか確認するスクリプトの作成
cat > check_gpu.py << EOL
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device name:", torch.cuda.get_device_name(0))
EOL

echo "Checking PyTorch GPU status..."
python3 check_gpu.py

echo "セットアップが完了しました！"