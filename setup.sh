#!/bin/bash

# エラーが発生した場合にスクリプトを停止
set -e

# ログ機能
log_file="/var/log/setup_script.log"
exec 1> >(tee -a "$log_file") 2>&1

echo "Setup started at $(date)"

# ディスク容量の確認
df -h /
available_space=$(df -k / | awk 'NR==2 {print $4}')
if [ "$available_space" -lt 30000000 ]; then  # 30GB未満の場合
    echo "Error: Insufficient disk space. At least 30GB required."
    exit 1
fi

# システムのアップデートとPython環境のセットアップ
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    nginx \
    linux-headers-$(uname -r) \
    make \
    wget \
    curl

# NVIDIAドライバーとCUDAのセットアップ
echo "Installing NVIDIA drivers and CUDA..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit

# ドライバーの確認
if ! nvidia-smi &>/dev/null; then
    echo "NVIDIA driver installation failed. Please reboot and run the script again."
    exit 1
fi

# CUDAバージョンの確認
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "Detected CUDA Version: $CUDA_VERSION"

# プロジェクトディレクトリの設定
PROJECT_DIR=~/image-classifier
echo "Setting up project directory at $PROJECT_DIR"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Python仮想環境の作成
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 必要なPythonパッケージのインストール
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install fastapi uvicorn python-multipart aiofiles python-jose[cryptography] passlib[bcrypt] python-dotenv

# PyTorch GPUサポートの確認
echo "Checking PyTorch GPU support..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

# アプリケーションの設定
echo "Configuring application..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "SECRET_KEY=$(openssl rand -hex 32)" > .env
    echo "PASSWORD=change_this_password" >> .env
fi

# Nginxの設定
echo "Configuring Nginx..."
sudo bash -c 'cat > /etc/nginx/sites-available/image-classifier << EOL
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOL'

sudo ln -sf /etc/nginx/sites-available/image-classifier /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

# システムサービスの設定
echo "Setting up systemd service..."
sudo bash -c 'cat > /etc/systemd/system/image-classifier.service << EOL
[Unit]
Description=Image Classifier Web Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/image-classifier
Environment="PATH=/home/ubuntu/image-classifier/venv/bin"
ExecStart=/home/ubuntu/image-classifier/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL'

sudo systemctl daemon-reload
sudo systemctl enable image-classifier
sudo systemctl start image-classifier

echo "Setup completed at $(date)"
echo "Please check $log_file for detailed logs"

# システムの状態確認
echo "Final system checks..."
nvidia-smi
systemctl status nginx
systemctl status image-classifier