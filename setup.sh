#!/bin/bash

# エラーが発生した場合にスクリプトを停止
set -e

# ログ機能
log_file="/var/log/setup_script.log"
exec 1> >(tee -a "$log_file") 2>&1

echo "Setup started at $(date)"

# S3バケット設定
S3_BUCKET="your-bucket-name"  # このバケット名は要変更
S3_MOUNT_POINT="/mnt/s3-bucket"

# ディスク容量の確認
df -h /
available_space=$(df -k / | awk 'NR==2 {print $4}')
if [ "$available_space" -lt 30000000 ]; then  # 30GB未満の場合
    echo "Error: Insufficient disk space. At least 30GB required."
    exit 1
fi

# システムのアップデートとPython環境のセットアップ
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    nginx \
    s3fs

# プロジェクトディレクトリの設定
PROJECT_DIR=$(pwd)
echo "Setting up project in: $PROJECT_DIR"

# Python仮想環境の作成
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 必要なPythonパッケージのインストール
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install fastapi uvicorn python-multipart aiofiles python-jose[cryptography] passlib[bcrypt] python-dotenv

# PyTorch GPUサポートの確認
echo "Checking PyTorch GPU support..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

# S3マウントの設定
echo "Setting up S3 mount..."
sudo mkdir -p $S3_MOUNT_POINT
sudo chown ubuntu:ubuntu $S3_MOUNT_POINT

# IMDSv2のトークン取得
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# リージョンの取得
REGION=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)

# fstabにS3マウント設定を追加
echo "$S3_BUCKET $S3_MOUNT_POINT fuse.s3fs _netdev,allow_other,use_path_request_style,iam_role=auto,url=https://s3-${REGION}.amazonaws.com 0 0" | sudo tee -a /etc/fstab

# S3マウント実行
sudo mount -a

# 環境設定ファイルの作成
echo "Creating environment file..."
if [ ! -f .env ]; then
    echo "SECRET_KEY=$(openssl rand -hex 32)" > .env
    echo "PASSWORD=change_this_password" >> .env
    echo "S3_BUCKET=$S3_BUCKET" >> .env
    echo "S3_MOUNT_POINT=$S3_MOUNT_POINT" >> .env
    chmod 600 .env
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
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOL'

sudo ln -sf /etc/nginx/sites-available/image-classifier /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

# システムサービスの設定
echo "Setting up systemd service..."
sudo bash -c "cat > /etc/systemd/system/image-classifier.service << EOL
[Unit]
Description=Image Classifier Web Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$PROJECT_DIR
Environment=\"PATH=$PROJECT_DIR/venv/bin\"
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL"

sudo systemctl daemon-reload
sudo systemctl enable image-classifier
sudo systemctl start image-classifier

echo "Setup completed at $(date)"
echo "Please check $log_file for detailed logs"

# 最終確認
echo "Final system checks..."
echo "1. NVIDIA Driver and CUDA:"
nvidia-smi
echo ""
echo "2. Services status:"
sudo systemctl status nginx --no-pager
sudo systemctl status image-classifier --no-pager
echo ""
echo "3. S3 mount status:"
df -h | grep s3fs

echo ""
echo "Setup complete! Please check the services status above."