#!/bin/bash

# エラーが発生した場合にスクリプトを停止
set -e

# ログ機能
log_file="/var/log/setup_script.log"
exec 1> >(tee -a "$log_file") 2>&1

echo "Setup started at $(date)"

# S3バケット設定
S3_BUCKET="your-dataset-bucket-name"  # このバケット名は要変更
S3_MOUNT_POINT="/mnt/s3-bucket"

# 必要なパッケージのインストール
echo "Installing required packages..."
sudo apt-get update
sudo apt-get install -y nginx s3fs

# S3マウントの設定
echo "Setting up S3 mount..."
sudo mkdir -p $S3_MOUNT_POINT
sudo chown ubuntu:ubuntu $S3_MOUNT_POINT

# IMDSv2のトークン取得とリージョンの設定
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
REGION=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
# fstabにS3マウント設定を追加
echo "$S3_BUCKET $S3_MOUNT_POINT fuse.s3fs _netdev,allow_other,use_path_request_style,iam_role=auto,url=https://s3-${REGION}.amazonaws.com 0 0" | sudo tee -a /etc/fstab

# もしくは
# AWS_REGION="ap-northeast-1"
# echo "$S3_BUCKET $S3_MOUNT_POINT fuse.s3fs _netdev,allow_other,use_path_request_style,iam_role=auto,endpoint=${AWS_REGION} 0 0" | sudo tee -a /etc/fstab

# 確認
sudo mount -a

# 仮想環境の作成とパッケージのインストール
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# https://pytorch.org/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install fastapi uvicorn python-multipart aiofiles python-jose[cryptography] passlib[bcrypt] python-dotenv



# 環境設定ファイルの作成
if [ ! -f .env ]; then
    echo "Creating .env file..."
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
WorkingDirectory=$(pwd)
Environment=\"PATH=$(pwd)/venv/bin\"
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
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

# サービスの状態確認
echo "Checking services status..."
sudo systemctl status nginx --no-pager
sudo systemctl status image-classifier --no-pager

echo "S3 mount status:"
df -h | grep s3fs