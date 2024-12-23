# open_api_repo


1. インスタンスタイプ: g4dn.xlarge
2. ストレージ: 最低40GB（PyTorch + CUDA関連で約20GB必要）
3. セキュリティグループ: 
   - SSH (22)
   - HTTP (80)
4. AMI: Ubuntu 22.04 LTS



EC2を起動、PuTTYでログインできるようにしておくこと。  


# 1. EC2インスタンスに接続後、gitリポジトリをクローン
git clone https://github.com/YOUR_USERNAME/open_api_repo.git
cd open_api_repo

# 2. セットアップスクリプトに実行権限を付与
chmod +x setup.sh

# 3. セットアップスクリプトを実行
./setup.sh

# 4. インストール後の確認
nvidia-smi  # GPUの確認
nvcc -V     # CUDAバージョンの確認
python3 -c "import torch; print(torch.cuda.is_available())"  # PyTorch CUDA確認

# 5. アプリケーションの状態確認
sudo systemctl status image-classifier
sudo systemctl status nginx

# 6. ログの確認
sudo journalctl -u image-classifier -f

# トラブルシューティング用コマンド



## ディスク容量の確認
df -h

## GPU状態の確認
nvidia-smi
nvidia-smi -l 1  # 1秒ごとに更新

## CUDA情報の確認
nvcc --version

## サービスの再起動
sudo systemctl restart image-classifier
sudo systemctl restart nginx

## ログの確認
tail -f /var/log/setup_script.log
sudo journalctl -u image-classifier -f