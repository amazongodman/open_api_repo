# open_api_repo


Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)  

インスタンスタイプ: g4dn.xlarge  
AMI: Ubuntu Server 22.04 LTS  
ストレージ: 100GB gp3  
セキュリティグループ設定:  


キーペアファイル(.pem)を安全な場所に保存  
マイIPを選択  
EC2を起動、PuTTYでログインできるようにしておくこと  


# repo構成

your-repo/  
├── main.py  
├── setup.sh  
└── README.md  

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
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"  

# 5. アプリケーションの状態確認
sudo systemctl status image-classifier  
sudo systemctl status nginx  

# 6. ログの確認
sudo journalctl -u image-classifier -f  

ブラウザで http://your-ec2-public-ip にアクセス  
Basic認証の要求が表示される  


認証時の入力:  
- ユーザー名: 何を入力しても構いません（空欄でもOK）  
- パスワード: .envファイルに設定したPASSWORDの値  

change_this_password  



### セキュリティのため、本番環境では必ずデフォルトパスワードを変更してください。パスワードの変更は以下のように行えます：


## .envファイルのPASSWORDを変更
nano ~/your-repo/.env  

## サービスを再起動
sudo systemctl restart image-classifier  



## 5.1 サービスが起動しない場合
sudo journalctl -u image-classifier -f  
cat ~/your-repo/.env  
cd ~/your-repo  
source venv/bin/activate  
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000  



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