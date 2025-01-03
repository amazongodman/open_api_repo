# open_api_repo


Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)  

インスタンスタイプ: g4dn.xlarge  
AMI: Ubuntu Server 22.04 LTS  
ストレージ: 100GB gp3  
セキュリティグループ設定:  


キーペアファイル(.pem)を安全な場所に保存  
マイIPを選択  
EC2を起動、PuTTYでログインできるようにしておくこと  




# EC2インスタンスにIAMロールを設定


EC2 → インスタンス → 該当のインスタンスを選択  

アクション → セキュリティ → IAMロールを変更  

ロールを作成  
信頼されたエンティティタイプで「AWS のサービス」を選択  
ユースケースで「EC2」を選択  
許可ポリシーで「AmazonS3FullAccess」  


アクション → セキュリティ → IAMロールを変更  


```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name/*",
                "arn:aws:s3:::your-bucket-name"
            ]
        }
    ]
}
```


# 確認
nvidia-smi  # GPUの確認  

```
Tue Dec 24 12:42:50 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       On  |   00000000:00:1E.0 Off |                    0 |
| N/A   23C    P8              9W /   70W |       1MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```



nvcc -V     # CUDAバージョンの確認  

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```



# repo構成

your-repo/  
├── main.py  
├── setup.sh  
└── README.md  

# 1. EC2インスタンスに接続後、gitリポジトリをクローン
git clone https://github.com/YOUR_USERNAME/open_api_repo.git  
cd open_api_repo  

# setup.shの修正

.shの中身を編集  
S3_BUCKET="your-dataset-bucket-name"  # このバケット名は要変更  

起動後にEC2コンソールから  
aws s3 ls s3://your-dataset-bucket-name/  
が通って中身が表示されたらOK  

# 2. セットアップスクリプトに実行権限を付与
chmod +x setup.sh  

# 3. セットアップスクリプトを実行
./setup.sh  

# 4. インストール後の確認
nvidia-smi  # GPUの確認  
nvcc -V     # CUDAバージョンの確認  



source venv/bin/activate  

pip list  # torchがはいっているはず


python3 -c "import torch; print(torch.cuda.is_available())"  # PyTorch CUDA確認  
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"  

# 5. アプリケーションの状態確認
sudo systemctl status image-classifier  
sudo systemctl status nginx  
sudo journalctl -u image-classifier -f  






インスタンスのセキュリティグループのインバウントルールにHTTP、ポート80、マイIPを追加  

ブラウザで http://your-ec2-public-ip にアクセス  
Basic認証の要求が表示される  


認証時の入力:  
- ユーザー名: 何を入力しても構いません（空欄でもOK）  
- パスワード: .envファイルに設定したPASSWORDの値  

デフォは「change_this_password」にした  

セキュリティのため、本番環境では必ずデフォルトパスワードを変更してください。  

# 推論

画像ダウンロード  
wget https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg  



![image](https://github.com/user-attachments/assets/34750786-688a-48eb-8a8f-c92b1a33364d)





## サービスを再起動

サービスはctrl+cで停止  
main.pyなどを書き換えたら以下で再起動をする  
sudo systemctl restart image-classifier  
sudo systemctl restart nginx  
sudo journalctl -u image-classifier -f  

pythonの仮想環境もはずれたらなこれ  
source venv/bin/activate  

再起動  
sudo reboot  

## ディスク容量の確認
df -h  

