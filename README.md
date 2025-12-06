# AICUP 2025 心臟 CT 影像分割
本專案參加 **AICUP 心臟影像分割競賽**，目標是針對心臟 CT 影像進行三維語意分割，預測多個心臟結構的標籤。  

注意：**原始競賽資料集不隨專案釋出**，請自行至主辦單位平台下載，並依照下文「資料放置方式」整理目錄。

## 1.取得專案 (Clone Repository)
```bash
git clone https://github.com/aa540799/aicup_competition_2025.git
```

## 2.環境安裝 (Environment Setup)
使用**miniconda** 安裝python3.10

```bash
conda create -n aicup_env python=3.10 -y
```
## 3.使用 setup.sh 一次安裝主要套件
(1) 確認已在專案根目錄：
```bash
cd aicup_competition_2025
```

(2) 確認已啟用對應的 conda 環境
```bash
conda activate aicup_env
```
(3) 執行安裝腳本：
```bash
./setup.sh
```

執行後須使用套件即可安裝完畢

## 4.資料準備（Data Preparation）
本專案不附任何影像或標註檔。
使用者需自行依競賽規則向主辦單位取得資料集。

（1）建立訓練資料存放資料夾
```bash
cd aicup_competition_2025
mkdir -p dataset/chgh
```
（2）將0001～0050的**影像（nii.gz）**及**標註檔（_gt.nii.gz）**放置資料到 dataset/chgh資料夾底下，結構如下
```text
dataset/
└── chgh/
    ├── patient0001.nii.gz
    ├── patient0001_gt.nii.gz
    ├── patient0002.nii.gz
    ├── patient0002_gt.nii.gz    
    └── ...
```

（3）建立推論資料存放資料夾
```bash
cd aicup_competition_2025
mkdir -p output/chgh
cd output/chgh
mkdir -p image infer
```
（4）將需要推論的0051～0100的**影像檔（nii.gz）**放置資料到 output/chgh/image資料夾底下，結構如下
```text
output/
└── chgh/
    └── image/
        ├── patient0051.nii.gz
        ├── patient0052.nii.gz
        ├── patient0053.nii.gz
        ├── patient0054.nii.gz  
        └── ...
```


實際命名與對應關係需與 exps/data_dicts/chgh/AICUP_training.json 中設定一致。

（2）設定 Data Dict（AICUP_training.json）
資料/驗證/訓練比例為 40/7/3

## 5.訓練流程（Data Preparation）
請確定已在conda環境中（aicup_env）如果還未在環境中請先使用
```bash
conda activate aicup_env
```

使用train_local.py
```bash
cd aicup_competition_2025
python train_local.py 
```

產生的權重檔會放在 exps/exps/swinunetr/chgh/tune.results/AICUP_training資料夾底下

## 6.推論與結果產生流程（Inference & Submission）
請確定已在conda環境中（aicup_env）如果還未在環境中請先使用
```bash
conda activate aicup_env
```

使用predict.py
```bash
cd aicup_competition_2025
python predict.py
```

推論後的結果會放到output/chgh/infer底下，並將所有檔案包成一個壓縮檔contest1.zip(檔案也放置於infer底下)用於送交競賽平台的結果檔

