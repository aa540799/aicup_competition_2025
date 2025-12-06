# AICUP 2025 心臟 CT 影像分割
本專案參加 **AICUP 心臟影像分割競賽**，目標是針對心臟 CT 影像進行三維語意分割，預測多個心臟結構的標籤。  
整體流程分為：

- CT 影像前處理（重採樣、裁切、強度正規化）
- 使用 MONAI 建立 3D U-Net / SwinUNETR 模型
- 透過 `Ray Tune` 做實驗管理與訓練
- 輸出 NIfTI segmentation 以及 CSV 結果，用於提交競賽

>注意：**原始競賽資料集不隨專案釋出**，請自行至主辦單位平台下載，並依照下文「資料放置方式」整理目錄。

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

(3) 執行安裝腳本：
```bash
chmod +x setup.sh
./setup.sh
```

執行後須使用套件即可安裝完畢

## 4資料準備（Data Preparation）
本專案不附任何影像或標註檔。
使用者需自行依競賽規則向主辦單位取得資料集。

（1）將0001～0050的**影像**及**標註檔**放置資料到 dataset/chgh資料夾底下，結構如下
```text
dataset/
└── chgh/
    ├── patient0001.nii.gz
    ├── patient0001_label.nii.gz
    ├── patient0002.nii.gz
    ├── patient0002_label.nii.gz    
    └── ...
```

實際命名與對應關係需與 exps/data_dicts/chgh/AICUP_training.json 中設定一致。

（1）設定 Data Dict（AICUP_training.json）