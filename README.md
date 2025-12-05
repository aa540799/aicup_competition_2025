# AICUP 2025 心臟 CT 影像分割
本專案參加 **AICUP 心臟影像分割競賽**，目標是針對心臟 CT 影像進行三維語意分割，預測多個心臟結構的標籤。  
整體流程分為：

- CT 影像前處理（重採樣、裁切、強度正規化）
- 使用 MONAI 建立 3D U-Net / SwinUNETR 模型
- 透過 `Ray Tune` 做實驗管理與訓練
- 輸出 NIfTI segmentation 以及 CSV 結果，用於提交競賽

> ⚠️ 注意：**原始競賽資料集不隨專案釋出**，請自行至主辦單位平台下載，並依照下文「資料放置方式」整理目錄。

---

## 1. 專案結構 (Project Structure)

專案根目錄大致結構如下（省略部分檔案）：

```text
aicup_competition/
├── train_local.py                 # 本機訓練入口程式（包一層參數、呼叫 expers/tune.py）
├── predict.py                     # 預測 / 產生提交檔的腳本
├── expers/
│   ├── args.py                    # 參數 parser 與 config 映射
│   ├── train.py                   # 單次訓練流程
│   ├── tune.py                    # 使用 Ray Tune 做實驗管理與訓練入口
│   ├── infer.py                   # 推論流程相關程式
│   └── ...
├── exps/
│   ├── data_dicts/
│   │   └── chgh/
│   │       └── AICUP_training.json   # 訓練 / 驗證資料分割設定
│   └── exps/
│       └── SwinUNETR/
│           └── chgh/
│               └── tune_results/     # 訓練過程產生的 log 與 checkpoint (gitignore)
├── dataset/
│   └── chgh/                      # 放競賽提供的 CT 與 label (不會 push 上 GitHub)
├── models/                        # 儲存最佳 / 最終模型權重 (gitignore)
├── output/
│   └── chgh/
│       ├── image/                 # 推論輸出之 NIfTI segmentation
│       └── infer/                 # 產生提交用 CSV 等
├── requirements.txt               # 專案依賴套件 (可選)
└── README.md
