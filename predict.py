#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import subprocess
import importlib
from pathlib import PurePath,Path

import pandas as pd
from ray import tune
from ray.train.trainer import BaseTrainer
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

WORKSPACE_DIR = str(PROJECT_ROOT)

# set package path
# 確保 CardiacSegV2 專案在 Python 的搜尋路徑中
sys.path.append(WORKSPACE_DIR) 

# ==================== 錯誤修正 ====================
# Ray Tune restore 需要原始的 'trainable' 函數物件，而不是字串
# 我們從 expers.tune 模組中匯入 main 函數
try:
    from expers.tune import main
except ImportError:
    print("錯誤：無法從 'expers.tune' 匯入 'main' 函數。")
    print("請確認 sys.path.append 的路徑是正確的。")
    sys.exit(1)
# ==================== 修正結束 ====================


def get_tune_model_dir(root_exp_dir, exp_name):
    experiment_path = os.path.join(root_exp_dir, exp_name)
    print(f"Loading results from {experiment_path}...")
    
    # 不提供 trainable 參數
    restored_tuner = tune.Tuner.restore(experiment_path)
    
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric="tt_dice", mode="max")
    
    print(f"\nBest trial {best_result.metrics['trial_id']}: ")
    print('config:', best_result.metrics['config'])
    print('tt_dice:', best_result.metrics['tt_dice'])
    if 'esc' in best_result.metrics:
        print('esc:', best_result.metrics['esc'])
    print(f'best log dir:', best_result.log_dir)
    
    model_dir = os.path.join(best_result.log_dir, 'models')
    return model_dir


# 輸出資料夾統一放在專案底下：./output/chgh/image, ./outputs/chgh/infer
OUTPUT_ROOT = PROJECT_ROOT / "output" / "chgh"
IMAGE_FOLDER = OUTPUT_ROOT / "image"
INFER_FOLDER = OUTPUT_ROOT / "infer"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(INFER_FOLDER, exist_ok=True)

image_folder = str(IMAGE_FOLDER)
infer_dir = str(INFER_FOLDER)

# ============ 配置參數 ============
# 請根據你的環境修改這些路徑
MODEL_NAME = 'swinunetr'  # 可選: unet3d, attention_unet, cotr, unetr, swinunetr, unetcnx_a1, testnet
DATA_NAME = 'chgh'
EXP_NAME = 'AICUP_training'
DATA_DICT_FILE_NAME = 'AICUP_training_sim.json'

# ============ 路徑設定 ============

# set exp dir
root_exp_dir = os.path.join(WORKSPACE_DIR,'exps','exps',MODEL_NAME,DATA_NAME,'tune_results')

# set data dir
root_data_dir = os.path.join(WORKSPACE_DIR,'dataset',DATA_NAME)
#data_dir = os.path.join(root_data_dir, sub_data_dir_name)
data_dir = os.path.join(root_data_dir)
# data dict json path
data_dicts_json = os.path.join(WORKSPACE_DIR, 'exps', 'data_dicts', DATA_NAME, DATA_DICT_FILE_NAME)

# set model, log, eval dir
# 這些是 'infer.py' 執行時可能需要的相對路徑
model_dir = os.path.join('./', 'models')
log_dir = os.path.join('./', 'logs')
eval_dir = os.path.join('./', 'evals')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# mkdir root exp dir
os.makedirs(root_exp_dir, exist_ok=True)


# --- 獲取最佳模型路徑 ---
print("--- 正在從 Ray Tune 獲取最佳模型路徑 ---")
model_dir_from_tune = get_tune_model_dir(root_exp_dir, EXP_NAME)
best_checkpoint = os.path.join(model_dir_from_tune, 'best_model.pth')
print(f"--- 找到最佳權重檔: {best_checkpoint} ---")

pred_img = []
#image_folder = image_folder
print(f"--- 正在從 {image_folder} 搜尋圖片 ---")
for root, dirs, files in os.walk(image_folder, topdown=False):
  for name in files:
    img_path = os.path.join(root, name)
    pred_img.append(img_path)
    print(f"找到: {img_path}")

# 執行推論
success_count = 0
fail_count = 0

print(f"\n--- 開始執行推論 ({len(pred_img)} 張圖片) ---")
for i, img_pth in enumerate(pred_img, 1):
    print(f"[{i}/{len(pred_img)}] 推論: {os.path.basename(img_pth)}")
    print("-" * 60)
    
    cmd = [
        sys.executable, # 使用當前 Python 環境
        os.path.join(WORKSPACE_DIR, 'expers', 'infer.py'), # infer.py 的絕對路徑
        '--model_name', MODEL_NAME,
        '--data_name', DATA_NAME,
        '--data_dir', data_dir,
        '--model_dir', model_dir_from_tune, # 使用從 tune 拿到的模型路徑
        '--infer_dir', infer_dir,
        '--checkpoint', best_checkpoint, # 使用最佳權重
        '--img_pth', img_pth, # 當前圖片路徑
        '--out_channels', '4',
        '--patch_size', '2',
        '--feature_size', '48',
        '--drop_rate', '0.0',
        '--depths', '2', '2', '4', '2',
        '--kernel_size', '5',
        '--exp_rate', '4',
        '--norm_name', 'instance',
        '--a_min', '-42',
        '--a_max', '423',
        '--space_x', '0.7',
        '--space_y', '0.7',
        '--space_z', '1.0',
        '--roi_x', '128',
        '--roi_y', '128',
        '--roi_z', '96', # 測試時為 128
        '--infer_post_process'
    ]
    
    try:
        # 重要：將工作目錄(cwd)設為 WORKSPACE_DIR
        # 這能確保 'infer.py' 內部的相對路徑 (例如 import) 能正常運作
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=WORKSPACE_DIR # <-- 在 CardiacSegV2 目錄下執行
        )
        print("✓ 成功")
        # print(result.stdout) # (可選) 印出詳細日誌
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"✗ 失敗")
        print(f"錯誤: {e}")
        if e.stderr:
            print(f"詳細錯誤: {e.stderr}")
        fail_count += 1

# 顯示結果
print("\n" + "=" * 60)
print("推論完成！")
print(f"成功: {success_count}/{len(pred_img)}")
print(f"失敗: {fail_count}/{len(pred_img)}")
print(f"推論結果已儲存於: {infer_dir}")
print("=" * 60)

# --- 壓縮結果 ---
try:
    zip_target_dir = infer_dir
    zip_path_out = infer_dir + "/contest1" # 壓縮檔將存為 contest1.zip
    
    print(f"正在將 {zip_target_dir} 壓縮至 {zip_path_out}.zip ...")
    
    shutil.make_archive(
        base_name=zip_path_out,
        format='zip',
        root_dir=infer_dir
    )
    print(f"✓ 壓縮完成: {zip_path_out}.zip")
except Exception as e:
    print(f"✗ 壓縮失敗: {e}")

print("--- 腳本執行完畢 ---")