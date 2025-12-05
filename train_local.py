#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本機訓練腳本 - 心臟分割
"""

import os
import sys
import subprocess
from pathlib import Path

# ============ 配置參數 ============
# 專案根目錄 = train_local.py 所在資料夾
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = str(PROJECT_ROOT)

MODEL_NAME = 'swinunetr'  # 可選: unet3d, attention_unet, cotr, unetr, swinunetr, unetcnx_a1, testnet
DATA_NAME = 'chgh'
EXP_NAME = 'AICUP_training'
DATA_DICT_FILE_NAME = 'AICUP_training.json'

# ============ 路徑設定 ============
# 設定實驗目錄
root_exp_dir = os.path.join(WORKSPACE_DIR, 'exps', 'exps', MODEL_NAME, DATA_NAME, 'tune_results')

# 設定數據目錄
root_data_dir = os.path.join(WORKSPACE_DIR, 'dataset', DATA_NAME)
data_dir = root_data_dir

# 數據字典 JSON 路徑
data_dicts_json = os.path.join(WORKSPACE_DIR, 'exps', 'data_dicts', DATA_NAME, DATA_DICT_FILE_NAME)

# 設定模型、日誌、評估目錄
model_dir = os.path.join('./', 'models')
log_dir = os.path.join('./', 'logs')
eval_dir = os.path.join('./', 'evals')

# 創建目錄
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(root_exp_dir, exist_ok=True)

# 模型路徑
best_checkpoint = os.path.join(model_dir, 'best_model.pth')
final_checkpoint = os.path.join(model_dir, 'final_model.pth')

# ============ 訓練參數 ============
TRAIN_PARAMS = {
    'tune_mode': 'train',
    'exp_name': EXP_NAME,
    'data_name': DATA_NAME,
    'data_dir': data_dir,
    'root_exp_dir': root_exp_dir,
    'model_name': MODEL_NAME,
    'model_dir': model_dir,
    'log_dir': log_dir,
    'eval_dir': eval_dir,
    'start_epoch': 0,
    'val_every': 10,
    'max_early_stop_count': 30,
    'max_epoch': 2000,
    'data_dicts_json': data_dicts_json,
    'pin_memory': True,
    'out_channels': 4,
    'patch_size': 2,
    'feature_size': 48,
    'drop_rate': 0.0,
    'depths': [2, 2, 4, 2],
    'norm_name': 'instance',
    'a_min': -42,
    'a_max': 423,
    'space_x': 0.7,
    'space_y': 0.7,
    'space_z': 1.0,
    'roi_x': 128,
    'roi_y': 128,
    'roi_z': 96,
    'optim': 'AdamW',
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'checkpoint': final_checkpoint,
    'use_init_weights': True,
    'infer_post_process': True,
        # 其他建議
    'batch_size': 1,  # 如果 GPU 記憶體不足就設為 1
}

def build_command(params, mode='train'):
    """構建訓練或測試命令"""
    cmd = [
        sys.executable,
        os.path.join(WORKSPACE_DIR, 'expers', 'tune.py')
    ]
    
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        elif isinstance(value, list):
            cmd.append(f'--{key}')
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(f'--{key}={value}')
    
    return cmd

def train():
    """執行訓練"""
    print("=" * 60)
    print("開始訓練...")
    print("=" * 60)
    
    cmd = build_command(TRAIN_PARAMS, mode='train')
    print("執行命令:")
    print(" ".join(cmd))
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n訓練完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n訓練失敗: {e}")
        sys.exit(1)

def test():
    """執行測試"""
    print("=" * 60)
    print("開始測試...")
    print("=" * 60)
    
    # 修改參數為測試模式
    test_params = TRAIN_PARAMS.copy()
    test_params['tune_mode'] = 'test'
    test_params['roi_z'] = 128  # 測試時使用 128
    test_params['resume_tuner'] = True
    test_params['save_eval_csv'] = True
    test_params['test_mode'] = True
    
    cmd = build_command(test_params, mode='test')
    print("執行命令:")
    print(" ".join(cmd))
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n測試完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n測試失敗: {e}")
        sys.exit(1)

def resume_train(checkpoint_path):
    """從檢查點繼續訓練"""
    print("=" * 60)
    print(f"從檢查點繼續訓練: {checkpoint_path}")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: 檢查點文件不存在: {checkpoint_path}")
        sys.exit(1)
    
    # 修改參數使用 SSL checkpoint
    resume_params = TRAIN_PARAMS.copy()
    resume_params['ssl_checkpoint'] = checkpoint_path
    
    cmd = build_command(resume_params, mode='train')
    print("執行命令:")
    print(" ".join(cmd))
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n繼續訓練完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n訓練失敗: {e}")
        sys.exit(1)

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='心臟分割訓練腳本')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'resume'],
                       help='執行模式: train (訓練), test (測試), resume (繼續訓練)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='繼續訓練時使用的檢查點路徑')
    
    args = parser.parse_args()
    
    # 檢查工作目錄
    if not os.path.exists(WORKSPACE_DIR):
        print(f"錯誤: 工作目錄不存在: {WORKSPACE_DIR}")
        print("請修改 WORKSPACE_DIR 變數為正確的路徑")
        sys.exit(1)
    
    # 檢查數據文件
    if not os.path.exists(data_dicts_json):
        print(f"警告: 數據字典文件不存在: {data_dicts_json}")
        print("請確保已正確設定數據路徑")
    
    # 執行對應模式
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'resume':
        if args.checkpoint is None:
            print("錯誤: 繼續訓練模式需要指定 --checkpoint 參數")
            sys.exit(1)
        resume_train(args.checkpoint)

if __name__ == '__main__':
    main()