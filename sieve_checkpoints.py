import os
import sys
import json
import argparse
import subprocess
import shutil

def main():
    """中文註解：自動遍歷與篩選各 Epoch Checkpoints 的回歸測試，挑選出 F1-Score 最高的最佳模型。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='訓練輸出目錄，如 validation_runs/six_class_candidate_v15')
    parser.add_argument('--candidate-base', default='six_class_candidate_v15', help='基本檔案名稱')
    parser.add_argument('--epochs', type=int, default=20, help='訓練的總 Epoch 數')
    args = parser.parse_args()

    best_f1 = -1.0
    best_epoch = None
    results = []

    print(f"==================================================", flush=True)
    print(f"Starting Automated Checkpoint Sieve for V15", flush=True)
    print(f"==================================================", flush=True)

    python_executable = sys.executable

    for epoch in range(1, args.epochs + 1):
        model_name = f"{args.candidate_base}_epoch{epoch}.pth"
        model_path = os.path.join(args.output_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Epoch {epoch}: Checkpoint not found at {model_path}, skipping.", flush=True)
            continue

        print(f"\n--- Sifting Epoch {epoch}/{args.epochs} ({model_name}) ---", flush=True)
        
        # 步驟一：執行三類別回歸測試
        reg_output_dir = os.path.join(args.output_dir, f"sieve_reg_epoch{epoch}")
        reg_cmd = [
            python_executable, "verify_current_solution.py",
            "--model", model_path,
            "--output-dir", reg_output_dir
        ]
        
        print(f"Running regression check...", flush=True)
        reg_res = subprocess.run(reg_cmd, capture_output=True, text=True)
        
        if reg_res.returncode != 0:
            print(f"Result: FAIL (Regression detected!)", flush=True)
            results.append({
                'epoch': epoch,
                'model': model_name,
                'regression': 'FAIL',
                'macro_f1': 0.0,
                'eligible': False
            })
            # 刪除暫存的 reg 輸出目錄以節省空間
            if os.path.exists(reg_output_dir):
                shutil.rmtree(reg_output_dir)
            continue
        
        print(f"Result: PASS (No regression!)", flush=True)
        # 刪除暫存的 reg 輸出目錄
        if os.path.exists(reg_output_dir):
            shutil.rmtree(reg_output_dir)

        # 步驟二：執行六類別 STAR 測試集驗證，獲取 macro F1
        val_output_dir = os.path.join(args.output_dir, f"sieve_val_epoch{epoch}")
        val_cmd = [
            python_executable, "run_six_class_validation.py",
            "--meta", "processed_data/star_meta.json",
            "--model", model_path,
            "--output-dir", val_output_dir
        ]
        
        print(f"Running six-class validation...", flush=True)
        val_res = subprocess.run(val_cmd, capture_output=True, text=True)
        
        # 清理 val 暫存目錄
        if os.path.exists(val_output_dir):
            shutil.rmtree(val_output_dir)

        macro_f1 = 0.0
        try:
            val_data = json.loads(val_res.stdout)
            macro_f1 = val_data.get('gate', {}).get('macro_f1', 0.0)
        except Exception as e:
            print(f"Warning: Failed to parse validation JSON output: {e}", flush=True)

        print(f"Result: PASS | STAR Macro F1 = {macro_f1:.4f}", flush=True)
        
        results.append({
            'epoch': epoch,
            'model': model_name,
            'regression': 'PASS',
            'macro_f1': macro_f1,
            'eligible': True
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch

    # 輸出最終篩選總覽
    print("\n" + "="*50, flush=True)
    print("Sieve Summary Table", flush=True)
    print("="*50, flush=True)
    print(f"{'Epoch':<6} | {'Model Checkpoint':<28} | {'Regression Check':<16} | {'STAR Macro F1':<13} | {'Status':<10}", flush=True)
    print("-"*80, flush=True)
    for res in results:
        status_str = "Eligible" if res['eligible'] else "Discarded"
        f1_str = f"{res['macro_f1']:.4f}" if res['eligible'] else "N/A"
        print(f"{res['epoch']:<6} | {res['model']:<28} | {res['regression']:<16} | {f1_str:<13} | {status_str:<10}", flush=True)
    print("="*80, flush=True)

    if best_epoch is not None:
        best_model_name = f"{args.candidate_base}_epoch{best_epoch}.pth"
        best_model_path = os.path.join(args.output_dir, best_model_name)
        target_best_path = os.path.join(args.output_dir, f"{args.candidate_base}_best.pth")
        
        shutil.copyfile(best_model_path, target_best_path)
        print(f"🎉 Promotion Success! Sieve selected Epoch {best_epoch} as the best model.", flush=True)
        print(f"Copied {best_model_path} -> {target_best_path}", flush=True)
        print(f"Promoted Best STAR Macro F1: {best_f1:.4f}", flush=True)
    else:
        print("❌ Sieve Failure: No checkpoint passed the regression check!", flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
