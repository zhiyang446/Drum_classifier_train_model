@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"

rem D108：只執行已核准的 D89 + D54 replay + ENST 單一 epoch 候選。
if exist "validation_runs\d108_d89_enst_lora_candidate" exit /b 20
if exist "validation_runs\d108_d89_enst_lora_candidate.final.run.log" exit /b 21
if exist "validation_runs\d108_d89_enst_lora_candidate.final.err.log" exit /b 22

"%PROJECT_ROOT%\.venv\Scripts\python.exe" -u "%PROJECT_ROOT%\train_d77_fused_lora.py" ^
  --phase D108 ^
  --candidate-name d108_d89_enst_lora_adapter.pth ^
  --metadata "%PROJECT_ROOT%\mixed_d54_stem\metadata_d54.json" ^
  --extra-metadata "%PROJECT_ROOT%\enst_d107\metadata_d107_train.json" ^
  --extra-per-class 24 ^
  --init-adapter "%PROJECT_ROOT%\validation_runs\d89_d82_tim_gm_lora_retry\d89_d82_tim_gm_lora_retry_adapter.pth" ^
  --output-dir "%PROJECT_ROOT%\validation_runs\d108_d89_enst_lora_candidate" ^
  --epochs 1 ^
  --patience 1 ^
  --batch-size 4 ^
  --lr 0.001 ^
  --rank 4 ^
  --alpha 8 ^
  --seed 1337 ^
  > "%PROJECT_ROOT%\validation_runs\d108_d89_enst_lora_candidate.final.run.log" ^
  2> "%PROJECT_ROOT%\validation_runs\d108_d89_enst_lora_candidate.final.err.log"

exit /b %errorlevel%
