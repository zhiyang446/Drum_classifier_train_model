@echo off
REM 中文註解：只啟動已核准的 D37 固定配額候選，輸出留在新的 retry 目錄。
set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"
"%PROJECT_ROOT%\.venv\Scripts\python.exe" -u "%PROJECT_ROOT%\train_six_class_candidate.py" --meta "%PROJECT_ROOT%\mixed_d36\metadata_d36.json" --checkpoint "%PROJECT_ROOT%\mixed_formal_kick375_snare18_hh12_candidate.pth" --output-dir "%PROJECT_ROOT%\validation_runs\d37_mixed_real_first_retry_dcnn_tcn_conformer" --per-class 400 --batch-size 4 --lr 0.0005 --epochs 5 --architecture dcnn-tcn-conformer --feature-mode true-superflux --validation-meta "%PROJECT_ROOT%\mixed_d36\metadata_d36.json" --validation-per-class 8 --source-quota-profile d37-real-first --freeze-bn --log-every 25 --candidate-name d37_mixed_real_first_candidate.pth 1>"%PROJECT_ROOT%\validation_runs\d37_mixed_real_first_retry_dcnn_tcn_conformer\training_stdout.log" 2>"%PROJECT_ROOT%\validation_runs\d37_mixed_real_first_retry_dcnn_tcn_conformer\training_stderr.log"
