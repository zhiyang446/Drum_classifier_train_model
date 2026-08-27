# STATE.md

Last run: 2026-07-26
Pattern: daily-triage
Level: L1 report-only
Kill switch: off

## High Priority

- Loop scaffold 已建立，`loop-audit.cmd . --suggest` 通過：Score `100/100`, Level `L3`。
- D90 已確認 D89 為 mixed regression，狀態為 `stop_same_data`；不再重建 D89、做同資料 LoRA 或 threshold sweep。
- D80 只讀盤點完成；候選清理項目須等人工確認，工作區目前有既有 dirty/untracked 變更，保持不動。

## Watch List

- Accepted checkpoint: `mixed_formal_kick375_snare18_hh12_candidate.pth`
- Accepted verifier: `.\.venv\Scripts\python.exe verify_current_solution.py`
- Round5 root cause: unsupported Ride/Crash/Tom proxying dominates the held-out HH error; the next planned path is a separate six-class candidate, while L1 remains report-only.
- D54 stem chain: `drumsep_d48`、`drumsep_d52`、`drumsep_d53` 與 `synthetic_midi_archive_d27` 保留；D54 manifest 或 hard link 仍依賴它們。
- 新音訊失敗時，先分 raw acoustic/model layer 與 brain/notation layer。

## Recent Noise

- `loop-cost.cmd --pattern daily-triage --level L1` 顯示預設 12 runs/day 的 realistic blend 約 `276k/day`，超過 `100k/day` 建議上限。

## Next Action

先取得人工確認，再決定是否清理 `__MACOSX`、`__pycache__`、D47 smoke input/output；保留 D47 audit。維持 L1 report-only，不刪除、不訓練、不啟用 auto-fix。
