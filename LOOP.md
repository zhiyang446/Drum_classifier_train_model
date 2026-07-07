# LOOP.md

## Active Loops

| Pattern | Cadence | Status | Command |
|---------|---------|--------|---------|
| daily-triage | 手動或每日最多 1-2 次 | L1 report-only | `$loop-triage` |

## Scope

Daily triage 只做狀態整理、風險提示與驗證建議。它可以讀取文件、執行唯讀檢查、更新 `STATE.md` 與 `loop-run-log.md`，但不得自行修改模型邏輯或啟動訓練。

## Required Reads

1. `STATE.md`
2. `current_status.md`
3. `todolist.md`
4. `spec.md`
5. `loop-constraints.md`

## Human Gates

- 訓練、微調、覆蓋或新增 checkpoint。
- 刪除資料集、驗證輸出、音訊或 checkpoint。
- 安裝依賴、下載資料、啟動長時間工作。
- `git push`、PR、merge、部署。
- 修改 `transcribe.py`、訓練腳本或驗證 gate。

## Verification

Loop scaffold 或文件變更後：

```powershell
loop-audit.cmd . --suggest
loop-cost.cmd --pattern daily-triage --level L1
```

模型或轉譜行為變更後：

```powershell
.\.venv\Scripts\python.exe verify_current_solution.py
```

## Budget

- Daily cap: `100k` tokens.
- Max runs/day: `2`.
- Max sub-agent spawns/run: `0` for L1.
- Kill switch: `loop-pause-all` in `STATE.md`.

## MCP

MCP is not required for the current L1 report-only loop. Add connector configuration only when a future GitHub/CI loop needs read-only external context.

## Worktree Policy

此工作區目前 `git status` 回報不是有效 Git repo；Loop 不得假設可 commit 或開 PR。修復 Git 狀態前，所有 loop 只保留本地文件與報告。
