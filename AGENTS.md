# AGENTS.md

## 環境

- 輸出結果使用繁體中文。
- 作業系統為 Windows。
- 指令需相容 PowerShell。
- Python 優先使用 `.\.venv\Scripts\python.exe`。

## 開發前規則

- 新任務開始前先讀 `todolist.md`、`spec.md`、`current_status.md`。
- 任何修改前先更新 `spec.md`；若涉及後端 API，先更新 `api.md`，並採 RESTful 風格。
- 進行開發前先拆分任務，寫入 `todolist.md`。
- 任務開始、進行中、完成時都要更新 `todolist.md`。
- 新任務開始前需先確認 `todolist.md`，並以新的上下文處理，不沿用未驗證假設。

## 規格文件要求

`spec.md` 需覆蓋：

1. 架構與選型
2. 資料模型
3. 關鍵流程
4. 虛擬碼
5. 系統脈絡圖
6. 容器/部署概觀
7. 模組關係圖
8. 序列圖
9. ER 圖
10. 類別圖
11. 流程圖
12. 狀態圖

## 程式規範

- 程式碼需要函式級中文註解。
- 重要變數或物件需要中文註解。
- 不做無關重構；一次任務只處理一個明確目標。

## 測試與驗收

- 任務完成前必須測試，測試通過後才能進入下一任務。
- 模型或轉譜行為修改後，優先執行：

```powershell
.\.venv\Scripts\python.exe verify_current_solution.py
```

- Loop 文件修改後，至少執行：

```powershell
loop-audit.cmd . --suggest
loop-cost.cmd --pattern daily-triage --level L1
```

## Loop Engineering

- 目前只允許 L1 report-only daily triage。
- 不得自動訓練、覆蓋 checkpoint、刪除資料、push、merge 或部署。
- 任何高風險操作必須先取得人工確認。
- 詳細規則見 `LOOP.md`、`loop-constraints.md`、`docs/safety.md`。

## AI 協作接力分支策略 (Antigravity & Codex)

- **分支分配**：
  - Antigravity AI (我) 的開發分支與 push 目標為 `antigravity` 分支。
  - Codex AI 的開發分支與 push 目標為 `codex` 分支。
- **接力工作流**：
  - 當 Codex 額度用完時，會將其最新進度 commit 並 push 至 `origin/codex`。
  - Antigravity (我) 開始新任務前，必須先執行 `git fetch origin`。
  - 檢查 `origin/codex` 是否有更新的進度。若有，須經由使用者確認後，將 `origin/codex` 合併（merge）或拉取至本地的 `antigravity` 分支，再繼續開發。
  - 任何在 `antigravity` 分支上的修改 commits，在 `git push` 前皆必須取得人工確認。

## DCNN + Conformer 強制接力規則

- 本輪模型路線固定為「真正 SuperFlux → 雙分支 DCNN → DCNN+TCN 隔離對照 → 小型 Conformer」。禁止改為純 Transformer。
- 其他 AI 修改程式或啟動訓練前，必須先 `git fetch origin`，讀取 `origin/codex` 最新進度及 `spec.md`、`todolist.md`、`current_status.md`，不得沿用未驗證假設。
- 不得使用 `test_real_audio` 固定五首做訓練、選 epoch、選 threshold、特徵參數或架構選擇；只有 STAR validation/test promotion 通過後才可執行五首商業 gate。
- 不得覆蓋既有 `.pth` 或產品模型；每輪使用全新 candidate 路徑並保留 train/validation 報告。
- Codex 每個 Phase 必須完成文件更新與規定測試後，才可 commit 並 push 至 `origin/codex`；失敗 Phase 也要提交失敗證據與 blocker，但不得標記為完成。
- 其他 AI 不得自行改變架構順序、資料隔離、訓練配方或驗收門檻。若有 materially different 提案，先寫入規格與證據並取得使用者明確確認。
