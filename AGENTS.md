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

