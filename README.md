# 🥁 自動打鼓轉譜系統 (Automatic Drum Transcription - ADT)

本專案是一個基於深度學習 Onset 識別模型與「智能大腦後處理 (Brain Post-processing)」技術的自動打鼓轉譜系統。系統能夠讀取打鼓音訊，自動偵測並分離出 **底鼓 (Kick, KD)**、**軍鼓 (Snare, SD)** 與 **踩镲 (Hi-Hat, HH)** 的物理敲擊時間，並結合節奏網格、速度 (BPM) 估算與拍號偵測，導出符合樂理語意且高精度的 MIDI 轉譜檔案。

---

## 🚀 核心功能

*   **三通道 Onset 識別**：採用卷積神經網路 (CNN) 對音訊進行頻譜分析，輸出 Kick, Snare, Hi-Hat 的敲擊事件機率與時間點。
*   **智慧速度與拍號自適應估算**：
    *   **無參 BPM 估計**：利用相鄰 onset 時長的中位數及多比例映射，解決傳統自相關算法易產生的速度倍增/減半（tempo alias）問題。
    *   **複合拍號偵測**：自動識別 `4/4`、`3/4`、`6/8`、`12/8` 等多種拍號，並針對附點音符（如 12/8 附點四分音符）進行節拍對齊優化。
*   **樂理網格對齊與 GPAR 啟發式規則**：
    *   **GPAR (Hi-hat Completion & Avoidance)**：自動補全弱拍踩镲，並智慧避免與強軍鼓合奏時產生的踩镲誤判。
    *   **節奏網格恢復**：支援自適應三連音與 12/8 拍 0.75 拍 dense HH 網格恢復。
*   **多維度評估報表**：
    *   提供原生 AI 識別率 (Precision, Recall, F1-score) 與大腦後處理轉譜結果的比對報表。
    *   支援匯出 `--event-debug` CSV，用以精確校正模型輸出機率。

---

## 📦 安裝與環境配置

本專案需要 Python 3.8+ 環境，請依以下步驟安裝相依套件：

1.  **建立並啟用虛擬環境**（推薦）：
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
2.  **安裝相依套件**：
    ```powershell
    pip install -r requirements.txt
    ```

> [!NOTE]
> **模型權重檔**：
> 專案內已內建通過各項迴歸測試的最新核心模型 `best_drum_model.pth`，無需額外下載或重新訓練，即可直接執行轉譜。

---

## 🏃 快速開始

### 1. 執行音訊轉譜
使用 `transcribe.py` 對打鼓音訊進行轉譜，預設會輸出對應的 MIDI 檔，並在終端機打印轉譜評估數據：
```powershell
.\.venv\Scripts\python.exe transcribe.py path/to/your/audio.wav
```

**常用參數說明**：
*   `--sync-audio`：啟用物理時間對齊（使轉出的 MIDI 能完美對齊第一拍物理起拍）。
*   `--event-debug`：匯出 Onset 識別的詳細 CSV 事件日誌（有助於除錯與機率閥值微調）。
*   `--model_path`：指定自定義的模型路徑（預設使用 `best_drum_model.pth`）。

### 2. 執行方案驗證與迴歸測試
在修改任何轉譜邏輯或模型後，請務必執行以下驗收腳本，以確保 blind-tests 與 hard-validation 等測試指標沒有衰退：
```powershell
.\.venv\Scripts\python.exe verify_current_solution.py
```

---

## 📁 檔案結構說明

```text
├── transcribe.py                # 轉譜核心入口 (讀取音訊、執行推理與大腦後處理)
├── dsp_utils.py                 # 音訊特徵提取與信號處理工具
├── drum_plugin.py               # 提供給外部插件呼叫的轉譜接口
├── verify_current_solution.py   # 自動化迴歸測試與驗收腳本
├── requirements.txt             # 專案相依 Python 套件清單
├── best_drum_model.pth          # 目前系統所採用之最佳模型權重 (KD/SD/HH Onset)
├── train.py                     # 模型基本訓練程式
├── train_egmd.py                # 基於 E-GMD 數據集的大規模 Onset 訓練管道
├── spec.md                      # ADT 系統的詳細技術規格書與指標記錄
├── todolist.md                  # 項目開發待辦與迭代歷史記錄
├── current_status.md            # 目前 Round4 驗證與模型調優狀態報告
└── test/                        # 迴歸測試用小樣本與驗證基準 CSV 檔
```

---

## 🛡️ 貢獻與開發規範

為維護專案穩定性，開發時請嚴格遵守以下準則：
1.  **測試優先**：任何修改前應確保 `verify_current_solution.py` 通過。
2.  **安全性限制**：不得自動訓練、覆蓋已接受的 `best_drum_model.pth`、或任意刪除資料。
3.  **閥值微調建議**：靈敏度微調應移交至前端產品介面的滑塊處理，避免過度擬合物理弱音。
