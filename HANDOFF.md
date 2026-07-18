# ADT 專案 AI 協作接力手冊 (HANDOFF.md)

本文件為 **Antigravity** 與 **Codex** 兩大 AI 協作開發的接力交接手冊，詳細記錄當前的系統狀態、已完成里程碑、安全防線、下一步方向與開發防踩坑指南。

---

## 1. 現在做什麼？ (Current Focus)
- **當前焦點**：當前已完璧完成 **`D7-A_opt` 閾值解碼校正版本** 的定版與發布（KD=0.50，其餘 5 類為優化值）。
- **系統狀態**：整個系統在 `A_opt` 下，於未見的 STAR test split 上 Macro F1 提振了 `+0.0087`，大鼓 KD 完全防守無退步，Round 4 實體驗收通過率維持最高 `35/36`。且三類發布模型回歸測試 100% 綠燈通過。
- **分支與推送**：最新代碼、定版 JSON 與治理規範已全部 push 至遠端 `antigravity` 分支。

---

## 2. 已經完成了什麼？ (Accomplishments)
- **Phase D12-A (多解析度 Log-Mel)**：完成了多解析度特徵融合的背景訓練與評估，判定其單單融合多解析度特徵無法在不加權的情況下挽救罕見鈸類，已完成並拒絕。
- **Phase D13 (D7 後處理優化與消融)**：
  - 在 validation 集合上進行了 6 類 Onset 解碼閾值的 Coordinate Ascent 尋優。
  - 執行了 A0 到 A6 的 7 組**單類別消融實驗 (Single-class Ablation Study)**，精確將大鼓 KD 閾值拉高至 0.60 定位為引發漏檢退步的唯一根源。
  - 合併並驗證了「五類優化、大鼓 KD 維持 0.50 基線」的 `A_opt` 配置，完美在封存測試集與 Round 4 上通過驗收，成功晉級上線。
- **治理規範落實**：正式在 `AGENTS.md` 與 `spec.md` 中確立了 **「推論解碼校正與版本發布治理規範」**。

---

## 3. 卡在哪裡？ (Blockers & Ambiguity)
- **無實質 Blockers**：本階段尋優與校正定版已完美閉環，無技術或工程阻礙。
- **潛在難點**：鈸類與罕見鼓件（TOM, CRASH, RIDE）由於資料庫極度不平衡，在不重訓模型的情況下，單純調整後處理閾值只能發揮防抖（減少假陽性）的作用，無法在物理上提高其檢測召回率 (Recall)。

---

## 4. 下一步做什麼？ (Next Steps)
1. **D12-C 方案（雙劍合璧）**：
   - 如果下一輪要重啟訓練突破性能，應執行 **D12-C 方案（多解析度特徵融合 + Class-Balanced BCE 平滑訓練）**。
   - 這能從網路底層（特徵與 Loss）徹底解決稀有鼓件的梯度不平衡，大幅提振鈸類 Recall。
2. **新歌 Shadow Run 觀測**：
   - 繼續對用戶回饋的真實打鼓音軌進行 shadow run，若發現 A_opt 表現退步，一秒帶上 `--rollback-baseline` 參數退回 A0。

---

## 5. 那些坑不要再踩？ (Anti-Patterns & Lesson Learned)

> [!CAUTION]
> **以下為高壓紅線與實踐血淚史，後續 AI 開發必須無條件迴避：**

1. **嚴禁對盲測 / 驗收歌曲進行過擬合特化 (No Blind Overfitting)**：
   - 絕對不要因為 Blind Tests 或 Round 4 的歌曲結果，回過頭來微調或挑選特定的閾值。**Validation 只用來尋優；Test 只用來一筆畫驗收**，否則模型將喪失泛化性。
2. **嚴禁無休止在解碼層調參 (Stop Param-sweeping on Decoder)**：
   - 當稀有鼓件出現召回不足時，**解法在於收集數據與重訓，而非繼續尋找新 threshold**。過高的 threshold 會扼殺真陽性（如大鼓漏檢）。
3. **加載 Checkpoint 時必須帶類別防禦隔離 (Class-count Defense)**：
   - 熱插拔 JSON 等配置時，必須用 `num_classes > 3` 進行隔離，確保三類產品模型（`drum_classifier.pth`）在 `verify_current_solution.py` 中 100% 維持原 MGPC 邏輯，防止代碼重構引發致命 regression。
4. **子進程崩潰會掩蓋真實評分 (Subprocess Crash Masking)**：
   - 當 `transcribe.py` 因模型鍵值不相容崩潰時，`run_egmd_round4_validation.py` 可能會因為讀取磁碟上殘留的舊預測 CSV 檔案而報告 `PASS`。
   - **防踩坑**：在評估前必須手動或程式清理 output 目錄，或在 `subprocess.run` 中帶上 `check=True` 捕獲 exit code。
