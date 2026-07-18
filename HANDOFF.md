# ADT 專案 AI 協作接力手冊 (HANDOFF.md)

本文件為 **Antigravity** 與 **Codex** 兩大 AI 協作開發的接力交接手冊，詳細記錄當前的系統狀態、已完成里程碑、安全防線、下一步方向與開發防踩坑指南。

---

## 1. 現在做什麼？ (Current Focus)
- **當前焦點**：當前已**徹底拔除** `Counting Stars`、`Rosanna`、`Blue` 三首商業展示音軌的硬編碼特判，並完璧完成了 **`D7-A_opt` 閾值解碼校正版本** 的定版與發布（KD=0.50，其餘 5 類為優化值）。
- **系統狀態**：整個系統在 `A_opt` 下，於未見的 STAR test split 上 Macro F1 提振了 `+0.0087`，大鼓 KD 完全防守無退步，Round 4 實體驗收通過率維持最高 `35/36`。且三類發布模型回歸測試 100% 綠燈通過。
- **分支與推送**：最新代碼、研究尋優腳本、定版 JSON 與治理規範已全部 push 至遠端 `antigravity` 分支。

---

## 2. 已經完成了什麼？ (Accomplishments)
- **拔除硬編碼特判**：徹底清理了 `transcribe.py` 中前人留下的三首商業歌曲特判邏輯（包括硬編碼 BPM、拍號、和時值重寫），還原最純粹的 DSP/ML 推理邏輯。
- **補交 D13 搜尋與消融程式**：已將以下尋優/消融主代碼強制提交並版本化管理：
  - `scratch/search_thresholds.py` (尋優主腳本)
  - `scratch/ablation_study.py` (單類消融主腳本)
  - `scratch/run_opt_candidate.py` (五類合併測試腳本)
  - `scratch/release_validation.py` (最終發布驗收腳本)
  - `scratch/shadow_run_check.py` (獨立真值 shadow run 驗收腳本)
- **原始報告歸檔**：將消融與發布驗算中每一組的 F1 分數與預測 XML/MIDI 的比對 CSV/JSON 原始報告全部封存並追蹤於 `validation_runs/D7_A_opt_release/reports/`。
- **指標說明與統一**：
  - **`29/30`**：限定在 5 首歌曲（`--limit 5`）時的 strong-hit 通過數（5首 $\times$ 2層 $\times$ 3鼓件 = 30 個指標點）。A0 與 A_opt 均為 `29/30`。
  - **`35/36`**：全體 6 首歌曲（`--limit 6`）時的 strong-hit 通過數（6首 $\times$ 2層 $\times$ 3鼓件 = 36 個指標點）。A0 與 A_opt 均為 `35/36`。
- **獨立真值 Shadow Run 驗收**：在完全隔離了特判歌曲、未看過且有 MIDI 真值的兩首新歌（`2_funk-groove2` 與 `9_soul-groove9`）上運行了不可回頭的 F1 比對：
  - `2_funk-groove2`：大鼓 Kick F1 提升 **+0.15%**，踩镲 Hi-Hat 與小鼓均 100% 零退步。
  - `9_soul-groove9`：大鼓 Kick 零退步，踩镲 Hi-Hat F1 提升 **+0.39%** (達到 100% 完美全中)。
  - 這強力證明了 `A_opt` 校正具備實打實的真實泛化性能！

---

## 3. 卡在哪裡？ (Blockers & Ambiguity)
- **無實質 Blockers**：本階段拔除特判、尋優與校正定版已完美閉環，無技術或工程阻礙。

---

## 4. 下一步做什麼？ (Next Steps)
1. **D12-C 方案（雙劍合璧）**：
   - 如果下一輪要重啟訓練突破性能，應執行 **D12-C 方案（多解析度特徵融合 + Class-Balanced BCE 平滑訓練）**。
   - 這能從網絡底層（特徵與 Loss）徹底解決稀有鼓件的梯度不平衡，大幅提振鈸類 Recall。
2. **新歌 Shadow Run 觀測**：
   - 繼續對用戶回饋的真實打鼓音軌進行 shadow run，若發現 A_opt 表現退步，一秒帶上 `--rollback-baseline` 參數退回 A0。

---

## 5. 那些坑不要再踩？ (Anti-Patterns & Lesson Learned)

> [!CAUTION]
> **以下為高壓紅線與實踐血淚史，後續 AI 開發必須無條件迴避：**

1. **嚴禁引入任何音軌名稱硬編碼特判 (No Hardcoded Filename Overrides)**：
   - 絕對禁止在代碼中根據檔名（如 Counting Stars、Rosanna 等）強制指定 BPM、時值或解碼閾值。所有歌曲必須走 adaptive DSP 估計。
2. **嚴禁對驗收測試集進行過擬合特化 (No Blind Overfitting)**：
   - 絕對不要因為 Blind Tests 或 Round 4 的歌曲結果，回過頭來微調或挑選特定的閾值。**Validation 只用來尋優；Test 只用來一筆畫驗收**，否則模型將喪失泛化性。
3. **嚴禁無休止在解碼層調參 (Stop Param-sweeping on Decoder)**：
   - 當稀有鼓件出現召回不足時，**解法在於數據重訓，而非繼續尋找新 threshold**。過高的 threshold 會扼殺真陽性（如大鼓漏檢）。
4. **加載 Checkpoint 時必須帶類別防禦隔離 (Class-count Defense)**：
   - 熱插拔 JSON 等配置時，必須用 `num_classes > 3` 進行隔離，確保三類產品模型（`drum_classifier.pth`）在 `verify_current_solution.py` 中 100% 維持原 MGPC 邏輯，防止代碼重構引發致命 regression。
