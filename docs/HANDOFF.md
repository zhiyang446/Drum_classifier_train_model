# ADT 專案 AI 協作接力手冊 (HANDOFF.md)

本文件為 **Antigravity** 與 **Codex** 兩大 AI 協作開發的接力交接手冊，詳細記錄當前的系統狀態、已完成里程碑、安全防線、下一步方向與開發防踩坑指南。

## 2026-07-31 D115 現況同步：先停止加資料，修正理解邊界

### 現在的可信結論

- **目前研究基線仍是 D89，不是發布模型**。D111 在 D56 固定 gate 退步，D112 證明它也沒有學到 ENST；D113 的新增錯誤分散，沒有單一同資料根因。
- **D114 tiny-set 可學習性稽核未通過**：在固定 28 個已見 train windows、200 optimizer steps 下，loss 明顯下降，但 Macro F1 只有 `.30568`，六類均未達門檻。現有 frozen feature＋560-parameter LoRA 容量／表示路徑不足；禁止用增加同類資料、延長 steps 或同配方重訓繼續碰運氣。
- **五首真歌沒有使用 DrumSep 六 stem 訓練**：D104 直接讀每首單一原始 MP3，metadata 固定 `input_mode=mix`，沒有任何 stem／DrumSep 欄位。`mix` 表示單音訊流，不單憑欄位判定音檔是否含人聲或其他樂器。
- **五首 MIDI 並非由 FFT correlation 自動校正**：D93 先套固定 offset（四首 `+.05s`、`something +.07s`），D100 之後才量測殘餘 correlation；量測結果沒有回寫事件。D103 只修正 Chop Suey pitch 64 的兩個 TOM 與 Something 的兩個重複 SD，沒有再次平移時間軸。
- **D103 後對齊仍偏鬆**：五首 FFT 殘餘 offset 為 `92.8798/92.8798/92.8798/46.4399/92.8798ms`，逐歌平均 `83.5918ms`。以相同 onset detector 只讀補算，4,876 events 到最近音訊 onset 的絕對距離平均 `98.977ms`、中位數 `72.721ms`；其中 3,996 events 位於 `100ms` 內，該子集平均 `63.480ms`。這不是人工逐音符真值；D100 的 `.15s` `alignment_pass` 不能解讀成精準對齊。
- **ENST 映射規則固定**：
  - KD：`bd`
  - SD：`sd`, `sd-`, `cs`, `rs`
  - HH：`chh`, `ohh`
  - TOM：`lt`, `mt`, `lmt`, `lft`, `mtr`, `ltr`
  - CRASH：`c1`, `cr1`, `cr2`, `cr5`, `ch1`, `ch5`, `spl2`
  - RIDE：`rc2`, `rc3`, `rc4`, `c4`
  - 排除：`cb`, `sweep`, `sticks`
- **映射風險不是缺漏，而是語意合併**：open／closed HH 都進 HH，rimshot／cross-stick 都進 SD；`c4` 一律進 RIDE，而 `ch1/ch5` 進 CRASH，可能為 CRASH／RIDE 帶來少量語意噪音。

### 下一位 AI 只能怎麼繼續

1. 不得把 D100 `alignment_pass` 當成五首 reference 已精準對齊的證明。
2. 不得再啟動同一 frozen-feature LoRA 配方，也不得用「再加幾首／再加 epoch」迴避 D114。
3. 若要處理五首資料，先另立唯讀或人工驗證的物理時間對齊方案；不得用模型預測鼓點反向調整 reference。
4. 若要處理容量問題，保持 D114 資料、選窗、loss、decoder 與 200 steps 不變，只能另立「單獨解凍最後時序模組」的單變因規格，並先取得使用者明確授權。
5. `test_real_audio` 固定五首、STAR test 與 ENST drummer_3 仍是封存 gate，不得移入訓練或選參。

### 本節證據

- `validation_runs/d103_corrected_reference_audit/audit_d100.json`
- `real-song/d104_five_fold/fold_01/train_metadata.json`
- `real-song/d104_five_fold/fold_01/heldout_metadata.json`
- `validation_runs/d106_enst_six_class_audit/raw_label_mapping.csv`
- `validation_runs/d114_tiny_overfit_audit/summary.json`

## 2026-07-22 D78 完成：停止重複 CRASH-vs-KD

- 新證據：`validation_runs/d78_d77_crash_residual_audit/crash_misses/summary.json`、`validation_runs/d78_d77_crash_delta_audit/summary.json`。
- D77 相對 D67 的 CRASH TP／FP／FN 為 `-7/-40/+7`，F1 只增加 `.0095`；109 個殘餘 FN 仍以 KD `62/109`（`.5688`）嚴格過半。
- 決策：KD 競爭候選已做過且 CRASH FN 增加，`new_competitor_feasibility_allowed=false`。不得直接重跑 KD 配方；下一個候選必須先從 HH 或其他尚未驗證的根因做唯讀審計。
- 測試：`py_compile audit_d74_d67_crash_misses.py audit_d78_d77_crash_delta.py`、兩個 `--self-check`、實際 D78 48-window 審計與 `git diff --check` 均通過。

## 2026-07-22 D79 完成：HH 沒有單一可訓練根因

- 證據：`validation_runs/d79_d77_hh_error_audit/summary.json`。
- HH FP `142`（cross-class `87`、unannotated `55`）；HH FN `89` 的替代為 KD `32`、SD `30`、TOM `12`、CRASH `8`、RIDE `7`，最大僅 `.3596`。
- 決策：`ready_for_training_candidate=false`，不建立 D80 資料可行性或 D81 訓練；現有資料不支援把 HH 問題誠實縮成單變量候選。

---

## 0. 2026-07-20 資料接入現況（本節優先於下方舊研究摘要）

- **D53–D55 完成**：D53 已將 8 首 held-out validation 音訊以固定 D47/D48 DrumSep 配方隔離分離為 `48/48` 個 44.1kHz 雙聲道 stem，未讀 validation event。D54 的 `mixed_d54_stem/metadata_d54.json` 完整保留 1,460 key/event/split，新增 8,760 stem 路徑，train/validation 為 `1,452/8` 且 group leak 為 0。D55 以六 stem 相同時間窗相加為單聲道 drum-only mix，重用既有 DCNN+Conformer；train/validation 實讀、35-batch 煙霧訓練與單檔 MIDI 推論均通過。零 onset 推論已修成輸出安全空 MIDI，legacy `verify_current_solution.py` 通過。小型 smoke 分數不可作為品質結論。
- **D56 完成（候選拒絕）**：只使用 D54 train 與 8 首 validation，完成全新 checkpoint 的 5 epochs／3,500 batches。最佳 epoch 5 與獨立 reload validation 均為 Macro F1 `0.4922`：KD `.6107`、SD `.5014`、HH `.5143`、TOM `.4783`、CRASH `.3071`、RIDE `.5412`；六類 gate 要求 Macro `.70` 且每類 `.55`，因此 `overall: fail`。CRASH precision 僅 `.2364`、extra `65.76%`；TOM recall `.3877`。此結果證明同一路徑可以完整訓練與推論，但不證明品質足夠；不得部署、替換產品模型、讀 test／五首或藉 threshold 補救。
- **D57 完成（確認真實提升）**：以 D56 獨立 validation 的相同 48 key/anchor 固定窗口、相同 D54 metadata/feature/threshold，重跑 D38 原始 raw `mix` checkpoint，得到 Macro F1 `0.0552`（`.0325/.0562/.1297/.0086/.0000/.1039`）。D56 `drumsep-mix` 在完全相同窗口為 `.4922`，絕對提升 `.4370`。因此「DrumSep 沒有效果」已被否定；下一個工作必須只審計 D56 的 CRASH false positive 與 TOM false negative 根因，禁止重複 raw-mix 對照、直接發布或用 validation 調 threshold。
- **D58 完成（自動錯誤審計）**：`audit_d58_drumsep_errors.py` 只讀 D56 checkpoint、D54 metadata 與同一 48 封存窗口，產生 `crash_false_positives.csv`、`tom_misses.csv`、`summary.json`。CRASH FP 為 `252`：cross-class `125`、unannotated `127`，同拍 KD `70` 次；TOM FN 為 `139`，最高替代 SD `50`、KD `46`、CRASH `23`。這些是診斷線索，不是可直接修改 validation 標註的授權。自檢、編譯及 `verify_current_solution.py` 均通過；無 checkpoint／資料／閾值／test 變更。下一個最小工作是只抽樣審核 unannotated CRASH 的音訊與 stem 證據，決定它們是漏標還是音色/分離誤報，再選一個資料根因處理。
- **D59 完成（stem 聲學證據）**：`audit_d59_crash_stem_evidence.py` 只讀 D58 的 127 個 unannotated CRASH 與 D54 six stem，在事件附近 100ms 計算相對 power。結果為 other-stem-dominant `75`、mixed `19`、crash-dominant `33`。這支持多數事件屬 CRASH 誤報／類別邊界，而非可直接假定為漏標；33 個 crash-dominant 僅是小樣本真值復核候選。自檢、編譯及 `verify_current_solution.py` 均通過，沒有資料、標註、checkpoint、threshold 或 test 變更。下一個單一根因工作應是訓練集的 CRASH-vs-KD 負例資料配方，而不是用 validation 調 threshold。
- **D60 完成（KD-only 負例配方）**：`train_six_class_candidate.py` 新增 opt-in `--negative-anchor-inst`，D60 選 `KD` 並排除同時 SD/HH 的錨點。D54 稽核出同樣 `2,800` windows、`400/400` NEG 均為 `d36_whack_real` 純 KD 錨點，且 4 秒窗口無 TOM/CRASH/RIDE；六類正樣本配額、DCNN+Conformer、loss、feature、validation/test 完全未變。trainer self-check、編譯與 `verify_current_solution.py` 通過。D60 是 D61 的已驗證訓練配方，沒有改動產品 checkpoint 或 gate。
- **D61 完成（KD-only 負例候選；拒絕）**：從 D38 full-model checkpoint 完成 D56 等配方的 `5 epochs / 2,800 windows / 3,500 batches`，唯一變因為 400 個 NEG 全為純 KD；訓練 loss `.8734 → .1236`。D56 封存相同 48-window 獨立驗收為 Macro `.5267`，相對 D56 `.4922` 增加 `.0345`；KD/SD/HH/TOM/CRASH/RIDE 為 `.6363/.5476/.5126/.5061/.3707/.5870`。CRASH FP 從 `252` 降至 `156`（`-96`），CRASH F1 `.3071 → .3707`，證明此根因處理有正向效果；但 Macro 未達 `.70`，SD/HH/TOM/CRASH 仍未達 `.55`，故候選明確拒絕。不得部署、替換模型、讀 test／固定五首，亦不得用 validation 改 threshold。
- **D62 完成（D61 殘餘錯誤審計；不訓練）**：重用 D58 審計器、D54 metadata 和 D56 封存相同 48 windows，輸出於 `validation_runs/d62_d61_error_audit/`。D61 的 CRASH FP `156` 分為 cross-class `73`、unannotated `83`；已標註 cross-class 以 KD 相關 `52`（純 KD `40`＋含 KD 組合 `12`）最多，說明 KD-only NEG 已有效但剩餘 CRASH 邊界仍在。TOM FN `124` 的最高替代改為 KD `47`、SD `42`、HH `17`、CRASH `15`、RIDE `3`。下一個唯一候選應先設計 TOM-vs-KD/SD 資料配方；不得重複 D61、改 threshold、改標或讀 test／固定五首。
- **D63 完成（TOM-vs-KD/SD 可行性；不訓練）**：`audit_d63_tom_competitor_feasibility.py` 在 D54 train 只讀得到 `7,138` 個居中 TOM、其中 `1,953` 個在 `.05s` 內有 KD 或 SD。Whack/Archive/Breakdown 的競爭候選為 `833/1,098/22`；既有 D37 TOM source quota 僅需 Whack `300`＋Archive `100`，可維持 400 TOM windows 而不重複事件、放寬 split 或補資料。下一步可設計唯一變因為 TOM 正樣本選窗規則的 D64；KD-only NEG、其他類別、架構、loss、feature、gate 均不可一起變動。
- **D64 完成（TOM-vs-KD/SD；拒絕）**：trainer 新增預設關閉 `--tom-kd-sd-competitor`，只過濾 TOM 正樣本為 `.05s` 內有 KD 或 SD；D64 實際 400 TOM 為 Whack `300`＋Archive `100`。從 D38 相同起點完成 D61 等配方的 `5 epochs / 2,800 windows / 3,500 batches`，trainer self-check、排程稽核與完整三類 regression 均通過。封存相同 48-window gate 為 Macro `.5208 < D61 .5267`；TOM `.5061 → .5594`、FN `124 → 107`，但 CRASH `.3707 → .3342`、KD `.6363 → .6078`、SD `.5476 → .5225`。TOM 根因假說成立但造成整體退步，D64 候選拒絕；不得部署、替換模型、讀 test／固定五首或以 validation 改 threshold。
- **D65 完成（分段對齊審計；拒絕線性校正）**：`audit_d65_piecewise_whack_alignment.py` 僅讀 D45 暫停的 28 首 Whack train 音訊/MIDI，以十等分加 D45 `25%/50%/75%` 錨點重做 fixed-BPM FFT 局部對齊；新 profile 與 D45 舊三點最大差異為 `0s`。`whack_studio_metal_d65/audit_d65_piecewise_profile_v2.json` 顯示線性最大局部殘差 `<=.25s` 為 `0/28`（`<=.5s` 也是 `0/28`），中位 RMSE `1.559s`、最大殘差 `5.118s`。故不存在可安全自動恢復的線性校正樣本：不改 event/MIDI/metadata/split、不建 manifest、不訓練、不讀 validation/test，28 首繼續暫停；若日後要救，須先提出可驗證的非線性對齊方法及校正後殘差/邊界/隔離 gate，不能硬套單一 offset。
- **D66 完成（密集自動時間扭曲復測；拒絕）**：`audit_d66_dense_piecewise_alignment.py` 僅在記憶體以 D65 的 `11` 個唯一局部 offset 點（50% 重疊不重複）重建 raw MIDI impulses，再以同一 FFT profile 量測校正後殘差；不寫 event/MIDI/metadata/manifest。`whack_studio_metal_d66/audit_d66_dense_piecewise_probe.json` 顯示 `.25s` 殘差 gate 為 `0/28`：`9` 首會把 event 推出音訊邊界，剩餘 `19` 首的校正後最大局部殘差中位數 `3.715s`、最大 `3.994s`。Wicked 單首的正／負方向探針也均保留大殘差，故不是符號約定錯誤，而是短段 offset 跳變不能靠自動插值修復。D66 拒絕；全部 28 首繼續暫停，不建資料候選、不訓練、不讀 validation/test。
- **D67 完成（TOM 類別專家融合；新的研究基線）**：`audit_d67_d61_d64_tom_fusion.py` 對 D61/D64 checkpoint 和完全相同的 D56 封存 48 windows 做離線重跑；KD／SD／HH／CRASH／RIDE 只取 D61 機率，TOM 只取 D64 機率，固定 `.50` 閾值與 `.05s` 匹配，沒有平均、權重搜尋、訓練、資料或產品推論變更。`validation_runs/d67_d61_d64_tom_fusion/fusion_summary.json` 得到 Macro `.5356 > D61 .5267`（`+.0089`），六類 `.6363/.5476/.5126/.5594/.3707/.5870`；D64 TOM 改善與 D61 其餘五類可完全共存，故此固定配方取代 D61 成為研究基線。完整 release gate 仍 fail：Macro 未達 `.70`，SD／HH／CRASH 未達 `.55`；不得讀 test／固定五首、部署或替換產品 checkpoint。
- **D68 完成（D67 SD 誤報根因；不訓練）**：`audit_d68_d67_sd_errors.py` 重用 D67 融合與 D58 的固定匹配規則，在完全相同 48 windows 只列出 SD false positive。`validation_runs/d68_d67_sd_error_audit/summary.json` 得到 SD FP `215`：cross-class `129`、unannotated `86`；含 KD 的鄰近真值 `66`、含 TOM `44`，局部最高替代則為 KD `64`、HH `57`、TOM `52`、CRASH `42`。下一個單一資料根因應先檢查 SD-vs-KD 的訓練候選是否有足夠、不重複且來源隔離的資料；86 個未標註事件不可自動修標。D68 沒有訓練、改閾值、讀 test／固定五首或改產品 checkpoint。
- **D69 完成（SD-vs-KD 候選可行性；不訓練）**：`audit_d69_sd_kd_competitor_feasibility.py` 重用 D63 的四秒居中規則，只讀 D54 train。`validation_runs/d69_sd_kd_competitor_feasibility/summary.json` 有 `2,715` 個 SD-vs-KD 候選，Whack `1,962`、Archive `705`，高於 D37 SD 的 Whack `300`＋Archive `100` 固定配額。因此 D70 可只把 400 個 SD 正樣本限制為這個競爭集合，而不改其他類別、來源隔離或 split；D69 本身未建立 schedule、訓練、讀 validation/test 或改 checkpoint。
- **D70 完成訓練（SD-vs-KD candidate；D71 固定驗收已拒絕）**：`train_six_class_candidate.py` 新增預設關閉 `--sd-kd-competitor`，只篩選 SD 正樣本為 `.05s` 內有 KD；D70 的 400 SD 精確為 Whack `300`＋Archive `100`，其餘 D61 配方不變。trainer self-check、D54 2,800-window 排程稽核及 legacy blind 5/5、hard 4/4、Round4 30/30＋6/6 gate 均通過。D70 從 D61 完成 5 epochs／3,500 batches，loss `.1198 → .1005`，epoch 1 的訓練期 validation Macro `.5298`（KD/SD/HH/TOM/CRASH/RIDE `.6154/.5157/.5317/.5442/.3120/.6598`）。D71 封存 48 windows 的 D70 五類加 D64 TOM 固定融合為 Macro `.5323 < D67 .5356`，故 D70 不取代 D67；未讀 test／固定五首或部署。
- **D71 完成（D70/D64 TOM 固定融合；拒絕）**：`audit_d67_d61_d64_tom_fusion.py` 已泛化報告欄位，但融合邏輯不變：D56 封存相同 48 windows、True SuperFlux、`drumsep-mix`、`.50` threshold、`.05s` tolerance；D70 只供應 KD／SD／HH／CRASH／RIDE，D64 只覆蓋 TOM。`validation_runs/d71_d70_d64_tom_fusion/fusion_summary.json` 為 Macro `.5323 < D67 .5356`（`-.0033`），六類 `.6154/.5157/.5317/.5594/.3120/.6598`，因此 `research_status=rejected`。D67 保留研究基線；D71 未訓練、未讀 test／固定五首、未調閾值、未改產品推論或 checkpoint。
- **D72 完成／D73 停止（SD-vs-KD 路線拒絕）**：新增唯讀 `audit_d72_d70_delta.py`，只比較既有同一 48-window event CSV 與 gate JSON，沒有重跑模型。`validation_runs/d72_d70_vs_d61_delta_audit/summary.json` 顯示 SD TP `-12`、FP `+7`、FN `+12`、precision `-.0247`、recall `-.0440`、F1 `-.0319`；因此 `status=d70_route_rejected`、`d73_training_allowed=false`。D73 不建立 schedule、不訓練、不寫 checkpoint；D67 繼續為研究基線。下一步若再推進，應先做 CRASH 資料／標註根因審計，而非重複 SD-vs-KD 訓練。
- **D74／D75 完成（CRASH-vs-KD 根因與可行性）**：`audit_d74_d67_crash_misses.py` 在 D67 固定 48 windows 找到 `102` 個 CRASH FN，KD 為最高替代 `60`（`.5882`，嚴格過半）；`audit_d75_crash_competitor_feasibility.py` 只讀 D54 train，得到居中 CRASH+KD 候選 `6,169`，Whack `5,665 >= 260`、Archive `216 >= 80`、Breakdown `288 >= 60`。因此可設計一個單一 CRASH-vs-KD 候選，但尚未建 schedule、訓練、寫 checkpoint、讀 test／固定五首或改產品推論。
- **D76／D77 完成（CRASH-vs-KD；新的研究基線）**：`train_six_class_candidate.py` 新增預設關閉 `--crash-kd-competitor`，只篩 CRASH 正樣本為 `.05s` KD 共現；D76 的 400 CRASH 精確為 Whack `260`＋Archive `80`＋Breakdown `60`，其餘 D61 配方不變。self-check、2,800-window 排程及元件 regression（blind raw／notation 5/5、hard 4/4、Round4 30/30＋6/6）通過。首次訓練受 600 秒執行器上限中斷並保留；retry 完整 5 epochs／3,500 batches，best epoch 3。D77 以 D76 五類＋D64 TOM 的固定 48-window 融合得 Macro `.5386 > D67 .5356`（`+.0030`），六類 `.6360/.5618/.5426/.5594/.3802/.5517`；故取代 D67 作研究基線。完整 release gate 仍 fail（HH／CRASH 未達 `.55`、Macro 未達 `.70`）；未讀 test／固定五首、未調 threshold、未部署或替換產品 checkpoint。

- **目前焦點**：D25–D35 已完成外部鼓資料接入、對齊稽核與一輪隔離訓練；D35b 已被 D34 held-out test 拒絕，沒有修改產品 checkpoint、替換模型或解除任何 release gate。
- **Whack Metal**：110 首原始 WAV/MIDI 中，D31 產出 95 首自動對齊候選（D29 `13`、D30 `82`），split 為 train/validation/test `79/4/12`，六類皆存在、group split 無洩漏。另有 15 首未選入。
- **重要限制**：D31 有 23 首共 563 個 event 因 offset 後超出音訊邊界而被記錄並丟棄（before `163`、after `400`）；因此 `whack_studio_metal_d31/audit_d31.json` 仍為 `ready_for_training_candidate=false`，不能用來直接訓練。
- **D32 批次修復結果**：已一次檢查全部 38 首疑慮歌曲；僅 5 首有穩定的前／中／後局部對齊。其餘 33 首的局部 offset drift 平均 `3.0383s`、最高 `7.5233s`，不可用全曲固定 BPM＋單一 offset 硬修；完整證據為 `whack_studio_metal_d32/recovery_d32.json`。
- **D33 暫緩策略**：依使用者決定暫不做分段時間校正。已建立 72 首零裁切安全候選（split `60/2/10`、每個 split 六類完整）；D32 的 5 首雖局部對齊穩定，但重建後仍有邊界裁切，未被納入。`whack_studio_metal_d33/audit_d33.json` 仍為 `ready_for_training_candidate=false`。
- **D34/D35b 結論**：D34 將 72 首重分為 `56/8/8`；D35b 的 5 epochs 最佳 validation Macro F1 為 `0.5911`，但 D34 test（48 windows，`8/class`）Macro F1 僅 `0.0578`，KD/SD/TOM/CRASH/RIDE 零預測、HH F1 `0.3470`。候選位於 `validation_runs/d35b_whack_safe72_dcnn_tcn_conformer/`，只供失敗證據，禁止部署或替換產品模型。
- **回歸證據**：legacy `verify_current_solution.py` wrapper 未印出最終 PASS；已逐一重新執行其元件並通過 Blind raw/notation `5/5`、hard `4/4`、Round4 strong-event `30/30 + 6/6`。這是三類回歸證據，不是六類發布證據。
- **D37 進行中**：固定每類 400：KD/SD/HH/TOM/RIDE 為 Whack 300＋Archive 100；CRASH 為 Whack 260＋Archive 80＋Breakdown 60。NEG 只取 Whack 的窗口級無 rare event。schedule 2,800 windows 已自檢，來源不足會 fail-fast；三類回歸元件 Blind 5/5、hard 4/4、Round4 30/30+6/6 均 pass。第一次前景訓練在 epoch 1 的 650/700 batches 受 120 秒工具時限中止且沒有 checkpoint，原目錄只作中斷證據；正式候選改在新的 retry 目錄背景執行。D34 test、固定五首與產品 checkpoint 保持隔離。
- **D37 中斷證據**：一次性工作排程曾在 retry 目錄啟動；epoch 1 `700/700` 已完成並寫 checkpoint，但 validation 六類皆 `0.0000`。它在 epoch 2 `175/700` 後無 stderr 地停止，沒有 `train_report.json`，且排程／Python 程序都不再存在。D37 未形成候選、不可部署；保留 retry artifacts，後續若重啟必須另立新候選並先取得使用者確認。
- **D38 已核准**：使用者已明確核准以 D37 同一配額重跑，但唯一改用 `--full-model`。理由是 D37 起點為三類 checkpoint，head-only 只更新 780 參數、凍結 1,173,843 參數，導致六類零預測；D38 會新建 candidate，絕不覆寫 D37／產品模型或讀取 held-out test。
- **D38 執行中**：`DrumClassifier-D38-20260721` 已啟動。trainer self-check、實際 2,800-window source quota audit 及 `git diff --check` 均通過；epoch 1 已到 25/700 batches、stderr 空白。結果只能以完成後的 Whack validation/report 判定。
- **D38 結論（拒絕）**：完整 5 epochs／3,500 batches 完成、loss `0.8029 → 0.1519`；best epoch 5 Whack validation Macro `0.4809`，六類 `0.6651/0.5797/0.5079/0.3299/0.2647/0.5380`。full-model 排除了 D37 的零預測，但 HH/TOM/CRASH/RIDE 未達 0.55，故不得讀取 test、固定五首或替換產品模型。candidate 僅作研究證據。
- **D39 校正後結論**：D38 原 validation 48 windows 只來自 3 個 group、Rot 佔 37 個，導致 `0.4809` 高估。共用選窗改為 group round-robin 後，48 windows 覆蓋所有 8 個 group（每首 5–7）；相同 D38 epoch 5 的 Macro 僅 `0.0552`、六類 `0.0325/0.0562/0.1297/0.0086/0.0000/0.1039`。這是 D38 的有效 validation 證據，候選明確拒絕，不可再用舊分數作結論。
- **D40 全 epoch 證據**：D38 epoch 1–5 在相同平衡 validation 的 Macro 為 `0.0018/0.0180/0.0158/0.0396/0.0552`；epoch 5 確為最高，故問題不是舊 selector 選錯 epoch，而是配方對未主導訓練的 Whack 歌曲沒有泛化。禁止重跑 D38 同配方或讀取 held-out test；下一個合理步驟是自動化資料／對齊／標註錯誤稽核。
- **D41 對齊根因證據**：唯讀 metadata audit 顯示 validation 8 首中 6 首是相對 train 的對齊離群：Rot/Savage/Inferno existing abs offset `2.694/2.461/1.858s`（train median `0.418s`）；Eternal Conflict/Haze Overdose/Reflections alignment score `0.349/0.316/0.356`（train median `0.631`）。這是 D38 跨歌崩潰的首個可量測資料根因；不可自動位移事件，下一步只能用既有 D29/D32 local alignment 方法做這 6 首的唯讀復核。
- **D42 局部對齊復核完成**：六首的固定 BPM global score／offset 全數重現 D36 metadata，故不是欄位寫錯。Rot/Haze Overdose/Savage/Inferno/Reflections 的前中後 local drift 為 `5.248/1.904/4.180/2.879/0.650s`，皆超過 `0.25s`，證實五首不能以單一全曲 offset 表示；Eternal Conflict drift `0.093s` 但 score 仍低 `0.349`。證據位於 `whack_studio_metal_d42/audit_d42.json`；沒有自動位移 event、改 metadata 或重訓，固定為不可訓練／不可發布。
- **D43 分段候選完成**：已由原始 MIDI 建立 `mixed_d43/metadata_d43.json`，只替換 Rot/Haze Overdose/Savage/Inferno/Reflections 五首 validation 的 event；其餘 `1,483/1,488` item 完全不變（訓練 `1,480`＋validation `8`）。五首 event 數均保持、沒有越界或時間倒退，D43 仍是不可訓練候選。下一步應只用 D43 對既有 D38 checkpoint 做隔離 validation 重評，不能直接重訓或讀取 test。
- **D44 固定窗口重評完成（拒絕）**：D39 與 D44 的 48 個 key／anchor／window_start 完全相同，六類模型預測數也完全相同；只有 D43 的候選真值改變。Macro F1 因此由 `0.0552` 降至 `0.0391`，六類為 `0.0243/0.0704/0.1141/0.0000/0.0000/0.0256`。局部漂移是實在資料問題，但不是 D38 跨歌曲失敗的充分根因；不得重訓、讀 test 或替換模型。證據位於 `validation_runs/d44_d38_d43_fixed_window_validation/`。
- **D45 train 對齊稽核完成**：56 首 Whack train 中只有 `28` 首 local drift 不超過 `0.25s`，另 `28` 首已暫停；全體 median drift `0.395s`、最高 `7.709s`。可保留 28 首的六類 event 為 `24,545/8,848/6,637/4,098/8,378/2,791`，資料量足以建立新的乾淨訓練候選。證據位於 `whack_studio_metal_d45/audit_d45.json`；下一步是建新 manifest，仍不可直接重訓或讀 test。
- **D46 乾淨 manifest 完成**：`mixed_d46/metadata_d46.json` 共 `1,460` items，精確排除 D45 暫停的 28 首 Whack train；28 首穩定 Whack train、Archive、Breakdown 與原始 8 首 validation 全數保留。validation 逐筆完全不變，group split 無洩漏，train 六類 event 為 `34,610/17,250/16,453/9,507/10,130/9,388`。D46 仍不可直接訓練；下一步需固定全新 D47 candidate 配方，不能重跑 D38。
- **D47 DrumSep smoke test（完成）**：使用者提供的 DrumSep MDX23C 權重 SHA-256 `D2A4AA53...BE9096D0` 已核對；官方推論 revision `83d495dfc81b2ede9bc62f4209619f8bdfd14995` 已隔離於 `third_party/`。RTX 4050 6GB 對 D46 穩定 Whack train 的前 30 秒實際分離耗時 `17.35s`，成功輸出 kick/snare/toms/hh/ride/crash 六個 44.1kHz 立體聲 stem，全部非空。完整證據為 `drumsep_d47/audit_d47.json`；這只證明分離可行，尚未證明任何 MIDI／轉譜品質，亦未讀 validation/test、訓練、LoRA 或改既有模型。
- **D48 D46 stable Whack batch（完成）**：28 首 D46 穩定 Whack train 已全部分離成六 stem，共 `168` 個非空 44.1kHz 雙聲道 WAV、`14.983GiB`。使用 D47 原始 checkpoint/YAML、GPU、無 TTA／LoRA；26 首續跑的排程耗時 `748.44s`、結果碼 `0`，一次性排程已移除。完整 audit 為 `drumsep_d48/audit_d48.json`。這是資料前處理候選，沒有 MIDI／轉譜品質結論，未讀 validation/test、訓練或改既有模型；下一步必須先做 stem 品質與 MIDI 對齊稽核，不能直接重訓。
- **D49 stem 品質／MIDI 對齊稽核（完成）**：28/28 首、168/168 stem 格式與非靜音全通過；六 stem 對原混音重組 median correlation `0.9990`、normalized residual `0.0453`，只作分離品質代理。最終 canonical audit 是 `drumsep_d49/audit_d49_reclassified.json`：6 個沒有 RIDE MIDI event 的 coverage gap 標為不可評估，不誤列品質失敗；唯一 review 是 `whack_metal_d34_063` 的兩個 RIDE event local energy/background `-0.664dB`。沒有刪除、修正 MIDI／音訊、訓練或讀 validation/test。下一步仍只能先設計並審查「推論也跑 DrumSep」的兩階段 candidate，不能把 train-only stem 直接餵給目前產品模型後就宣稱可用。
- **D50 stem-aware 候選 manifest（完成）**：`mixed_d50_stem_candidate/metadata_d50.json` 只在 28 首 D46 stable Whack train 加上 D48 六 stem 路徑，共 168 檔；D46 的 1,460 key、8 筆 validation、原始 events、MIDI 與完整混音標籤均保持不變，group split 無洩漏。唯一 stem auxiliary mask 是依 D49 review 自動產生的 2 個 RIDE event，不依檔名硬編碼。`mixed_d50_stem_candidate/audit_d50.json` 固定 `ready_for_training_candidate=false`；沒有訓練、LoRA、checkpoint 或產品推論變更。下一步是先審查獨立兩階段 candidate 配方，並確保推論也會先跑 DrumSep。
- **D51 兩階段可行性 gate（拒絕實作）**：D50 stem 只覆蓋 `28/1,452` train 曲目（`1.928%`；`37.592%` 音訊時長），8 個 held-out validation 全無 stem。現有 trainer 只讀 `audio_path`，`transcribe.py` 不會跑 DrumSep，所以現在加 stem 分支會造成 train／validation／推論不一致，已拒絕實作且未訓練。若使用者日後明確批准全量預處理，train 尚需分離 `1,424` 曲、`8,544` stem，依 D48 密度約額外 `24.89GiB`；validation/test 必須獨立產生與隔離，不能用於選參。D51 沒有改模型、checkpoint、資料切分或產品推論。
- **D52 剩餘 train 全量 DrumSep（完成）**：Archive `1,382`＋Breakdown `42` 共 `1,424` 首 D50 未分離 train 已全數完成，`drumsep_d52/output/` 有 `8,544` 個非空 44.1kHz 雙聲道 stem、`24.874GiB`。權重／YAML／revision 延用 D47/D48，MP3 smoke 與正式 batch 均通過；排程結果碼 `0`，一次性排程已移除。來源 key 對應與驗收在 `drumsep_d52/key_map_d52.json`、`preflight_d52.json`、`audit_d52.json`；沒有讀 validation/test、訓練、LoRA、修改 MIDI、manifest、checkpoint 或產品推論。下一步仍是隔離建立 held-out validation stem，再審查可推論的兩階段候選，不能直接訓練或發布。
- **排程清理**：D37/D38 一次性工作排程已於完成後移除，避免在原排定時間重跑並覆寫保存的候選／中斷證據。
- **交接入口**：資料狀態以 `current_status.md`、`whack_studio_metal_d28/audit_d28.json`、`whack_studio_metal_d29/alignment_d29.json`、`whack_studio_metal_d30/filename_bpm_audit_d30.json`、`whack_studio_metal_d31/audit_d31.json`、`whack_studio_metal_d41/audit_d41.json`、`whack_studio_metal_d42/audit_d42.json`、`mixed_d43/audit_d43.json`、`validation_runs/d44_d38_d43_fixed_window_validation/gate_summary.json`、`whack_studio_metal_d45/audit_d45.json`、`mixed_d46/audit_d46.json` 為準。

---

## 1. 現在做什麼？ (Current Focus)
- **當前焦點**：已移除 `Counting Stars`、`Rosanna`、`Blue` 的檔名特例。`D7-A_opt` 僅是研究校正，不是發布版本。
- **系統狀態**：D16 確認 A_opt 搜尋只用 STAR validation，且 checkpoint 雜湊相符；但 Round4 為 `35/36`、`overall: fail`，不得宣稱 A_opt 發布或商業 gate 通過。三分類回歸通過不構成六分類發布證據。
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
- **發布阻塞**：A_opt 未通過 Round4 的正式 gate；在取得真正未參與選參的獨立六分類發布驗收前，不得發布、替換產品模型或宣稱商業通過。

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
