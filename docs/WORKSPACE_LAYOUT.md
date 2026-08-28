# 工作區結構與安全操作

## 整理結論

目前不直接把根目錄的 Python 腳本搬進子目錄。這些腳本不是互相獨立的檔案：它們使用根層 import，也有多個工具依賴 `Path(__file__).resolve().parent` 作為資料根目錄。直接搬移會讓既有命令找不到模組或資料。

本文件提供分類索引；現階段採「保持路徑、整理認知」的非破壞方案，不刪除、不覆蓋、不移動資料。

## 根目錄檔案分類

| 類別 | 主要檔案形式 | 用途 | 是否可直接搬移 |
|---|---|---|---|
| 產品推論 | `transcribe.py`、`drum_plugin.py`、`dsp_utils.py` | 音訊特徵、模型推論與 MIDI 輸出 | 不可；保持根層入口 |
| 模型與訓練 | `model_*.py`、`train*.py` | 模型結構、訓練與候選實驗 | 不可；有根層 import |
| 前處理與建置 | `preprocess*.py`、`build*.py`、`convert*.py` | metadata、資料切分與候選資料建置 | 不可直接搬；依賴根目錄資料 |
| 驗證與稽核 | `run*.py`、`audit*.py`、`evaluate*.py`、`compare*.py`、`verify*.py`、`select*.py` | regression、六類驗證與研究證據 | 不可直接搬；命令與相互 import 已固定 |
| 測試 | `test*.py` | 模型、特徵與推論 self-check | 暫留根層，避免既有命令失效 |
| 治理文件 | `AGENTS.md`、`spec.md`、`todolist.md`、`current_status.md`、`STATE.md`、`LOOP.md` | 規格、任務、狀態與安全規則 | 保持根層 |

## 根目錄資料夾分類

### D 槽 junction：必須保持原路徑

目前已驗證的 junction 如下：

| C 槽邏輯路徑 | 實體目標 |
|---|---|
| `e-gmd-v1.0.0` | `D:\DrumDatasets\E-GMD\e-gmd-v1.0.0` |
| `STAR_Drums_full` | `D:\DrumDatasets\STAR_Drums_full` |
| `drumsep_d48` | `D:\DrumDatasets\DrumSep\drumsep_d48` |
| `drumsep_d52` | `D:\DrumDatasets\DrumSep\drumsep_d52` |
| `drumsep_d53` | `D:\DrumDatasets\DrumSep\drumsep_d53` |

執行需要資料的命令前，先連接 D 槽；不要改名、刪除或把這些 junction 再包進其他資料夾。

### 本機原始資料與衍生資料

`Drumsep`、`800000_Drum_Percussion_MIDI_Archive[6_19_15]`、`Whack Studio Metal Drum Tracks`、`Breakdown MIDI Pack`、`real-song`、`synthetic_midi_archive_d27`、`mixed_d*`、`whack_studio_metal_d*`、`enst_d107` 等屬於資料或研究產物，不由 Git 管理，也不在本次整理中移動。

### 本機環境與輸出

- `.venv`：Python 虛擬環境，重新建立比搬移安全。
- `.codegraph`、`.codex`、`.agents`：工具／代理設定，保持原位。
- `validation_runs`、`scratch`、`__pycache__`：驗證、暫存與快取，禁止以整理名義批次刪除。
- `third_party`、`assets`：第三方依賴與音色資源，保留原位，因為部分工具用固定相對路徑讀取。

## 已完成的第二階段分類

為了讓根目錄只保留專案入口、核心程式與 loop 文件，以下檔案已依用途分類：

### `config/expected/`

- `blind_user_tests_expected.csv`：第一批盲測的使用者期望值。
- `egmd_round4_expected.csv`：E-GMD Round4 對照期望值。
- `round2_expected.csv`：Round2 盲測期望值。
- `round3_expected.csv`：Round3 盲測期望值。

相關 Python 工具的預設參數已同步改為 `config/expected/`，仍可從專案根目錄使用原本命令。

### `config/manifests/` 與 `config/examples/`

- `config/manifests/test_real_audio_end_to_end_manifest.json`：五首真實音訊 end-to-end gate 的固定 manifest。
- `config/examples/real_drum_manifest.example.json`：不含實際音訊的 manifest 範例。

### `docs/`

- `docs/HANDOFF.md`：AI 協作接力與研究狀態交接文件。
- `docs/DATASET_STORAGE_GUIDE.md`：D 槽資料集、junction 與重新建立環境的操作指南。
- `docs/WORKSPACE_LAYOUT.md`：本目錄分類與不可移動範圍說明。

根目錄仍保留 `requirements.txt`、`README.md`、治理文件、核心 Python 工具與啟動入口；這些檔案不是遺漏，而是為了相容既有命令與工具的固定入口。

## 日後真正模組化的條件

## 已完成的啟動檔整理

為了減少根目錄雜項，但不改變 Python 腳本的根目錄依賴，以下三個啟動檔已移至 `tools/launchers/`：

- `tools/launchers/run_d108_enst_candidate.cmd`
- `tools/launchers/run_d37_retry.cmd`
- `tools/launchers/run_d38_full_model.cmd`

這三個檔案會由自身位置回算專案根目錄，再使用原本的 Python、metadata、checkpoint、`validation_runs` 路徑。可在專案根目錄執行，例如：

```powershell
& .\tools\launchers\run_d108_enst_candidate.cmd
```

這只是啟動訓練的入口；本次整理沒有執行它們。其他 Python 腳本仍保留根目錄，直到另立模組化任務並完成 import、路徑與回歸驗證。

若日後要把歷史工具移到 `tools/`，必須另立一個獨立任務，至少完成：

1. 建立套件入口並修正所有根層 import。
2. 把資料根目錄改成明確參數或單一共用設定。
3. 更新 `.cmd`、文件與所有使用方式。
4. 先跑各工具 self-check，再跑 `verify_current_solution.py`。

未完成上述條件前，從專案根目錄執行既有命令是安全的使用方式。
