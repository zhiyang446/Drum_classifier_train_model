# Dataset 儲存與日後使用說明

最後更新：2026-08-28

## 目前的儲存方式

程式碼與 Python 環境仍在 C 槽；大型資料實體放在外接 D 槽。C 槽保留原本的資料夾名稱，但它們是 NTFS junction，會自動轉到 D 槽。

| C 槽原路徑 | 實際資料位置 | 類型 |
|---|---|---|
| `C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model\e-gmd-v1.0.0` | `D:\DrumDatasets\E-GMD\e-gmd-v1.0.0` | 原始 E-GMD |
| `C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model\STAR_Drums_full` | `D:\DrumDatasets\STAR_Drums_full` | 原始 STAR |
| `C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model\drumsep_d48` | `D:\DrumDatasets\DrumSep\drumsep_d48` | DrumSep 衍生資料 |
| `C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model\drumsep_d52` | `D:\DrumDatasets\DrumSep\drumsep_d52` | DrumSep 衍生資料 |
| `C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model\drumsep_d53` | `D:\DrumDatasets\DrumSep\drumsep_d53` | DrumSep validation 資料 |

目前 C 槽沒有這五組資料的實體副本；D 槽沒有連接時，C 槽 junction 會無法讀取，這是正常現象。

## 以後重新訓練前

1. 接上外接硬碟，確認磁碟代號仍然是 `D:`。
2. 在 PowerShell 進入專案：

```powershell
Set-Location 'C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model'
Get-Volume -DriveLetter D
Get-Item -Force .\e-gmd-v1.0.0, .\STAR_Drums_full, .\drumsep_d48, .\drumsep_d52, .\drumsep_d53
```

3. 確認輸出中的 `LinkType` 是 `Junction`，且 `Target` 指向 `D:\DrumDatasets\...`。
4. 使用專案原本的 Python：

```powershell
.\.venv\Scripts\python.exe verify_current_solution.py
```

5. 驗證通過後，再依專案規格開始訓練。不要把 D 槽資料重新複製回 C 槽，也不要修改 metadata 的舊 C 槽絕對路徑。

## 如果 D 槽代號改變

請先在 Windows「磁碟管理」把外接硬碟重新指定為 `D:`，再執行上面的檢查。不要直接刪除 C 槽 junction，也不要建立指向錯誤磁碟代號的新 link。

## 以後要移植新的大型資料夾

以下流程只適用於已取得人工確認的資料夾；來源資料未完成比對前不可刪除。

```powershell
$root = 'C:\Users\zhiya\Documents\MyProject\Drum_classifier_train_model'
$name = '資料夾名稱'
$source = Join-Path $root $name
$target = "D:\DrumDatasets\$name"
$staging = "D:\DrumDatasets\$name.__staging_YYYYMMDD"

# 1. 先複製到 staging，不刪除 C 槽來源
robocopy $source $staging /E /COPY:DAT /DCOPY:DAT /R:0 /W:0 /MT:8 /XJ /NP

# 2. 唯讀比對；必須看到 exit 0
robocopy $source $staging /L /E /BYTES /R:0 /W:0 /NJH /NJS /NDL /NFL

# 3. 比對通過後，才把 staging 改為正式目標
Move-Item -LiteralPath $staging -Destination $target

# 4. 保留 C 槽 backup，建立原路徑 junction
$backup = "$source.__backup_YYYYMMDD"
Move-Item -LiteralPath $source -Destination $backup
New-Item -ItemType Junction -Path $source -Target $target

# 5. 驗證 junction、抽樣檔案與專案流程後，才可刪除 backup
Get-Item -Force $source
# 確認所有驗收通過後才執行：
Remove-Item -LiteralPath $backup -Recurse -Force
```

如果複製、比對或驗證失敗，停止並保留 C 槽 backup；不要執行最後的 `Remove-Item`。若需要回復，先移除 junction，再把 backup 改回原本的資料夾名稱。刪除 junction 時只能針對 junction 本身操作，不能對 D 槽目標資料夾執行刪除。

## 本次已完成的驗收

- E-GMD、STAR、D48、D52、D53 均完成來源／D 槽完整 Robocopy 比對。
- 五組 C 槽原路徑均為 junction，metadata 舊路徑仍可讀取。
- D48/D52/D53 檔案數與邏輯容量一致，抽樣 WAV hash 一致。
- `verify_current_solution.py` exit `0`。
- C 槽實體 backup 已在驗證通過後刪除；目前資料唯一實體位置是 D 槽，因此日後應另外維持 D 槽備份或備份硬碟。
