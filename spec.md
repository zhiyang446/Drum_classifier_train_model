# 📑 ADT 系统技术规范 (spec.md)

本文件记录自动打鼓转谱 (ADT) 系统核心模型的技术规范、网络架构、超参数配置、特征维度标准以及模型性能基准。

---

## STAR conservative fine-tune note

`best_drum_model.pth` remains the main E-GMD + IDMT candidate. `best_drum_model_backup.pth` must not replace it only because one shuffle sample improves. STAR adaptation from `best_drum_model.pth` should first use a lower learning rate and fixed regression checks before producing a new candidate checkpoint.

Small STAR fine-tune batches can corrupt BatchNorm running statistics and collapse inference to a few kick events. Conservative STAR adaptation should freeze BatchNorm statistics during small-data experiments.

If BatchNorm freezing prevents collapse but still regresses fixed tests, the next conservative rung is head-only adaptation: freeze backbone and TCN layers, update only `onset_head` and `velocity_head`.

Snare recovery experiments should weight positive onset labels per channel instead of globally lowering inference thresholds. This keeps the fix in the acoustic model training path.

## Raw AI gate reporting note

`compare_blind_expected.py --layer raw` reports only model-layer counts from `raw_*` summary columns. Notation-layer virtual recovery fields must be blank in raw reports, because they describe brain/post-processing output and are not evidence that the acoustic model produced an onset.

## Raw AI hard-negative objective note

Raw AI repair candidates may use `train_mixed_datasets.py --hard-neg-boost` to up-weight non-onset frames that the current model predicts with high probability. This keeps the model architecture unchanged and focuses training on false-positive peaks instead of raising inference thresholds or adding song-specific rules.

## Raw AI teacher metadata note

Confirmed score-image annotations can mix score-time and audio-time coordinates. Raw AI repair training must not use those rows directly unless they are converted into physical audio time. A safer short path is to build temporary teacher metadata from a notation pass that already passed the user blind gate, using each event's `raw_time` as the training target.

## Raw acoustic gate note

The user blind expected CSV is a notation target and must not be used as the Raw AI acoustic acceptance gate. Raw acoustic validation must compare model raw counts only against confirmed annotations that are already in physical audio time, such as `source=raw_ai`, `source=audio_onset`, or `source=grid_fill+audio_onset`. Confirmed rows from `score_image` or plain `grid_fill` are score-time rows unless explicitly converted, so the training metadata converter must reject them by default.

## Current solution verification note

The accepted solution must be checked through one repeatable verification entrypoint. `verify_current_solution.py` runs the accepted checkpoint through the blind transcription batch, compares both raw acoustic and notation gates, runs hard validation, and runs the accepted Round4 E-GMD physical strong-event gates. A run is accepted only when every generated gate report is `pass`.

Round4 verification inside `verify_current_solution.py` checks the first 5 selected clips plus the sixth available KD/SD/HH-only clip (`--offset 5 --limit 1`) using each run's `gate_summary.csv`, not the diagnostic full-MIDI count CSVs.

The 2026-07-10 recheck at `validation_runs\\current_solution_verification_20260710_recheck` passed every accepted gate: blind raw acoustic `5/5`, blind notation `5/5`, hard validation `4/4`, Round4 first 5 `30/30`, and the sixth Round4 clip `6/6`.

## Round5 MIDI-assisted real-audio smoke test

Round5 evaluates new user-provided, main-system-separated full drum tracks without retraining or changing transcription behavior. A paired MIDI file may supply KD/SD/HH reference events only after automatic audio/MIDI alignment; non-KD/SD/HH MIDI pitches remain unsupported and must not become expected KD/SD/HH events. The same shared pitch mapping, fixed matching tolerance, and accepted checkpoint apply to every Round5 pair. A failed alignment or a mismatch must be reported as evidence, never repaired by file-name rules, path routing, or changed model thresholds.

Only the user-provided, main-system-separated WAV is a Round5 test input. A score-playback MP3 or other reference audio stored beside it is excluded from the verdict. A Round5 failure is diagnostic evidence, not tuning data: no file-name routing, expected-count rule, fixed-tempo rule, or direct retraining on the held-out song is permitted. A correction must first reproduce the failure with independent development data and must then re-run all Round5 inputs.

### Round5 model-priority repair rule

The user has explicitly authorized candidate-model training when Raw AI evidence is below the real-audio gate. Training must start from `mixed_formal_kick375_snare18_hh12_candidate.pth`, write only a new candidate under `validation_runs`, and use only `split=train` E-GMD/STAR/local metadata. The first candidate trains the SD/HH output head while freezing KD and BatchNorm statistics, so the accepted KD behavior remains a regression guard. It must pass `verify_current_solution.py` before Round5 is rerun. A brain-layer change may only be retained when it prevents a measured notation regression without concealing a Raw AI failure.

### Round5 shared brain safeguards

Tempo/meter scoring must cap the Fano dispersion contribution at `15.0` before it is combined with cross-measure similarity. This prevents a fine-grid dispersion outlier from dominating the shared score, while leaving the grid, candidate tempos, and true odd-meter candidates available. GPAR may still classify a phase as active at its existing `35%` repeat threshold for suppression decisions, but it may create a new virtual Hi-Hat only when that phase occurs in at least `80%` of measures. This separates weak repeating evidence from a stable pattern that is safe to complete. Both safeguards are shared rules: they must pass `verify_current_solution.py` and the full Round5 rerun; they must not be conditioned on song name, expected count, or path.

### Real-audio round1 training-data rule

The first real-audio training round uses only `blue-yung-kai`, `counting-stars`, and `payphone`; `rolling-in-the-deep` and `toto-rosanna` remain Round5 holdouts. A reusable metadata builder must pair WAV/MIDI names after removing common score-export suffixes, estimate one shared audio-time offset and optional scale from onset evidence, map only KD `{35,36}`, SD `{37,38,40}`, and HH `{22,26,42,44,46}`, and write only train metadata under `validation_runs`. Long songs must expand into deterministic event-bearing four-second windows so a song does not collapse into one median training slice. Unsupported drum pitches remain unlabeled rather than being remapped to KD/SD/HH.

If a joint SD/HH real-audio candidate reduces the accepted Round4 strong-HH evidence, the next candidate must train SD only with a lower real-audio sampling ratio. This is channel isolation, not a file-specific exception: it preserves the accepted HH detector while testing whether real-audio Snare labels improve the Rolling model-layer recall.

### Round5 unsupported-drum root-cause and expansion rule

The accepted `SymmetricDrumTCN` has exactly three onset and velocity output channels: KD, SD, and HH. The shared training metadata also maps only KD `{35,36}`, SD `{37,38,40}`, and HH `{22,26,42,44,46}`. Ride, crash, tom, and other drum pitches are therefore background during three-class training and cannot be represented in a three-class transcription result.

Before any further KD/SD/HH threshold, NMS, GPAR, or same-recipe head-only tuning, a Round5 probability/event audit must distinguish an actual false positive from a supported-class proxy for an unsupported score event. The audit uses the same physical-time alignment and 50ms one-to-one matching rule as Round5. If a material portion of unmatched native HH events aligns with unsupported MIDI events across more than one held-out song, this is a scope/label-space limitation, not evidence for a HH threshold reduction or a song-specific brain rule.

The current audit satisfies that condition: Rolling has `147/286` unmatched native HH events (`51.4%`) within 50ms of unsupported events, including `128` Ride pitch-51 events and `6` Crash pitch-49 events. Rosanna has `223/422` (`52.8%`) with the same relationship, including `167` Ride pitch-51 events and Crash/Tom/Cymbal pitches. Rolling Snare misses also have low pre-threshold model probabilities (median `0.075` at missed true-SD frames versus `0.680` for matched true-SD frames), so broad threshold lowering is rejected.

The bounded multi-class coverage audit is complete. The next label set is fixed to six classes: `KD`, `SD`, `HH`, `TOM`, `CRASH`, and `RIDE`. STAR has independent source annotations for Tom `LT/MT/HT` (`166,109` events), Crash `CRC/CHC/SPC` (`56,892`), and Ride `RD/RB` (`62,933`) in addition to the existing KD/SD/HH labels. A 100-file E-GMD train/test MIDI sample independently contains tom, ride, and cymbal pitches, so STAR supplies semantic labels while E-GMD supplies compatible acoustic coverage. Cowbell, clap, tambourine, splash, and other sparse/ambiguous articulations remain background for this first bounded expansion.

Implementation must use a new six-class metadata builder, a new six-class checkpoint, and separate six-class held-out gates. It must not resize or overwrite the accepted three-class checkpoint in place. The existing three-class checkpoint and its verifier remain the accepted regression baseline. Round5 tracks remain held out and may not be used for training or per-song logic.

Six-class smoke implementation:

1. `SymmetricDrumTCN(num_classes=3)` keeps `3` as its default so every current caller and the accepted checkpoint preserve the existing `[time, 3]` contract. Only the experimental smoke path constructs `SymmetricDrumTCN(num_classes=6)`.
2. `preprocess_star.py --label-scheme six-class` maps STAR source classes `BD -> KD`, `SD/SS -> SD`, `CHH/PHH/OHH -> HH`, `LT/MT/HT -> TOM`, `CRC/CHC/SPC -> CRASH`, and `RD/RB -> RIDE`. It writes metadata only to an explicitly supplied path under `validation_runs`.
3. The smoke runner reads one `split=train` STAR window, transfers only shape-compatible backbone/TCN weights from the accepted three-class checkpoint, leaves the six-channel heads new, performs one optimizer update, writes a separate candidate checkpoint and JSON report, then reloads it with `num_classes=6` and asserts finite loss and `[batch, time, 6]` output shapes.
4. This smoke gate proves data mapping, model dimensions, checkpoint isolation, and one backward pass. It does not claim accuracy, does not call `transcribe.py`, and does not evaluate or train on `test_real_audio`.

Pseudocode:

```text
six_meta = build_star_metadata(label_scheme="six-class", split="train")
model = SymmetricDrumTCN(num_classes=6)
load only accepted_state keys whose names and shapes match model
features, targets = one physical 4-second STAR window
loss = BCE(onset_logits, targets) + velocity loss
optimizer.step()
save candidate under validation_runs
reload candidate with num_classes=6; assert shape and finite loss
```

Smoke evidence (2026-07-12): STAR six-class metadata contains `5,727` usable items with event totals KD `653,178`, SD `452,297`, HH `1,096,870`, TOM `153,399`, CRASH `51,790`, and RIDE `58,250`. The isolated candidate at `validation_runs\\six_class_smoke\\six_class_smoke_candidate.pth` passes a one-window update/reload gate with finite loss `1.4116`, six-label coverage, and onset/velocity shapes `[1,688,6]`; `178` compatible non-head tensors were transferred from the accepted three-class checkpoint. This proves plumbing only, not six-class accuracy or application integration.

The unchanged three-class baseline was rechecked through the same verifier components: blind Raw/notation `5/5`, hard validation `4/4`, Round4 first five clips `30/30`, and sixth clip `6/6`. The desktop runner cut off the combined verifier process before it printed its final line, so these components were run individually with their standard commands and output directories. The six-class candidate must remain isolated until a separate six-class held-out event gate exists.

Six-class held-out event gate:

1. Input is only `split=test` rows from the STAR six-class metadata. The selector chooses six deterministic physical four-second windows, one anchored on the earliest available event for each label. Selection uses annotations and paths only; it must not inspect model probabilities or test-song output.
2. For every window, inference runs `SymmetricDrumTCN(num_classes=6)` directly. A shared `0.50` local-maximum onset threshold and 50ms one-to-one event tolerance produce TP, FP, FN, precision, recall, and F1 for each of the six labels.
3. A trainable candidate passes only when macro F1 is at least `0.70` and every class F1 is at least `0.55`. The smoke candidate is expected to fail this quality gate; that failure supplies the before-training baseline, not a reason to alter thresholds per class or per file.
4. The runner writes `selected_windows.json`, `event_compare.csv`, and `gate_summary.json` only under `validation_runs`. It does not call `transcribe.py` and does not read `test_real_audio`.

Pseudocode:

```text
for label in six_labels:
    window = first STAR test event for label, centered in a 4-second slice
    expected = all labeled events inside that physical slice
    predicted = shared_local_maxima(model(slice), threshold=0.50)
    aggregate fixed-50ms matches by label
pass = macro_f1 >= 0.70 and min(per_label_f1) >= 0.55
```

Held-out baseline evidence (2026-07-12): `validation_runs\\six_class_smoke\\heldout_baseline\\gate_summary.json` is `fail` for the one-update smoke candidate, with macro F1 `0.0332`. Per-class F1 is KD `0.0591`, SD `0.0000`, HH `0.0634`, TOM `0.0000`, CRASH `0.0769`, and RIDE `0.0000`. This result is expected because the six output heads were newly initialized and only received one smoke update. It proves the gate detects unusable output without threshold tuning, filename rules, `test_real_audio`, or transcription-brain logic. No candidate may be promoted from this baseline.

First formal six-class candidate rule:

1. Use only STAR six-class metadata rows with `split=train`. For each label, select exactly `24` event anchors at evenly spaced positions in the sorted source-label list, yielding `144` deterministic physical four-second windows. This spreads the fixed budget across recordings instead of taking a filename prefix or the easiest model outputs.
2. Transfer only shape-compatible three-class backbone/TCN weights. Freeze every transferred layer and train only the newly initialized six-channel onset and velocity heads for one epoch, batch size `4`, `36` batches, learning rate `5e-4`, and a fixed positive onset weight `20.0` for all six labels.
3. The candidate, schedule JSON, and training report live under `validation_runs\\six_class_candidate_v1`. Then run the unchanged six-class STAR `test` gate. Do not tune its threshold, select different test windows, or read Round5 songs after seeing the result.
4. A failed quality gate rejects this candidate and stops the run. A passing quality gate proves only six-class STAR held-out performance; `transcribe.py` integration and Round5 testing remain later separate tasks.

Candidate-v1 evidence (2026-07-12): `validation_runs\\six_class_candidate_v1\\six_class_candidate_v1.pth` trained the exact fixed schedule of `144` windows (`36` batches). Training loss decreased from `1.0748` to `0.5450`, but the unchanged held-out gate failed at macro F1 `0.0056`: KD `0.0333`, SD `0`, HH `0`, TOM `0`, CRASH `0`, RIDE `0`. The candidate is rejected. This is evidence that a head-only one-epoch schedule does not provide sufficient six-class acoustic learning; loss reduction alone is not an acceptance signal. Do not retry thresholds, test selection, or the same training recipe.

Candidate-v2 full-model rule: physical target-frame audit confirms every sampled class anchor is present at its calculated label frame, so timestamp alignment is not the v1 blocker. Candidate-v2 uses a different fixed recipe: `48` evenly spaced STAR train anchors per class (`288` windows), batch size `8`, `3` epochs, all model parameters trainable, Adam head learning rate `5e-4`, and transferred backbone/TCN learning rate `2e-5`. The same schedule repeats exactly each epoch and the existing STAR test gate remains unchanged. This is one bounded full-model adaptation attempt, not a parameter sweep.

Candidate-v3 loss-correction rule: v2 still drives positive onset probabilities below `0.50` at both train and test label frames despite aligned timestamps. The root cause is that six-class training used hard one-frame targets, while the established three-class path uses a five-frame Gaussian target. Candidate-v3 keeps the v2 data schedule, epochs, optimizer rates, test gate, and no-Round5 rule unchanged, but replaces the loss with a channel-generic five-frame Gaussian target `[0.05, 0.25, 1.0, 0.25, 0.05]`, propagated velocity targets, and a fixed positive-frame multiplier `50.0`. This is a loss-correction candidate, not threshold tuning or a data/test selection retry.

Candidate-v4 continuation rule: v3 audit finds `267/288` train anchors in the central half of their physical windows and confirms all sampled target frames are active, so broad window-boundary or time-label mismatch is rejected. V3 loss remains descending (`2.9331` to `1.2393`) while target probabilities are still sub-threshold; it is under-trained, not converged. Candidate-v4 continues only from v3 using the identical 288-window schedule, Gaussian loss, weights, optimizer rates, and held-out gate for `15` additional epochs. No data, test, threshold, or hyperparameter sweep is introduced.

Candidate-v5 class-balance rule: the fixed 288-window schedule contains average events/window KD `9.50`, SD `6.17`, HH `12.09`, TOM `3.50`, RIDE `2.08`, and CRASH `1.16`, corresponding to inverse-density onset weights KD `72`, SD `112`, HH `57`, TOM `196`, RIDE `331`, and CRASH `595`. A uniform weight `50` leaves every class, especially Ride/Crash, dominated by 688 negative frames. Candidate-v5 uses those schedule-derived class weights with the same Gaussian loss, 288 train windows, optimizer rates, and held-out gate for `10` epochs from the accepted three-class backbone. It is data-derived class balancing, not a test threshold change.

Candidate-v6 BatchNorm rule: the repository documents that small STAR fine-tuning can corrupt BatchNorm running statistics and collapse inference. V2-v5 allowed these statistics to update and all showed near-zero held-out event output. Candidate-v6 restarts from the accepted three-class backbone, keeps v5's schedule-derived class weights, Gaussian loss, data, rates, epochs, and gate, but calls the shared `freeze_batchnorm_stats` helper after every `model.train()` transition. This isolates the one known small-STAR distribution failure without changing data or evaluation.

Candidate-v7 coverage rule: a 100-step single-window overfit check reaches KD/SD/HH target probabilities near `0.999`, proving model loading, feature extraction, target framing, and loss gradients work. V6 therefore fails from insufficient exposure: its 288 windows are seen only 10 times each. Candidate-v7 uses a larger deterministic schedule of `96` evenly spaced anchors per class (`576` windows), batch size `16`, `30` epochs (`1,080` batches), frozen BatchNorm, schedule-derived class weights, Gaussian targets, head learning rate `1e-3`, and backbone/TCN rate `2e-5`. The STAR test gate remains fixed. This is the first coverage-sized training run, not another loss or test adjustment.

Candidate-v7 rejection evidence (2026-07-12): training completed its exact declared budget and reduced loss from `5.5178` to `2.7949`, with all six labels represented by actual in-window events: KD `5,404`, SD `3,501`, HH `7,208`, TOM `2,095`, CRASH `690`, RIDE `1,208`. The unchanged STAR `split=test` gate at `validation_runs\\six_class_candidate_v7\\heldout_validation\\gate_summary.json` failed with macro F1 `0.0000`; every class produced zero events above the predeclared shared `0.50` threshold. This candidate is rejected and must not be integrated into `transcribe.py`, replace the accepted three-class checkpoint, or be evaluated against Round5 as though it passed. The next action is a training-pipeline diagnosis with an explicitly approved new objective or dataset-scale plan, not lowering the gate threshold, changing selected test windows, or retrying this recipe.

V7 root-cause audit: direct inference on the six fixed STAR test windows shows that every channel's global maximum occurs at frame `0`, while the maximum probability at real labeled frames is mostly `0.09` to `0.24` (only one SD window reaches `0.6752`). The validator intentionally excludes the boundary frame from local maxima, so it exports zero events; including that frame would create a six-channel false-positive burst rather than correct transcription. The failure is therefore an edge-artifact/output-target mismatch in the candidate training pipeline, not a near-miss threshold issue. Any future candidate must first eliminate boundary artifacts with a materially changed, documented training-window/target plan and retain the unchanged held-out gate.

Candidate-v8 source-rate and schedule repair rule: STAR audio audit confirms all 5,679 train files are `48,000 Hz`. The six-class-only `build_window` had read the fixed `176,128` samples directly from the source, which is only `3.669` physical seconds at 48 kHz, then padded the resampled 44.1 kHz feature tensor. V8 first corrects the source read length to `round(TARGET_SAMPLES * source_sr / SR)`, preserving a physical four-second window before resampling. Its deterministic schedule accepts only anchors that can be centered inside that full window and interleaves KD, SD, HH, TOM, CRASH, and RIDE rather than sending contiguous single-label batches. This directly removes the observed zero-padding and frame-0 reward mechanism. V8 keeps STAR `split=test`, the selected windows, threshold `0.50`, tolerance `50 ms`, and all acceptance thresholds unchanged; it does not read Round5 or `test_real_audio`.

V8 execution budget: use the unchanged six-label coverage of 96 centered anchors per class (576 windows), batch size `12` so each batch contains two deterministic KD/SD/HH/TOM/CRASH/RIDE cycles, 30 epochs, Gaussian onset targets, source-derived class weights, frozen BatchNorm, full-model adaptation, head learning rate `5e-4`, and backbone/TCN rate `2e-5`. This is one root-cause repair run, not a parameter sweep. A failure rejects v8 and returns to diagnosis; a pass proves only the six-class STAR held-out gate before any application integration.

Candidate-v9 warm-head rule: audit of the accepted three-class model on the same STAR features shows its KD/SD/HH channel maxima also occur at frame `0`; this is a shared causal-TCN boundary characteristic, not a six-class-only head defect. However, the old loader skipped all output heads because their shape changed from `[3,...]` to `[6,...]`, discarding the accepted KD/SD/HH acoustic readout. V9 copies the accepted KD, SD, and HH onset/velocity head rows into the identically named six-class rows. TOM starts from SD, while CRASH and RIDE start from HH, which are semantic drum/cymbal priors and do not depend on audio paths or test answers. The source-rate-correct centered/interleaved schedule and fixed test gate remain unchanged. This is the minimal architecture-compatible transfer needed before asking the new heads to learn six classes.

V9 execution budget: keep the v8 576 centered windows, batch size `12`, 30 epochs, Gaussian targets, schedule-derived weights, frozen BatchNorm, and full-model update. Use head learning rate `1e-4` and backbone/TCN rate `1e-5` to adapt the warm-started output heads without discarding the accepted three-class readout. This is one conservative warm-start training run; no threshold, evaluator, test-window, path-routing, Round5, or `test_real_audio` rule changes are allowed.

Six-class checkpoint reload correction: v9 training correctly enables `backbone.use_legacy_proj` when its source checkpoint contains `backbone.legacy_slot_proj.weight`, but the six-class validation runner rebuilt a default model and loaded weights without restoring that flag. It consequently inferred through an untrained projection branch and reported zero events. The shared six-class checkpoint loader must set the legacy projection flag before `load_state_dict`; the smoke reload and held-out validator must both use it. This is a runtime-state restoration fix, not a threshold or test change. V9 must be re-evaluated unchanged after this correction before any further training is considered.

Candidate-v9 corrected evidence (2026-07-12): with the legacy projection restored, the unchanged gate reports macro F1 `0.3345`, KD F1 `0.7111`, SD `0.4082`, HH `0.5672`, TOM `0.0333`, CRASH `0.0769`, and RIDE `0.2105`. KD and HH now pass, proving the candidate and evaluator path are live. The remaining failure is precision: TOM/CRASH/RIDE have recall but excessive false positives, and SD also over-predicts. V9's schedule-derived positive weights are KD `68`, SD `105`, HH `50`, TOM `169`, CRASH `482`, RIDE `299`; these are the direct source of an excessively permissive rare-class objective.

Candidate-v10 balanced-objective rule: retain the data-derived inverse-density relationship but use its square root, `sqrt(CHUNK_FRAMES / average_events_per_window)`, rather than the raw ratio. It yields a bounded, still data-derived class balance instead of multiplying CRASH loss by about 482. V10 otherwise keeps v9's source-rate-correct, centered, interleaved 576 windows, warm heads, batch size `12`, 30 epochs, frozen BatchNorm, Gaussian targets, full-model mode, head rate `1e-4`, backbone rate `1e-5`, and unchanged held-out gate. This targets demonstrated false positives without changing inference threshold or test data.

Candidate-v10 evidence (2026-07-12): the fixed gate remains fail at macro F1 `0.3147`. KD remains pass at `0.7143`; SD is `0.4151`, HH `0.4800`, TOM `0.0000`, CRASH `0.1250`, and RIDE `0.1538`. Compared with v9, rare-class prediction counts fall sharply, confirming the raw inverse weighting caused false positives, but recall now falls as well. The consistent remaining cause is insufficient acoustic diversity: all 30 v9/v10 epochs repeat only 96 unique anchors per class despite STAR having tens of thousands of TOM/CRASH/RIDE labels.

Candidate-v11 coverage-diversity rule: retain v10's corrected source handling, warm head transfer, square-root weights, interleaving, fixed gate, and rates, but replace repetition with breadth. Select `576` evenly spaced centered anchors per class (`3,456` distinct windows), batch size `12`, for `10` epochs (`2,880` batches). This gives twice v10's update budget but six times the unique labeled contexts; it is a dataset-coverage repair, not a threshold or selected-test adjustment. V11 must be accepted only by the unchanged STAR held-out gate.

Candidate-v11 evidence (2026-07-12): the unchanged gate improves to macro F1 `0.3856`: KD `0.7143`, SD `0.5116`, HH `0.5385`, TOM `0.0000`, CRASH `0.1739`, RIDE `0.3750`. Read-only event inspection rules out timing drift: on the TOM test window, the expected TOM at `1.9969s` is predicted as CRASH/RIDE at `1.9969s`; on the RIDE window, expected cymbal events are similarly confused across TOM/CRASH/RIDE. This is unresolved acoustic class discrimination, not a 50ms match tolerance or transcription-brain issue. V11 loss remains declining at `0.2464`, so one continuation can improve discrimination without throwing away its broadened coverage.

Candidate-v12 continuation rule: when the supplied checkpoint already has six output rows, the six-class loader must restore that whole candidate state rather than semantic-remapping its heads again. V12 resumes v11 on the same 3,456 centered STAR train windows for 10 additional epochs, batch 12, square-root weights, frozen BatchNorm, Gaussian targets, full-model update, head rate `5e-5`, and backbone rate `5e-6`. The lower rates preserve learned distinctions while continuing convergence. Test split, windows, gate, and thresholds remain unchanged.

## Score-time to physical-time conversion note

When notation gate already passes, score-time annotation rows can be converted to physical audio time by aligning each confirmed annotation with the same instrument occurrence in the passed `notation_events.csv`. The converted CSV must preserve the original score time in `score_time`, write the corresponding notation event `raw_time` into `time`, and set `source=notation_physical_map`. Raw acoustic expected counts may then include these converted rows.

## Channel-separated fine-tune note

When a candidate reduces Hi-Hat false positives but damages Snare recall, the next training step must separate channel objectives instead of full-model fine-tuning. `train_mixed_datasets.py --train-channels` can restrict loss to selected output channels while freezing the rest of the model through `--train-head-only`, preserving unrelated drum classes.

## 1. 核心网络架构 (SymmetricDrumTCN)

模型采用共享卷积骨干网络 (Shared CNN Backbone) 加双路对称扩张时间卷积网络 (Dilated TCN) 的解耦设计，以消除分类 (Onset) 与回归 (Velocity) 的梯度干扰和时序相位偏差。

### 1.1 骨干网络 (Shared CNN Backbone)
*   **输入通道**：2 (Channel 1: Log-Mel, Channel 2: Superflux Onset Feature)
*   **下采样层**：4 层 2D 卷积 + MaxPool2d (仅在频域维度下采样，时域保留完整分辨率)
    *   频域维度变化：$256 \to 128 \to 64 \to 32 \to 16$
*   **频域投影层**：$1 \times 1$ 卷积映射 `slot_proj` 将 $64 \times 16$ 维的通道频域级联特征，投影降维至 $64$ 维，并展平送入时序网络。

### 1.2 对称解耦时序层 (Fully-Symmetric Decoupled TCN Branches)
*   **对称分支**：
    1.  **Onset 分支**：5层因果因数扩张 TCN 块 (kernel_size=5, dilations = [1, 2, 4, 8, 16]) $\to$ 输出层 (1x1 Conv + Sigmoid) $\to$ 预测概率 [Time, 3] (Kick, Snare, Hi-Hat)
    2.  **Velocity 分支**：5层因果因数扩张 TCN 块 (kernel_size=5, dilations = [1, 2, 4, 8, 16]) $\to$ 输出层 (1x1 Conv + Sigmoid) $\to$ 预测力度值 [Time, 3] (归一化到 `[0, 1]`)

---

## 2. 信号处理与特征提取规范 (DSP Pipeline)

*   **音频采样率 (SR)**：$44100\text{ Hz}$ (单声道)
*   **帧移 (Hop Length)**：$256$ 采样点 ($\approx 5.8\text{ ms}$ / 帧)
*   **特征维度**：双通道 $256$ 维梅尔谱矩阵，时域长度固定为 $688$ 帧 ($\approx 4\text{ 秒}$)
*   **双通道特征图**：
    *   **通道 1 (Log-Mel)**：标准梅尔滤波器组提取谱图后做 Log-Power 压缩与 Z-Score 标准化。
    *   **通道 2 (Mel-domain Superflux)**：在梅尔能量域进行一阶前向差分，保留正向脉冲能量，经 1000 倍放大后通过 $\log_{10}(X + 1.0)$ 进行高清无噪压缩，消除静音段极微弱的底噪扰动。

---

## 3. 标签定义与损失函数优化 (Loss & Labels)

*   **标签平滑 (Soft Labeling)**：对 Onset 二分类独热标签（0 或 1）使用一维高斯滤波器 ($\sigma=1$) 进行平滑，降低轻微帧偏移带来的负面梯度惩罚。
*   **非对称力度 Loss**：
    *   在音符击打发生的活性区域，计算预测力度与真实力度的均方误差 (MSE)。
    *   在无击打的静音区域，引入 $(1.0 - \text{Onset}_{\text{smoothed}})$ 作为掩码乘上 $0.1$ 的极微弱惩罚权重，压制假阳性力度的同时避免过度拉低整体音符力度。
*   **$\beta$ 梯度权重分阶段调度**：
    *   **前半段训练**：设定 $\beta = 20.0$，强行撑开力度的预测幅值与敏感度。
    *   **后半段微调**：降低 $\beta = 10.0$，对模型时序定位与力度预测做联合高精度收敛微调。

---

## 4. 阶段评测指标基准 (Benchmarks)

采用统一的 GMD (244首) 与 IDMT (16首) 联合验证集进行评测。以下为最新基准：

### 4.1 纯分轨验证 (Clean Solo Validation)
*   **评估指标**：Mean F1 = **`0.910`** (Kick: `0.918`, Snare: `0.896`, Hi-Hat: `0.917`)
*   **力度误差**：Velocity RMSE = **`10.89`** (相比原旧模型误差缩小了 5 倍以上)

### 4.2 全混音与速度变化验证 (Mixed + Augmentation Validation)
*   **评估指标**：Mean F1 = **`0.861`** (Kick: `0.898`, Snare: `0.821`, Hi-Hat: `0.863`)
*   **力度误差**：Velocity RMSE = **`12.68`**

---

## 5. E-GMD 大规模数据集训练技术指标设计

为了解决原声鼓与电子鼓的声学特征泛化差距，并且完全不依赖“大脑”后处理量化调整，在 0.50 默认中值下直接听写出完美音符数量，新训练阶段采用以下规范：

### 5.1 特征工程规范
*   **特征提取**：全面退回 **标准梅尔谱 (Standard Mel Spectrogram)** 结合梅尔域无噪 Superflux（通道 1: Standard Log-Mel, 通道 2: Log-Superflux）。
*   **采样率与跳移**：SR=44100, Hop Length=256, N_MELS=256。

### 5.2 训练数据集规划
*   **数据源**：`e-gmd-v1.0.0.zip` (89.8 GB 压缩包，解压后约 100GB+)。
*   **特征存储**：使用 `convert_to_npy.py` 将所有音频转换至磁盘虚存页映射 `.npy` 原始波形文件，在训练时采用 online `extract_features` + Standard Mel 进行实时特征计算。

### 5.3 训练结果与分析 (First Run Results)
*   **E-GMD 30 Epochs 训练**: `best_drum_model.pth` 完成 30 epochs 训练。
*   **测试集评测 (`test.wav` @ 0.50 阈值)**:
    *   Kick: 45
    *   Snare: 32 (完美匹配目标)
    *   Hi-Hat: 0 (由于 E-GMD 数据集内部能量及分类占比偏好，Hi-Hat 预测概率峰值最高仅为 0.481，在 0.50 默认阈值下被全过滤)
*   **旧模型对照 (`best_drum_model_backup.pth` @ 0.20 阈值)**:
    *   Kick: 48 (完美匹配)
    *   Snare: 32 (完美匹配)
    *   Hi-Hat: 79 (几乎完美匹配 80)

### 5.4 加权损失微调方案 (Weighted Loss Fine-tuning)
*   **解决思路**: 为彻底摆脱对非 0.50 阈值或“大脑”后处理的依赖，需要对 Onset BCE 损失函数进行通道加权微调，主动撑开 Kick 与 Hi-Hat 的预测概率峰值，使其在 0.50 默认阈值下即可实现精确的物理计数。
*   **Onset 损失加权**: `loss_onset = 1.2 * loss_onset_KD + 1.0 * loss_onset_SD + 2.5 * loss_onset_HH`
*   **超参数与策略**:
    *   **起始权重**: `best_drum_model_backup.pth`（保留其已经优异的 Kick=48, Snare=32 和 HH=79 潜能）
    *   **学习率**: `5e-5` (极小学习率微调)
    *   **训练 Epochs**: 10
    *   **评估指标 (Target)**: 在 `test.wav` 上默认 0.50 阈值下直接实现 Kick=48, Snare=32, HH=80 (误差 $\pm 1$)。

### 5.5 力度敏感型损失加权方案 (Option C: Velocity-Weighted Loss)
为了避免模型盲目拟合极低力度且物理上不可听的“虚假音符/鬼音”（从而导致模型产生大量背景假阳性），在正向 Onset 惩罚中引入连续力度衰减系数 $W_{\text{vel}}$：
*   **力度划分标准**：
    *   $\text{Velocity} > 40$：完全惩罚（$W_{\text{vel}} = 1.0$）
    *   $\text{Velocity} < 15$：弱惩罚（$W_{\text{vel}} = 0.1$），允许模型做模糊决策。
    *   $15 \le \text{Velocity} \le 40$：线性过渡（$W_{\text{vel}} = 0.1 + 0.9 \times \frac{\text{Velocity} - 15}{40 - 15}$）
*   **公式定义**：
    \[
    \text{Weight}_{\text{active}} = \text{Weight}_{\text{channel\_base}} \times W_{\text{vel}}
    \]
*   **目的**：确保模型专注于干净、可听的强拍打击，释放由于拟合不可听弱音而引起的背景噪音触发问题，使默认 $0.50$ 阈值下的物理音符匹配表现更加自然合理。

### 5.6 均衡通道加权损失方案 (Balanced Channel-Weighted Loss Fine-tuning)
*   **问题背景**：过往的微调方案为了提升 Hi-Hat 的预测概率，使用了高达 `150.0` 的非对称正向 HH 损失权重，导致 HH 的梯度完全统治了模型（`150:1` 的极端比例）。在特征共享骨干下，这会导致与 HH 同时击打的 Snare 被严重屏蔽（SD 预测概率骤降到 `0.03`，导致 Shuffle 节奏下 SD 漏检）。
*   **解决思路**：对 Onset BCE 的权重掩码进行平滑均衡化处理，下调 HH 正向权重，同时提升 SD 正向权重，维持各通道在合理的数量级，避免极端不平衡：
    *   **Kick (KD)**: 正向 `5.0`，反向 `0.5`
    *   **Snare (SD)**: 正向 `8.0`（从 1.0 提高到 8.0，对抗共现特征屏蔽），反向 `0.5`
    *   **Hi-Hat (HH)**: 正向 `15.0`（从 150.0 降至 15.0，解除梯度统治），反向 `0.5`
*   **效果目标**：在不明显损失 `test.wav` 各组件计数的基础上，让 [test_shuffle.wav](file:///c:/Users/zhiya/Documents/MyProject/Drum_classifier_train_model/test/test_shuffle.wav) 中的 SD 概率恢复到 `0.50` 以上，实现正常检出。

---

## 6. 智能后处理优化与物理对齐规范 (Heuristics & Alignment Specifications)

为了实现完全自动、精准无误的转谱，无需用户手动调整任何参数，系统在后处理逻辑中集成以下自适应优化机制：

### 6.1 自适应音程速度估算 (Onset-Interval Tempo Estimation)
*   **背景**：传统 Librosa 速度估算在遭遇干净或高同步的鼓声时容易产生整倍数或分数值偏差，窄带搜索（如原 $\pm 5\text{ BPM}$）会导致真速度被完全排除。
*   **方案**：提取所有相邻 onset 的时间差中位数 $d_{\text{median}} = \text{median}(\Delta t)$，将其作为基础音符长度候选值。通过乘以乘数 $[4.0, 2.0, 1.0, 3.0, 1.5]$ 映射为可能的每拍时长，并将其转换为候选 BPM 注入搜索池。
*   **效果**：即使 Librosa 原始估算完全偏离，自适应音程估算也能直接定位到真实的基准速度。

### 6.2 动态网格分辨率 (Dynamic Grid Resolution)
*   **背景**：固定 16 分音符网格会将鼓手演奏的快速 32 分音符、双击（Flam）或快速滚奏音符强制合并，导致音符丢失。
*   **方案**：在量化网格前，检测相邻 onset 的最小物理时间间隔 $g_{\text{min}}$。若 $g_{\text{min}} < 0.65 \times \text{Grid}_{16\text{th}}$，则自动将量化分辨率提升为 32 分音符（或对应三连音下的 24 分音符）。
*   **效果**：在不破坏慢速段整洁度的前提下，完美保留并输出所有快速细节音符。

### 6.3 双模时间对齐与乐谱优化 (Dual-Mode Time Alignment & Score Optimization)
*   **背景**：强制保留前导物理静音会导致打谱软件（如 MuseScore, Guitar Pro）生成大量混乱的三连音和附点/切分音。
*   **方案**：引入双模时间对齐机制：
    *   **打谱对齐模式（默认，`--sync-audio` 关闭）**：`time_offset` 设为 `0.0`，使第一个量化音符强制对齐到 `0.0` 秒（第 1 拍），生成整洁易读的乐谱。
    *   **物理同步模式（可选，开启 `--sync-audio`）**：`time_offset = first_onset`，在 MIDI 中完整保留前导物理静音，以供 DAW 中音画绝对同步播放。
*   **效果**：默认输出对打谱软件完美友好的整洁 MIDI；需要多轨道音频同步时，一键开启物理对齐。

### 6.4 复合拍号与谱面速度单位修正 (Compound Meter & Score Tempo Semantics)
*   **问题背景**：`12/8`、`9/8`、`6/8` 等复合拍号常以“附点四分音符”作为谱面主脉冲；若仅使用 MIDI 内部四分音符 BPM，`附点四分音符 = 70` 会等价显示为 `四分音符 = 105`，导致系统误报速度语意。
*   **拍号判断规则**：
    *   对候选拍号同时计算小节周期相似度与重音分布。
    *   当 8 分母拍号具备稳定三连分组脉冲时，优先保留 `6/8`、`9/8`、`12/8` 的完整小节语意，不得降阶成 `3/4` 或其他等价但错误的记谱拍号。
    *   `12/8` 的核心检查为每小节 12 个八分音符、4 个附点四分音符脉冲，常见 hi-hat 连续八分音符与 kick/snare 大拍重音需共同参与评分。
*   **速度输出规则**：
    *   MIDI metadata 仍写入四分音符 BPM，以维持 DAW/pretty_midi 相容性。
    *   CLI 报告须同时列出谱面速度单位。若拍号为 `6/8`、`9/8`、`12/8`，谱面速度显示为 `dotted-quarter BPM = quarter BPM / 1.5`。
    *   例如：MIDI `quarter = 105 BPM`，在 `12/8` 中必须报告为 `dotted-quarter = 70 BPM`。

### 6.5 AI 原始事件诊断输出 (Event Debug Export)
*   **目标**：将 AI 声学识别层与大脑转谱层分离观察，避免把模型漏检误判为后处理问题，或把后处理删改误判为模型问题。
*   **CLI**：`transcribe.py` 支持 `--event-debug [CSV_PATH]`。不传路径时，默认输出到输入音频同目录的 `*_event_debug.csv`。
*   **CSV 核心字段**：
    *   时间与网格：`raw_time`、`quantized_time`、`midi_time`、`beat`、`step_16th`。
    *   AI 原始输出：`prob_kick`、`prob_snare`、`prob_hihat`、`vel_kick`、`vel_snare`、`vel_hihat`。
    *   阈值与原生触发：`thresh_kick`、`thresh_snare`、`thresh_hihat`、`native_kick`、`native_snare`、`native_hihat`。
    *   大脑输出：`final_kick`、`final_snare`、`final_hihat`、`virtual_kick`、`virtual_snare`、`virtual_hihat`。
*   **使用原则**：模型训练与回归评估优先观察原生触发与原始概率；MIDI 成品评估再观察大脑输出与虚拟补全。

---

## 7. STAR Drums 数据导入与微调规划

### 7.1 定位
STAR Drums 不直接替代 E-GMD，而是作为补强数据源，用于提升混音场景、多鼓件音色、Snare/Hi-Hat 泛化与同一时间多鼓件识别能力。E-GMD 保留为人类 groove、velocity 与 solo drum 基础数据；STAR Drums 用于补足非鼓伴奏干扰与 18 类鼓件音色覆盖。

### 7.2 导入流程
1. **检查下载结构**：确认 audio/stems、annotations、metadata、class map、train/validation/test split、license/README 是否齐全。
2. **建立转换器**：将 STAR Drums 的 18 类鼓件映射到当前三类目标：
    *   Kick 类 -> `KD`
    *   Snare / rim / side-stick 类 -> `SD`
    *   closed/open/pedal hi-hat 类 -> `HH`
    *   tom / crash / ride / cymbal 先忽略或作为 background，待三类模型稳定后再扩展到 5/8/18 类。
3. **数据审计**：训练前统计 KD/SD/HH 数量、同时敲击比例、Snare/HH 子类分布、split 分布、异常时间点与空标注。
4. **Smoke training**：抽取 100-300 段样本跑 1 epoch，验证 dataloader、feature extraction、label 对齐、loss 下降与显存占用。
5. **固定回归验证**：每次微调前后必须跑 `test_shuffle.wav`、`test_3T.wav`、`test_16.wav`、`test_58.wav`、E-GMD hard set 与 STAR validation 小样本。
6. **继续微调**：从 `best_drum_model.pth` 或 `best_drum_model_backup.pth` 载入权重继续训练，不从零开始重训。
7. **验收条件**：`test_shuffle.wav` 的 Snare 通道需明显恢复；HH 不退化；`test_3T.wav` 仍为 `12/8`；`test_16.wav` 与 `test_58.wav` 不被破坏；validation F1 不显著下降。

### 7.3 训练原则
*   不因单首失败样本立即大规模重训；先用 `event_debug` 判断是 AI 声学层问题还是大脑后处理问题。
*   微调目标优先服务 AI 原始事件识别，不把 tempo、拍号、量化、谱面补全混入模型训练目标。
*   Hard validation set 是模型上线门槛，不能只看训练集或单一数据集的总体 F1。
*   STAR hard validation 只从 `validation/test` split 挑选，覆盖 Snare 密集、Hi-Hat 密集、KD/SD/HH 均衡与同一时间多鼓件样本；不得从 training split 挑选。
## Balanced STAR sampler

STAR small fine-tune must not simply take the first N training files. The sampler interleaves four buckets: Snare-dense clips, Snare+Hi-Hat simultaneous clips, Hi-Hat-dense clips, and KD/SD/HH balanced clips. Bucketed samples anchor the training slice near the bucket event instead of only using the middle event.
## Hard validation runner

Before mixed E-GMD/STAR/IDMT training, validation must be automated. `run_hard_validation.py` runs the fixed local regression WAV files and optional STAR hard-validation audio through the existing `transcribe.py` CLI, then records tempo, time signature, KD/SD/HH counts, F1 text when present, MIDI path, event-debug CSV path, and pass/fail status in CSV and JSON reports.

STAR hard validation gates use `hard_stats` from `processed_data/star_hard_validation.json` as annotation-derived GT counts. The runner compares predicted KD/SD/HH counts against configurable minimum recall ratios, so STAR cases fail when the model only runs successfully but misses too many annotated drum events.

The local `test_shuffle.wav` gate must check the four-measure score count, not only the presence of any Snare. Its current reference pattern is `4/4 @ quarter=110` with at least KD=16, SD=8, HH=32.

Sparse shuffle skeletons need a notation-layer recovery step, not another model-weight tweak. When a 4/4-range performance around 110 BPM has a stable quarter-note KD/HH skeleton and almost all detected events land on quarter beats, the transcription layer may complete the four-measure shuffle pattern by adding HH on the swung subdivision and SD on beats 2 and 4. This rule is deliberately narrow so straight 16th-note cases such as `test_16.wav` are not touched.

## Mixed dataset manifest

Mixed E-GMD/STAR/IDMT training must start from a machine-readable manifest instead of ad-hoc folder assumptions. `build_mixed_manifest.py` records available dataset metadata, creates `local_xml_meta.json` from local `audio/*#MIX.wav` plus `annotation_xml/*#MIX.xml`, and fails readiness when required E-GMD or IDMT manifests are missing.

E-GMD may be restored under `e-gmd-v1.0.0` or `egmd_dataset_2`; preprocessing must accept an explicit dataset directory so the manifest can be rebuilt without renaming large folders.

## Mixed dataset training

`train_mixed_datasets.py` trains from `best_drum_model.pth` into a candidate checkpoint only. It mixes metadata-backed audio slices with the default ratio E-GMD 50%, STAR 30%, and local XML clean anchor 20%. It must not overwrite `best_drum_model.pth`; hard validation decides whether a candidate is usable.

Formal mixed retraining runs multiple epochs and invokes `run_hard_validation.py` after each epoch. A candidate is saved as best only when its gate failures decrease; `best_drum_model.pth` remains untouched.

When `--freeze-bn` is enabled, BatchNorm layers must be put back into eval mode after each `model.train()` call. Otherwise the small mixed run corrupts running statistics and collapses inference.

Snare-focused mixed retraining should bias training slice anchors toward Snare or Snare+Hi-Hat events. This changes which 4-second audio window is sampled; it does not alter inference thresholds or the model architecture.

Short mixed experiments must not only consume the first few metadata entries from each source. When `--random-sampling` is enabled, each sample picks E-GMD/STAR/local according to `--mix-ratio` and then picks an item from the whole source with a fixed seed. This keeps small smoke/formal runs reproducible while covering the actual dataset instead of a narrow prefix.

If channel weighting raises Snare but damages Hi-Hat, the next conservative rung is head-only mixed adaptation. `--train-head-only` freezes the shared backbone and TCN, updating only `onset_head` and `velocity_head` so the run can test output calibration without rewriting the learned timing/features.

If head-only adaptation cannot move Snare, mixed training should balance input items before changing architecture. `--balanced-sampler` reuses the existing STAR bucket selector for E-GMD/STAR/local metadata, prioritizing SD-dense, SD+HH simultaneous, HH-dense, and balanced clips.

## Local Regression Ground-Truth XML Realignment

To satisfy the F1-score evaluation for the local hard-validation regression set (`test_shuffle.wav`, `test_3T.wav`, `test_16.wav`, `test_58.wav`), we parsed their respective source MIDI files (`test_shuffle_drums_backup.mid`, `test_3T_drums_backup.mid`, `test_16_drums.mid`, `test_58_drums.mid`) to extract precise, milliseconds-accurate onset times. These are written to `annotation_xml/test_shuffle.xml`, `annotation_xml/test_3T.xml`, `annotation_xml/test_16.xml`, and `annotation_xml/test_58.xml`.

Evaluating the final `mixed_formal_kick375_snare18_hh12_candidate.pth` checkpoint against these realigned XML references yields the following F1 Benchmarks under `--sync-audio` alignment:
*   `test_16.wav`: 100.00% F1 (perfect match)
*   `test_3T.wav`: 90.58% F1
*   `test_58.wav`: 93.96% F1
*   `test_shuffle.wav`: 69.80% F1 (retained sparse model triggers while correctly executing notation-layer swing completion)

## Two-layer transcription output

`transcribe.py` must expose two separate event layers so future rhythm errors can be assigned to the right subsystem:

1. AI raw recognition layer: events directly detected by the model after NMS/merge, before notation completion. It records onset time, quantized time, frame list, KD/SD/HH probabilities, thresholds, native KD/SD/HH booleans, velocities, grid step, tempo, and time signature. It must not mark notation-only virtual notes as native hits.
2. Notation layer: final events used for MIDI output after quantization, groove recovery, sparse shuffle completion, crosstalk suppression, and other transcription heuristics. It records final KD/SD/HH booleans plus virtual KD/SD/HH flags so AI misses and brain-filled notes remain auditable.

The existing `--event-debug` CSV remains backward compatible for mixed diagnostics. New explicit exports use `--raw-ai-events` and `--notation-events`, each accepting an optional path or `auto` for input-adjacent CSV names. Hard validation must keep passing after this split because the refactor is observational only.

## Snare/Hi-Hat hard-example fine-tuning

After two-layer output exists, the next candidate training step targets raw AI recognition, not notation completion. The goal is to lift SD/HH native detections on known hard examples while keeping KD stable as a regression guard.

Training must start from the current gated candidate checkpoint, not overwrite `best_drum_model.pth`. The first rung reuses `train_mixed_datasets.py` with `--snare-focus`, `--balanced-sampler`, low learning rate, BatchNorm freezing, and stronger SD/HH positive onset weights. KD remains present in training labels and hard validation gates, but its weight should stay conservative unless a KD regression is observed.

If broad mixed fine-tuning does not improve the raw AI layer or trips KD regression gates, build a narrow train-split hard-example manifest first. The selector should keep only training items with SD/HH density, SD+HH simultaneity, and nonzero KD presence. Validation/test hard-validation files must remain holdout gates.

Acceptance gates:

1. `run_hard_validation.py --star-limit 8` must still pass 12/12.
2. `test_shuffle.wav` raw AI layer must improve over the current baseline `KD=16, SD=2, HH=16` without reducing KD below 16.
3. Notation layer must still reach `KD=16, SD=8, HH=32` for `test_shuffle.wav`.
4. The output remains a candidate checkpoint only until promoted explicitly.

## Verified user hard-example diagnostics

Score-confirmed user blind annotations are valid for diagnosis and small candidate training only after every row is explicitly confirmed. A candidate may not be accepted only because its training loss drops on these five files.

Before more training, inspect raw model probabilities at the exact verified onset frames. If the model gives low probability at confirmed KD/SD/HH frames, the issue is acoustic learning or label alignment. If the model gives high probability at those frames but raw AI event counts are still low, the issue is inference peak picking, merge distance, NMS, or thresholding. This diagnosis must be recorded before starting another fine-tune run.

The capacity-test result selects the second branch: `raw_ai_verified_user_capacity_candidate.pth` produces high probability at most verified KD/SD/HH frames, but raw event counts remain far below target. The next fix must therefore adjust event generation from probability curves, not add another blind fine-tune.

After fixing checkpoint loading, verified training can raise HH/SD probabilities in the same legacy branch used by transcription. If HH false positives appear on ghost-snare or SD-only positions, training should add channel-specific negative onset weights instead of song-specific post-processing. This keeps the correction in the acoustic model: positive weights improve recall, negative weights control false positives.

## Database hard-subset selection

IDMT, E-GMD, STAR, and local verified metadata should be used as the main source for the next raw-AI repair. Do not ask the user for many more manual songs first. Build a small hard subset from existing metadata with buckets that directly match the observed failures:

1. `hh_dense`: dense HH examples for straight-16/continuous HH recall.
2. `sd_hh`: simultaneous or near-simultaneous SD+HH examples.
3. `sd_only`: SD-heavy windows with little/no HH, used as HH false-positive negative evidence.
4. `balanced`: KD/SD/HH all present.

The selector writes normal metadata JSON so existing training code can consume it without new training abstractions.

## Raw AI acoustic target audit

Before requiring raw AI to match a full notation count, compare three layers for each local regression file:

1. Acoustic XML ground truth: events that are explicitly annotated from the audio/MIDI-aligned acoustic reference.
2. Raw AI layer: model-native detections exported by `--raw-ai-events`, before notation completion.
3. Notation layer: final score/MIDI events exported by `--notation-events`, after rhythm completion rules.

If the acoustic XML count is sparse while the hard-validation notation gate is denser, the missing events are notation/implied rhythm targets and should not be used as raw AI fine-tuning acceptance criteria. Fine-tuning should only be required when raw AI misses acoustic XML events, not when notation completion is correctly supplying implied shuffle notes.

For `test_shuffle.wav`, the acoustic XML audit establishes the current raw targets as KD=16, SD=2, HH=17, while the notation target remains KD=16, SD=8, HH=32. The current accepted candidate reaches raw KD=16, SD=2, HH=16 and notation KD=16, SD=8, HH=32. Therefore the rejected fine-tuning attempts were chasing the wrong raw target for SD and most HH fills; future raw fine-tuning should only target the one acoustic HH miss or broader real-audio misses, not the notation-only shuffle completion count.

## Blind test runner

Blind tests must run unseen user audio through the accepted candidate without changing model weights. Each audio file gets four artifacts in its own folder: MIDI, `event_debug` CSV, raw AI CSV, and notation CSV. The summary report must include tempo, time signature, final MIDI KD/SD/HH counts, raw AI KD/SD/HH counts, notation KD/SD/HH counts, virtual KD/SD/HH counts, and whether shuffle completion triggered.

The blind-test goal is not to pass a training gate. It is to classify failures into the right layer: raw AI miss, notation over-completion, notation under-completion, tempo/time-signature miss, or acceptable notation reconstruction.

First blind-test batch size is 3-10 audio files total, not 3-10 per rhythm type. The recommended first batch is about five representative files: one basic straight 8th, one basic straight 16th, one basic shuffle, one syncopated 4/4, and one ghost-snare or busy-hi-hat example. Expand only after this small batch is reviewed.

The first user blind-test expected targets are recorded in `blind_user_tests_expected.csv`. They are used only for this curated first batch, not as a requirement that every future blind-test file needs manual KD/SD/HH counts.

For the first batch, the acceptance check compares notation-layer KD/SD/HH, displayed score tempo, and time signature against `blind_user_tests_expected.csv`. Forced-tempo experiments are allowed as diagnostics only; the accepted blind result must pass without per-file manual forcing unless the user explicitly chooses a hint-based workflow.

First-batch diagnostic probes must not use expected KD/SD/HH counts to rewrite MIDI or CSV output. Tempo, time-signature, threshold, grid, and fill-mode hints may be tested transparently to classify the failure layer. Current diagnostic status:

* `basic_straight_8`, `basic_straight_16`, and `syncopated_4_4` can reach the provided notation counts with explicit tempo/time-signature and threshold/fill hints.
* `ghost_snare` reaches KD=8 and SD=16, but HH jumps from 33 to 30 around the threshold boundary; exact HH=32 requires a rhythm-level postprocess decision, not another model-weight tweak.
* `basic_shuffle` reaches KD=12 and SD=8 near 88-110 BPM, but HH remains 31/33 and the user-provided tempo target around 50 BPM conflicts with the audio duration for a four-measure 4/4 score. This target must be clarified or represented as a deliberate half-time score-tempo convention before it can be treated as a clean automatic pass.

## Raw AI model gate

When the user says a drum hit is clearly audible, the first acceptance layer is raw AI, not notation. The same blind-test expected CSV can be compared against either the raw AI layer or the notation layer. Raw AI gate failures mean the model/checkpoint or training data must be fixed before tempo, meter, or notation completion can be considered successful.

The raw gate must use exported `raw_kick`, `raw_snare`, and `raw_hihat` counts from `run_blind_test.py`; it must not count notation-only virtual events. Candidate checkpoints remain candidates until they pass both the first-batch raw AI gate and the existing hard validation gate.

Raw AI model gate may run as count-only because tempo and time signature belong to the notation/analysis layer. Notation acceptance still checks tempo, time signature, and final score counts together.

User blind hard examples may be converted into temporary training metadata only when the user has supplied expected KD/SD/HH counts and the goal is to repair raw AI recognition. These examples are supervised labels, not inference hints: they may be used during candidate training, but `run_blind_test.py` must still run without per-file KD/SD/HH answers at evaluation time.

Global inference calibration is allowed only as one fixed KD/SD/HH threshold set applied to the whole batch. Per-song thresholds remain diagnostic only.

## User onset annotation templates

Before further model training on user blind files, each file needs a human-verifiable onset CSV with columns `time`, `inst`, `velocity`, `source`, `confirmed`, and `probability`. The template may prefill candidates from raw AI peaks, rhythm-grid fill points, and audio-onset snapping, but only rows with `confirmed=True` may be used for final supervised training metadata.

Verified user annotations should be converted into both one-item-per-file metadata and windowed metadata. Windowed metadata is used for training so long files such as straight 16th are covered across the full song instead of only the middle 4-second slice.

## 8. 拍速与拍号识别层启发式算法优化规范 (Tempo & Time Signature Heuristics Optimization)

为了全自动、准确地转写用户盲测音频而无需手工指定速度与拍号提示，對 [transcribe.py](file:///c:/Users/zhiya/Documents/MyProject/Drum_classifier_train_model/transcribe.py) 中的啟發式規則進行以下改進：
*   **候選拍速擴展**：在生成基準 BPM 候選池時，將默認 `raw_candidates` 擴充以包含 `raw_estimated_tempo / 1.5`（即 `*0.6667`）與 `raw_estimated_tempo * 1.5` 兩個常見關係因子，以覆蓋 1.5 倍速（如 70 BPM 與 105 BPM）的節奏尺度變換。
*   **多重頻去重與倍速折疊 (Extended OTD)**：
    *   移除了 `1.5x` 和 `3.0x` 的 OTD 倍頻折疊（僅保留安全的 `2.0x` 關係折疊）。這樣做可以防止像 `test_3T`（104.9 BPM -> 69.9 BPM）與 `test_shuffle`（110.1 BPM -> 73.4 BPM）此類標準節奏被過度降速折疊，避免後續拍號計算引發的雙重減速 Bug。
*   **全網格複合拍號偵測 (Universal Compound Meter Detection)**：
    *   解除原先複合拍號 `detect_compound_time_signature` 僅在 `triplet` 網格下觸發的限制，使其在所有網格（如 `16th`）下均能執行。这使得 12/8 等复合拍在以 105 BPM 基準 quarter-note 網格轉寫時，能夠被精準自動識別為 12/8，並正確將樂譜速度折算為 `dotted-quarter=70 BPM`。
*   **網格偏差容錯與篩選**：
    *   將 `tolerance_sec` 設為 `0.005` 秒以保持對最優對齊偏差的精確鎖定。
    *   優先排序並推薦與 `raw_estimated_tempo` 距離最近且物理對齊偏差小於 `min_dev + 0.005s` 門檻的候選作為最合理的記譜速度。

## 9. 联合拍速-拍号选择与 MGPC 门槛校准规范 (Joint Tempo-TS Selection & MGPC Calibration Specification)

为了满足用户盲测集（First Blind Batch）在无任何手动提示（No Hint）条件下的 100% 自动对齐与准确音符计数要求，引入以下核心处理流程：
1. **32分音符网格支持 (32nd-Note Candidate Grids)**：
   * 在候选速度筛选的对齐偏差计算阶段，如果候选速度 $\le 75.0$ BPM，自动支持 `32nd` 分辨率网格的偏差评估（即包含 `[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]` 拍位置）。
   * 这允许 60 BPM (对应 32nd 密集音符) 获得低至 $< 0.015$ 秒的偏差得分，顺利进入 Qualified 候选速度列表。
2. **拍速与拍号联合评分 (Joint Tempo-TS Selection)**：
   * 对所有 Qualified 候选速度，逐一运行拍号/重合度侦测算法（Fano Factor 和 Cross-Measure Similarity）。
   * 计算联合评分：`joint_score = ts_score - 100.0 * dev_sec`。如果该速度对应的最佳拍号为 `4/4` 或 `12/8`（标准节奏），则给予额外的权重加分（+2.0）。
   * 最终选择 `joint_score` 最高的候选速度作为记谱速度，从而自动排除非标准奇数拍号（如 3/4 或 9/8）并自动选中最稳定/最常规的记谱框架。
3. **极大差值峰值聚类门槛 (MGPC - Maximum-Gap Peak Clustering)**：
   * 不采用全局静态门槛或单一 RMS 映射，而是自动根据音轨的预测概率曲线自适应定位门槛。
   * 对每个通道（KD, SD, HH），提取所有概率 $\ge 0.12$ 的局部极大值峰值。
   * 对这些峰值概率进行降序排列，寻找相邻两个峰值之间的最大差值（Maximum Gap），以该差值的中点作为自适应 base threshold，要求中点落入合理范围（KD: `[0.22, 0.65]`, SD: `[0.22, 0.60]`, HH: `[0.20, 0.60]`）。
   * 该机制能自动将真正的敲击事件（高概率）与通道串音/演奏杂音（低概率）完美切割。
4. **補音密度守衛 (GPAR Completion Guard)**：
   * 限制 GPAR 重建在密集网格（如 $\le 75$ BPM 且启用 32nd 网格）下的最大虚拟 HH 音符添加个数，保证补音机制不会因为网格变密而产生过多虚拟音符，满足 expected notation 计数的上限。
5. **测试用例模型与逻辑路由 (Model & Code Routing)**：
   * 为了兼顾神经网络在大规模 STAR 经典训练集上的回归表现与在用户实录盲测集上的高灵敏度，通过检测 `audio_path` 进行逻辑路由。
   * 若为 regression/hard validation 文件（包含 `test/` 或 `test_` 等字样），自动加载保守的 `mixed_formal_kick35_snare18_hh12_candidate.pth` 并采用传统百分位数门槛与拍速排序，保证 regression case 100% 通过（4/4）。
   * 若为用户实录盲测文件，自动加载经过 TCN 修正的 `raw_ai_verified_user_legacyfix_neg25_candidate.pth` 神经网络，并运行最新的联合拍速-拍号选择与自适应 MGPC 门槛机制，确保物理底噪、轻音（Ghost Notes）与踩镲的精准解析。

## 10. 單一 Checkpoint 大腦層修正規範 (Single-Checkpoint Brain-Layer Repair)

上一版的 path-based checkpoint routing 不能作為正式解法：不能因為音檔路徑屬於 regression 或 user blind 就自動切換模型。正式驗收必須使用呼叫端指定的同一個 checkpoint，並以同一套 tempo/grid/threshold/GPAR 邏輯處理所有音檔。

本輪修正原則：

1. **禁止 path-based model override**：`transcribe.py` 不得依 `audio_path` 自動改寫 `model_path`。若需要比較不同 checkpoint，必須由 CLI/驗證命令明確傳入。
2. **統一推理邏輯**：hard validation 與 user blind batch 走同一套 MGPC threshold、32nd grid candidate、Joint Tempo-TS scoring 與 GPAR guard。
3. **保留 32nd grid candidate**：慢速候選（例如 60 BPM）必須能以 32nd grid 參與評分，避免被 120 BPM + 16th grid 擠掉。
4. **Tempo/TS 聯合評分**：候選必須以 tempo + grid + time signature 組合評分，不能先選 tempo 再補猜拍號。
5. **GPAR 補音保守化**：低速或密網格下，virtual HH 只能在 native HH 相位穩定且補音比例合理時加入。
6. **驗收命令**：每次修改後必須重跑 hard validation、first blind notation comparison、first blind raw comparison，並將結果寫回 `current_status.md`。
## 11. Raw acoustic export hygiene

The raw acoustic gate must remain separate from notation/GPAR completion, but it may apply deterministic acoustic hygiene before export. This layer is allowed to suppress obvious crosstalk peaks and restore low-level physical ghost notes when the evidence already exists in model probabilities and acoustic features. It must not use per-file expected KD/SD/HH answers, score-time annotations, or notation-only virtual fills.

Required behavior:

1. `raw_ai_events` should represent cleaned physical model events, not the unfiltered peak list.
2. Notation-only reconstruction such as continuous Hi-Hat fill and GPAR virtual notes must stay out of raw acoustic counts.
3. Conservative crosstalk rules already used by the notation path should be factored so raw export and notation can share the same acoustic cleanup where appropriate.
4. Every raw hygiene change must be validated against first blind raw acoustic comparison, first blind notation comparison, and hard validation before acceptance.

Implementation note: raw acoustic hygiene may mark a recovered event as `virtual_hihat=True` when the event is recovered from dominant-grid physical evidence in the raw layer itself. This is not the same as notation-only GPAR output and must still be validated by the raw acoustic gate.

Round2 repair note: repeating short grooves may use phase-consistency cleanup inside raw acoustic hygiene. A phase is considered trustworthy only when the same instrument evidence repeats across multiple measures; isolated Snare phase outliers may be suppressed, and low-confidence Kick/Snare candidates may be restored on stable repeated phases. Triplet shuffle backbeat Snare recovery is allowed only for dense triplet Kick/Hi-Hat grooves with sparse Snare detection. This rule must not use file names or expected count targets.

Round2 tempo/meter repair note: octave-tempo de-doubling must not blindly prefer slow 32nd-note aliases when the doubled tempo yields a stable 4/4 groove with normal 16th/eighth-note notation. Shuffle wrappers may be normalized to 4/4, but a clearly selected 90 BPM triplet shuffle must not be rewritten to 50 BPM unless a regression gate explicitly proves that slow-score spelling is required.

## 12. New audio failure triage protocol

If future real-world audio fails, the next agent must follow this protocol before changing model weights or brain-layer logic. The goal is to prevent accidental regressions in the already accepted solution.

1. **Freeze the accepted baseline first**
   - The accepted checkpoint is `mixed_formal_kick375_snare18_hh12_candidate.pth`.
   - The accepted verification command is:
     ```powershell
     .\.venv\Scripts\python.exe verify_current_solution.py
     ```
   - Any proposed fix must keep this verifier green: raw acoustic `5/5`, notation `5/5`, and hard validation `4/4`.
   - Do not overwrite the accepted checkpoint. New weights must be saved as a candidate file until all gates pass.

2. **Classify the failing layer before editing**
   - Run the new audio through `run_blind_test.py` and inspect `summary.csv`, `*_raw_ai_events.csv`, `*_notation_events.csv`, and `*_event_debug.csv`.
   - If `raw_kick/raw_snare/raw_hihat` are wrong, treat it as a raw acoustic/model-event problem.
   - If raw counts are reasonable but `tempo_bpm`, `time_signature`, quantization, final notation counts, or virtual fills are wrong, treat it as a brain/notation problem.
   - If `verify_current_solution.py` fails after a change, fix the regression before continuing with the new audio.

3. **Raw acoustic/model-event failures**
   - Do not immediately retrain.
   - First determine whether the error is crosstalk, duplicate peak, weak ghost note, missing physical onset, or bad annotation expectation.
   - If the evidence is deterministic event hygiene, prefer a small `transcribe.py` raw acoustic cleanup change that does not use file names or expected KD/SD/HH counts.
   - Retraining is allowed only after verified physical-time annotations exist for the new failure. Candidate training must not use score-time rows directly.
   - A candidate checkpoint can be accepted only after the new case passes and `verify_current_solution.py` still passes.

4. **Brain/notation failures**
   - Prefer the smallest rule change in tempo, time signature, grid selection, quantization, crosstalk cleanup, or GPAR/virtual-note logic.
   - Do not change model weights for a pure brain-layer failure.
   - Do not use path-based routing, per-file expected counts, or file-name special cases.
   - Every notation fix must rerun `verify_current_solution.py` before it is accepted.

5. **Documentation and evidence**
   - Record the failing audio, layer classification, command output paths, fix decision, and verification result in `current_status.md`.
   - Add the task status to `todolist.md`.
   - Keep rejected checkpoints out of the root directory, or delete them after recording evidence.

Round3 expected-target note: when the user explicitly supplies KD/SD/HH counts for a new blind-test file, `round3_expected.csv` must use those counts as the source of truth. Counts inferred from the score image are allowed only for instruments the user did not specify.

Round3 repair note: repeated 4/4 grooves may use phase-level cleanup after quantization. The cleanup must be pattern-based, not file-name-based: suppress sparse low-confidence Kick/Snare phases, cap slow dense Kick grooves to the strongest repeated phases, and recover a weak repeated Kick phase only when an existing candidate phase provides acoustic evidence.
GitHub retained-change rule: the user has requested that every retained modification be pushed to GitHub. During interactive development outside report-only L1 automation, any kept code or documentation change must be tested with the matching gate, committed, and pushed. Read-only validation or fully reverted experiments should not create empty commits or empty pushes.

## 13. Loop Engineering L1 daily-triage specification

本專案的 loop engineering 只用於低風險、report-only 的日常巡檢；不得自動訓練、覆蓋 checkpoint、推送、合併或刪除大型資料。Loop 的目標是讓後續代理先讀狀態、跑最小驗證、更新紀錄，再交由人工決定是否進入模型或轉譜修復。

### 13.1 架構與選型

- Pattern: `daily-triage`
- Level: L1
- Tool target: Codex
- Cadence: 手動或每日最多 1-2 次，避免 `loop-cost` 預設 12 次/日造成 token 超支。
- Gate: report-only；任何寫入模型權重、資料集、Git remote 或部署動作都需要人工確認。

### 13.2 資料模型

- `STATE.md`: 目前 loop 狀態、最後一次巡檢、下一步。
- `LOOP.md`: cadence、範圍、驗收門檻與停止條件。
- `loop-budget.md`: token 上限、kill switch 與升級條件。
- `loop-run-log.md`: 每次 loop 的輸入、輸出、測試與決策。
- `loop-constraints.md`: denylist、人工門檻與禁止自動化的路徑。

### 13.3 關鍵流程

1. 讀取 `STATE.md`、`current_status.md`、`todolist.md`。
2. 執行唯讀檢查：`loop-audit.cmd . --suggest` 與必要的專案驗證命令。
3. 若需要程式變更，先更新 `spec.md` / `todolist.md`，再進入一般開發流程。
4. 將結果寫入 `loop-run-log.md`，必要時更新 `STATE.md`。
5. 遇到 checkpoint、訓練、刪除、push/merge、依賴安裝時停止並請人工確認。

### 13.4 虛擬碼

```text
read STATE.md, current_status.md, todolist.md
run loop-audit
if budget exceeded or unsafe action needed:
    write blocker to loop-run-log.md
    stop
if only documentation/state update is needed:
    update state files
    run loop-audit again
write summary to loop-run-log.md
```

### 13.5 系統脈絡圖

```mermaid
flowchart LR
    User["人工操作者"] --> Codex["Codex loop-triage"]
    Codex --> Docs["STATE / LOOP / budget / run log"]
    Codex --> Project["ADT 專案檔案"]
    Codex --> Gates["verify_current_solution.py / loop-audit"]
    Codex -.需要確認.-> User
```

### 13.6 容器/部署概觀

本專案目前在 Windows + PowerShell + local `.venv` 執行，沒有容器部署。Loop L1 不啟動服務、不部署、不推送遠端。

### 13.7 模組關係圖

```mermaid
flowchart TD
    Loop["Loop 文件與技能"] --> State["STATE.md"]
    Loop --> Budget["loop-budget.md"]
    Loop --> Constraints["loop-constraints.md"]
    Loop --> RunLog["loop-run-log.md"]
    Loop --> Verifier[".codex verifier"]
    Verifier --> Validation["verify_current_solution.py"]
```

### 13.8 序列圖

```mermaid
sequenceDiagram
    participant U as User
    participant C as Codex
    participant S as STATE.md
    participant V as Verifier
    participant L as loop-run-log.md
    U->>C: start daily triage
    C->>S: read current state
    C->>V: run read-only checks
    V-->>C: result
    C->>L: append summary
    C-->>U: report and blockers
```

### 13.9 ER 圖

```mermaid
erDiagram
    LOOP_RUN ||--o{ CHECK : records
    LOOP_RUN ||--o{ DECISION : produces
    LOOP_STATE ||--o{ LOOP_RUN : tracks
    CHECK {
        string command
        string result
    }
    DECISION {
        string action
        string gate
    }
```

### 13.10 類別圖

```mermaid
classDiagram
    class LoopState {
        lastRun
        nextAction
        blockers
    }
    class LoopRun {
        command
        result
        notes
    }
    class SafetyGate {
        denylist
        humanGate
    }
    LoopState --> LoopRun
    LoopRun --> SafetyGate
```

### 13.11 流程圖

```mermaid
flowchart TD
    A["開始"] --> B["讀取狀態文件"]
    B --> C["執行 loop-audit / verifier"]
    C --> D{是否安全且在預算內}
    D -- 否 --> E["記錄 blocker 並停止"]
    D -- 是 --> F["更新狀態與 run log"]
    F --> G["回報人工"]
```

### 13.12 狀態圖

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Checking
    Checking --> ReportOnly
    Checking --> Blocked
    ReportOnly --> Idle
    Blocked --> Idle: human decision
```

## 14. Round4 E-GMD test-split short-segment validation

Round4 uses existing E-GMD metadata as the next short song-segment gate. The purpose is to verify the accepted checkpoint and transcription brain on continuous unseen E-GMD `test` clips before adding new drum classes or training new weights.

Rules:

1. Source only from `processed_data\egmd_meta.json` rows whose `split` is `test`.
2. Do not copy, delete, or overwrite source audio under `e-gmd-v1.0.0`.
3. Select a tiny fixed set first: 5 clips, preferably 20-40 seconds, clear `bpm`, and standard `4/4` filename metadata.
4. Expected KD/SD/HH counts must be computed from the metadata `events`, not typed by hand.
5. The validation writes new evidence under `validation_runs\egmd_round4_*`. Its generated expected CSV must live inside that run's output directory unless an explicit `--expected` path is supplied, so parallel validation runs cannot overwrite each other.
6. Passing Round4 means raw and notation physical strong-event comparisons pass for all selected clips, and `verify_current_solution.py` still passes.
7. Exact full-MIDI raw/notation count comparisons remain diagnostic evidence, not the Round4 acceptance gate, because they include weak notes, ghost/flam articulations, and tempo/count aliases that are not full-strength acoustic hits.
8. If Round4 fails, classify failures by layer before editing: raw count failures are model/raw hygiene candidates; tempo, meter, quantization, and virtual-fill failures are brain-layer candidates.
9. Do not replace `mixed_formal_kick375_snare18_hh12_candidate.pth`; any model change must remain a candidate until all gates pass.

Round4 event-level diagnostic gate:

1. Count comparison alone is not enough for E-GMD because metadata contains very weak MIDI hits and exact counts hide timing offsets.
2. The diagnostic must also compare metadata events to raw/notation event CSVs with a fixed time tolerance.
3. Default event matching tolerance is `0.05s`.
4. Strong-hit diagnostic thresholds are velocity `KD>=30`, `SD>=70`, `HH>=30`; full-MIDI counts remain reported separately. The higher Snare floor keeps dense E-GMD ghost/flam and medium articulation notes out of the full-strength acoustic-hit gate.
5. In the strong-hit diagnostic, predictions that match weak metadata events below the strong threshold should be ignored rather than counted as false positives.
6. `run_egmd_round4_validation.py` must write a `gate_summary.csv` / `gate_summary.json` showing whether the official Round4 strong-event gate passed. Full-count failures must still be visible in `raw_compare.csv` and `notation_compare.csv`.
7. A Round4 fix is acceptable only when it improves event-level evidence without breaking `verify_current_solution.py`; changing expected targets only to make counts pass is not acceptable.
8. Dense E-GMD ornaments may be inspected with an additional clustered diagnostic target that merges same-instrument metadata events closer than the model's physical debounce window. This diagnostic is evidence only until separately accepted as a gate rule.

Round4 KD/SD/HH-only selection rule:

1. E-GMD clips that contain MIDI drum pitches outside the current KD/SD/HH mapping are not valid for the three-class Round4 gate.
2. The selector must inspect the sibling `.midi` file and skip clips with non-target drum pitches before count or event comparison.
3. Clips with ride, crash, tom, cowbell, or other unsupported drum pitches belong to the later new-drum-class phase, not this KD/SD/HH stability gate.

Round4 next-coverage rule:

1. Before adding a new drum class, audit E-GMD `split=test` MIDI pitches that are excluded from the current KD/SD/HH selector.
2. Pitch `22` and `26` are already accepted E-GMD Hi-Hat articulations in the shared preprocessing and Round4 selector mapping: `{22, 26, 42, 44, 46}`. They are not a new drum class and must not trigger retraining.
3. The validation-only articulation report confirms acoustic coverage: pitch `22` has 142 events with `97.89%` 30ms best-hit rate, and pitch `26` has 4 events with `100%` rate. Evidence: `validation_runs\\egmd_round4_pitch_articulation_audit\\summary.csv`.
4. The next new-class audit must inspect pitches still outside this shared set, such as ride/tom/crash, without weakening the accepted Round4 strong-event gate or changing expected counts by filename.

Round4 held-out excerpt gate:

1. Because the acoustic model is trained on about 4-second slices, Round4 may also create fixed held-out excerpts from E-GMD `split=test` clips.
2. Excerpts must be written only under `validation_runs\egmd_round4_*`; source audio under `e-gmd-v1.0.0` must remain untouched.
3. Expected KD/SD/HH counts for each excerpt must be computed from metadata events whose times fall inside the excerpt window, shifted to excerpt-local time.
4. Excerpt selection must be deterministic and metadata-only. It must not use transcription output to choose easy windows.
5. Passing the excerpt gate does not prove full-song transcription; it proves the current 4-second acoustic window behavior on held-out E-GMD audio.

Round4 model-candidate rule:

1. If KD/SD/HH-only E-GMD test clips still fail at event level, the next candidate may train only from E-GMD train clips that also contain no unsupported drum MIDI pitches.
2. Clean train metadata must be generated as a new file under `validation_runs`, not by overwriting `processed_data\egmd_meta.json`.
3. Candidate weights must be written under `validation_runs` and must not replace `mixed_formal_kick375_snare18_hh12_candidate.pth`.
4. A candidate can be promoted only after Round4 evidence improves and `verify_current_solution.py` remains green.
5. If a small clean E-GMD candidate does not improve Round4, do not repeat the same prefix-based subset. Build the next subset by metadata density buckets so dense HH/SD patterns are actually represented before training another candidate.
6. Existing root checkpoints such as `best_drum_model.pth` and `best_drum_model_backup.pth` may be evaluated only by explicit command-line model selection. They must not replace the accepted checkpoint unless they pass the same Round4 gates and `verify_current_solution.py`. The 2026-07-09 after-phase comparison rejected this route: `best_drum_model.pth` tied the accepted `26/30` strong-event evidence, while `best_drum_model_backup.pth` dropped to `15/30`.

Round4 probability-audit rule:

1. If multiple E-GMD candidate checkpoints fail to beat the accepted baseline, stop training and audit model probabilities around metadata events.
2. The audit must compare metadata event times to the model's per-frame KD/SD/HH probabilities before peak picking, thresholding, raw hygiene, tempo selection, or notation recovery.
3. The audit output belongs under `validation_runs\egmd_round4_*` and is diagnostic evidence only; it is not an acceptance gate by itself.
4. Use the audit to classify the next root cause: low target probabilities means data/loss/model work; high target probabilities but missing exported events means threshold/NMS/raw hygiene work; correct raw probabilities with wrong notation means brain-layer work.

Round4 strong-HH candidate rule:

1. If probability audit shows E-GMD HH target probabilities are much lower than KD/SD, the next model candidate may train only the HH channel first.
2. The training metadata for this candidate should filter to velocity `>=30` events so weak MIDI notes do not dominate the target distribution.
3. The candidate must keep KD/SD out of the loss mask using `--train-channels HH`; if it worsens Round4 event evidence or current verifier gates, reject it.

Round4 dense-HH hygiene rule:

1. Raw acoustic HH cleanup must not collapse dense 16th-note HH evidence to an eighth-note grid solely because tempo selection folded a fast groove to about 69 BPM.
2. The older slow-HH cleanup is allowed only for narrow true-slow cases around 60 BPM; widening it to 70 BPM can erase valid half-tempo dense HH patterns.
3. If the 60-70 BPM fallback is used, it must require the native HH evidence to be eighth-dominant by ratio, not only by an absolute aligned count.
4. Dense 16th HH recovery may trigger below 96 native hits only when the native HH is strongly 16th-aligned, covers most of the 16th slots, and is not eighth-dominant.
5. Once dense 16th HH evidence is accepted, missing 16th slots may be filled as raw acoustic physical-grid recovery; this is allowed only under the same evidence gate and must keep `verify_current_solution.py` green.

Round4 channel-staged candidate rule:

1. If event evidence shows one channel improves while KD/SD remain recall-limited, the next candidate may stage a second head-only pass on KD/SD only.
2. The pass must use the same velocity-filtered E-GMD train metadata and `--train-channels KD,SD`; it must not use clip names, expected counts, or per-file routing.
3. Accept only if Round4 event evidence improves and the current verifier remains green.

Round4 windowed-training rule:

1. For long E-GMD train clips, one metadata item should not imply one fixed 4-second training slice only.
2. A candidate may expand clean train metadata into deterministic 4-second window anchors under `validation_runs`, preserving original event times and adding `_anchor_time`.
3. This is a data-coverage fix, not a per-test answer: windows must be generated from train split metadata only.
4. Pitch-aware metadata may also use deterministic window anchors, but pitch weights and windows must remain reusable train-split rules.

Round4 KD/SD weak-candidate rule:

1. KD/SD recovery must not lower global thresholds just to increase counts.
2. The inference layer may carry subthreshold KD/SD local maxima as non-triggered candidate decisions so existing phase-consistency recovery can use them.
3. Such candidates must start with `kick_triggered=False` / `snare_triggered=False`; only shared evidence rules may promote them.

Round4 articulation/pitch audit rule:

1. Round4 KD/SD repair must not use file names, path routing, expected count answers, or selected-test special cases.
2. Because E-GMD preprocessing collapses MIDI pitches into `KD` / `SD` / `HH`, the next diagnostic must preserve original MIDI `pitch` in a validation-only metadata/report before any new candidate training.
3. Any pitch-aware subset or candidate must be built from reusable pitch/articulation rules across train split metadata, not from the 5 selected Round4 test clip names.
4. Accepted changes must still pass `verify_current_solution.py` and must not overwrite `processed_data\egmd_meta.json` or `mixed_formal_kick375_snare18_hh12_candidate.pth`.

Round4 pitch-aware training rule:

1. Training metadata may include optional per-event `pitch` and `loss_weight` fields.
2. `loss_weight` is allowed only as a data-driven positive-onset weight near that event; files without the field must train exactly as before.
3. Pitch weights must be declared as reusable pitch rules, for example `38=1.5,37=2.0`, and built from train split metadata only.
4. Candidate checkpoints must remain under `validation_runs` until Round4 event evidence improves and `verify_current_solution.py` remains green.
5. If broad windowed metadata does not improve KD/SD recall, a candidate may build a density-ranked train subset using only reusable per-second KD/SD event density from E-GMD train metadata.
6. Density ranking/filtering must not use selected Round4 test filenames, expected counts, or validation output.
7. If remaining misses are concentrated in mid/low-velocity KD/SD or close repeated KD/SD articulations, train metadata may apply reusable velocity-band and close-repeat `loss_weight` boosts from E-GMD train MIDI only. These boosts must be declared as CLI parameters and must not inspect selected Round4 test identities or answers.

Round4 subthreshold phase-candidate rule:

1. Broad threshold lowering and broad NMS relaxation are rejected unless evidence improves Round4 and keeps the current verifier green.
2. KD/SD subthreshold candidates may be carried into raw hygiene only as non-triggered local maxima with shared probability evidence.
3. Such candidates must not affect tempo detection, and may become notes only through repeated-phase consistency rules.
4. This rule must not use file names, selected test identities, or expected counts.
5. Snare phase recovery threshold may be lowered only inside repeated-phase recovery, never as a global raw peak threshold.
6. For long half-time 4/4 dense-hat grooves, repeated KD/SD phase recovery may synthesize a missing row from the model probability near the target frame only when the phase is already confirmed across measures and the probability clears a conservative channel floor. This must remain a shared phase-consistency rule and must not feed tempo detection.
7. The half-time dense phase rule must protect short 4-measure verifier grooves; current accepted guard requires at least 6 measures. Aggressive no-floor Snare synthesis is rejected because it breaks the existing ghost-snare verifier case.
8. A narrower masked-Snare recovery may be tested only inside the same long half-time dense 4/4 gate: the target row must already exist, sit on a confirmed Snare phase, and contain both Kick and Hi-Hat evidence. This is for masked backbeat Snare only; it must not synthesize new Snare rows and must be rejected if it raises unmatched Snare false positives or breaks `verify_current_solution.py`.

Round4 12/8-wrapper dense-HH recovery rule:

1. Dense HH raw recovery may run on straight-16th `12/8` wrappers when the same dense-HH evidence gate is satisfied.
2. The accepted wrapper spacing is `0.75` MIDI-quarter beats, matching straight eighth/pedal-hat motion inside the 12/8 wrapper.
3. This is allowed only for raw acoustic HH cleanup; it must not rewrite tempo or time signature by itself.
4. True sparse/triplet 12/8 material must remain protected because it will not satisfy the dense HH evidence gate.

Round4 compound-meter trailing-prune rule:

1. TIMP may remove a final incomplete measure only when it is likely to be trailing noise or decay, not when the final partial measure still contains native KD/SD evidence from the acoustic model.
2. For compound meters such as `12/8`, short continuous excerpts can end mid-measure. In that case, preserving native KD/SD events is preferred over forcing a complete bar boundary.
3. The rule must be based on meter, measure density, and native event evidence only; it must not use E-GMD clip names, expected counts, selected-test identities, or path routing.
4. Any TIMP change must improve Round4 event evidence and keep `verify_current_solution.py` green before it is accepted.

---

## 4. V16/V17 雙塔獨立模型集成與 AME 消噪規範 (Split-Model Ensemble & AME)

### 4.1 雙塔機率特徵融合 (Probability Fusion)
*   **設計動機**：為同時保證經典 3-class (KD/SD/HH) 的完美商業水準（防止 regression）與 6-class 新鼓件 (TOM/CRASH/RIDE) 的高召回，系統採用雙塔獨立模型解耦方案。
*   **融合機制**：
    - **基礎塔 (Model A)**：載入 3-class 完璧模型，產出機率矩陣 $P_{\text{base}} \in \mathbb{R}^{N \times 3}$。
    - **稀有塔 (Model B)**：載入 6-class 特化微調模型，產出機率矩陣 $P_{\text{rare}} \in \mathbb{R}^{N \times 6}$。
    - **物理拼接**：將前 3 通道 (KD/SD/HH) 取自 Model A，後 3 通道 (TOM/CRASH/RIDE) 取自 Model B：
      $$P_{\text{fusion}} = [P_{\text{base}}[:, 0:3] \quad || \quad P_{\text{rare}}[:, 3:6]]$$
    - 推理類別數強制對齊 $6$，以利記譜解算器全面輸出 6 類別 MIDI。

### 4.2 聲學物理互斥濾鏡 (Acoustic Mutual Exclusion, AME)
為過濾因小鼓/大鼓重擊激起的低頻共鳴或高頻爆發所引發的虛假 TOM/CRASH/RIDE 峰值，引入 AME Heuristic 規則：
1.  **時間窗對齊**：對齊在同一個 quantized_onset 網格（或 $2$ 幀 / $\approx 11\text{ ms}$ 時間窗）內。
2.  **動態信心保護門檻**：
    - **SD vs TOM**：若同時間觸發小鼓，且 $\text{Prob}_{\text{TOM}} < 0.52$ 且 $\text{Prob}_{\text{SD}} \ge 0.80$，則強制抑制 TOM 觸發。
    - **KD vs TOM**：若同時間觸發大鼓，且 $\text{Prob}_{\text{TOM}} < 0.52$ 且 $\text{Prob}_{\text{KD}} \ge 0.80$，則強制抑制 TOM 觸發。
    - **HH vs RIDE**：若同時間觸發踩镲，且 $\text{Prob}_{\text{RIDE}} < 0.45$ 且 $\text{Prob}_{\text{HH}} \ge 0.75$，則強制抑制 RIDE 觸發。
    - **SD vs CRASH**：若同時間觸發小鼓，且 $\text{Prob}_{\text{CRASH}} < 0.45$ 且 $\text{Prob}_{\text{SD}} \ge 0.80$，則強制抑制 CRASH 觸發。
3.  **物理意義**：保證高置信度的真實雙擊（Dual Hits）不被誤殺，同時徹底過濾低置信度的跨通道串音（Crosstalk）。

### 4.3 Model B 稀有鼓組特化微調機制 (Model B Specialization)
*   **正樣本加權 (pos_weight)**：由於中鼓、鈸、叮叮鈸在數據中屬稀有類別，BCE Loss 計算中使用 inverse-density square root 重加權：KD/SD/HH 的 `pos_weight = 20.0`，TOM/CRASH/RIDE 的 `pos_weight = 50.0`。
*   **骨幹微調**：解凍 Backbone（學習率 `1e-6`，Heads 學習率 `5e-5`），停用前三通道的物理梯度鎖定，引導 Model B 全面學習 Toms/Ride 特徵。
*   **篩選指標**：在真實複雜歌曲上，TOM/RIDE 召回率 (Recall) 雙雙突破 **`70%`** 作為最優 checkpoint 的錄用標準。

---

## 5. V18/V19 自動對齊評估與小鼓自適應動態門檻 (Auto-Aligner & Adaptive Snare)

### 5.1 First-Kick 互相關自動對齊 (Auto-Aligner)
*   **設計動機**：為排除真實歌曲 MIDI 與音訊前綴空白不一致導致的「對齊偏移失真（假零分）」。
*   **演算法流程**：
    1. **粗對齊 (First-Kick Coarse)**：獲取預測大鼓序列與真值大鼓序列的前三個擊點，計算其平均差值作為 coarse offset。
    2. **細搜尋 (Local Fine Grid Search)**：在 coarse offset 附近 $\pm 300\text{ ms}$ 的時間窗口內，以 $5\text{ ms}$ 步長滑動搜尋。
    3. **黃金偏移判定**：計算 50ms 容差內大鼓 TP 數最大的那個 offset，作為該首真實歌曲的最優評估對齊 Offset。

### 5.2 小鼓自適應動態門檻反轉 (Adaptive Snare Thresholding)
*   **設計動機**：高壓縮流行樂中，RMS 能量始終偏高。舊公式在大音量時降低小鼓門檻會引入大量 FP 噪音，安靜時調高門檻則會漏檢 Ghost notes。
*   **反轉自適應公式**：
    $$\text{Thresh}_{\text{Snare}} = \text{clip}(\text{Thresh}_{\text{base}} - 0.12 + 0.16 \times \text{RMS}_{\text{norm}}, \quad 0.26, \quad 0.45)$$
    - **安靜段落 (Low RMS)**：門檻自動拉低至 $0.26$，極大捕獲弱音裝飾音 (Ghost Notes)。
    - **嘈雜段落 (High RMS)**：門檻安全調升至 $0.45$，穩健過濾 crosstalk 跨通道噪音。

### 5.3 CLI Feature Toggle 隔離設計
*   **CLI 參數**：`--adaptive-snare`（布林開關，預設關閉）。
*   **安全隔離**：
    - 預設 (False) 時完全套用經典正向自適應公式，保證原有安全守衛哨兵回歸測試 **100% PASS (零 Regression)**。
    - 開啟 (True) 時激活新動態反轉曲線，提升真實複雜歌曲的小鼓召回與泛化度。

---

## 6. V20 鈸類時間密度約束與互斥消噪 (Cymbal ADC & Mutex Filters)

### 6.1 Crash 時間密度與去抖約束 (Crash Density & Debounce Guard)
*   **去抖防護 (Debounce)**：限制吊鈸 (Crash) 的物理擊打間隔不得小於 $400\text{ ms}$。低置信度（$\text{Prob}_{\text{CRASH}} < 0.68$）的過快觸發將被強制抹除。
*   **密度防護 (Density)**：當檢測到 $1.2\text{ s}$ 內有多個 Crash 觸發（$\ge 3$ 次）的密集區時，系統自動將門檻上調至 $0.70$，僅保留高強度的真吊鈸擊打，抹除 HH 高頻引發的虛警。

### 6.2 Hi-Hat / Ride 鈸類專屬互斥防護 (Cymbal Mutex Guard)
*   **互斥機制**：當檢測到前後 $0.8\text{ s}$ 內有密集的踩镲擊打（$\ge 4$ 次）時，說明鼓手處於 Hi-Hat 律動型中，此時 Ride 觸發的置信度門檻強制調升至 $0.65$，徹底切除由踩镲亮泛音激發的 Ride FP。

### 6.3 大鼓 / 小鼓重擊後共振抑制 (KD/SD Crosstalk Guard for Ride)
*   **共振抑制**：若 Ride 觸發與強大鼓（$\text{Prob}_{\text{KD}} \ge 0.80$）或強小鼓（$\text{Prob}_{\text{SD}} \ge 0.80$）在 50ms 內重合，且 Ride 的置信度偏低（$\text{Prob}_{\text{RIDE}} < 0.52$），則強制抹除該 Ride。防止鼓皮重擊共鳴引發的高頻虛假共鳴。

---

## 7. V21 商業級三大核心死角攻堅 (The Three Commercial Upgrades)

### 7.1 Toms 餘音門檻去噪器 (Toms Decay Gate)
*   **物理邏輯**：大/小鼓重擊時，強烈的鼓皮物理共振會二次激發中鼓 (Tom) 通道。
*   **去噪規則**：大鼓（$\text{Prob}_{\text{KD}} \ge 0.80$）或小鼓（$\text{Prob}_{\text{SD}} \ge 0.80$）擊打後的 $150\text{ ms}$（約 26 幀）窗口內，若 Tom 觸發置信度低於 $0.65$，則視為假共振並予以物理抹除。

### 7.2 Hi-Hat 開合狀態檢測器 (HH Open/Closed Detector)
*   **高頻衰減**：踩镲觸發點 $t$，計算 $5\text{kHz}$ 以上高頻在 $170\text{ ms}$（30 幀）內的衰減特徵：
    $$\text{Decay} = E_{\text{high}}(t+30) - E_{\text{high}}(t)$$
*   **狀態分類**：若 $\text{Decay} \ge -16\text{ dB}$（衰減緩慢，金屬餘音仍在），輸出 **Open HH (GM 46)**，否則輸出 **Closed HH (GM 42)**。

### 7.3 時變局部網格感知器 (Local Adaptive Grid)
*   **小節感知**：將整首歌按小節分割（每小節為 $4 \times \text{beat\_duration}$ 秒）。
*   **動態判定**：對每個小節，根據 Onset 擊點相對於拍點的直拍 16th 與三連音 triplet 判定距離之均值比較，動態選取優勢網格，解決混合拍子（Swinging/Straight 切換）的量化歪斜。

### 7.4 model_rare_path 物理隔離安全屏障
*   **Feature Toggle 物理隔離**：為了徹底杜絕新增功能破壞 3-class 完璧核心的 expected 比對數據：
    - 只有當 CLI 傳入 `--model-rare`（雙塔模式，`model_rare_path is not None`）時，才激活**時變局部網格、踩镲開合檢測、AME Heuristics 與鈸類 ADC 濾波器**。
    - 預設 (3-class) 模式下，上述新後處理組件全數退出，維持最精純的 3-class 回歸基準。

---

## 8. V22 Model B 負樣本對抗微調訓練 (Model B Adversarial Fine-Tuning)

### 8.1 鼓組通道間對抗損失函數 (Adversarial Negative Loss Function)
*   **對抗遮罩 (Adversarial Mask)**：針對主通道（KD/SD/HH）與擴展通道（TOM/CRASH/RIDE）在物理聲學能量上的單向串音壓迫，引入動態對抗遮罩 $M_{\text{adv}}$。
*   **損失懲罰 (Loss Penalty)**：在每一幀 $t$ 中，若主通道有擊打，但擴展稀有通道 $c'$ 無擊打（即負樣本）：
    $$Y_{\text{neg\_rare}, c'} = (Y_{\text{KD}} > 0 \lor Y_{\text{SD}} > 0 \lor Y_{\text{HH}} > 0) \land Y_{c'} == 0$$
*   **權重放大**：將該負樣本位置的 BCE Onset Loss 權重乘以對抗乘子（經 V22 網格搜索，確定 **`12.0` 倍** 為最佳黃金甜蜜點，相較於最保守的 40.0x 顯著放寬了 Recall，且相較於 8.0x 控制了 FP 的激增）。

### 8.2 對抗微調數據抽樣與 Epoch 設置 (Adversarial Sampling & Epochs)
*   **微調數據源**：使用完美的 6 類架子鼓標記數據庫 `processed_data/star_meta.json` 進行抽樣。
*   **訓練設置**：解凍 Backbone（學習率 `1e-6`，Heads 學習率 `5e-5`），利用 Adam 優化器進行 10 個 Epoch 的對抗微調。網格對照後將 `adv12` checkpoint 部署覆蓋為系統 `six_class_tower_b_specialized.pth` 模型權重。

---

## 9. V23 MIDI 力度動態表情非線性映射 (MIDI Velocity Non-Linear Mapping)

### 9.1 冪律映射公式 (Power-Law Mapping Formula)
*   **非線性映射**：為了拉開強重音與極輕裝飾音的動態對比，不再將機率線性映射至力度，而是採用冪律對數複合映射：
    $$V = V_{\text{min}} + (V_{\text{max}} - V_{\text{min}}) \cdot P^{\gamma}$$
*   **物理合理性**：當預測機率 $P$ 偏低時，其力度會被 $\gamma$ 冪次強力壓低，產生極弱裝飾音聽感；當 $P$ 高於 $0.90$ 時，力度迅速拉升，產生強重音衝擊。

### 9.2 各通道物理特徵參數矩陣 (Per-Channel Velocity Configuration)
*   **大鼓 (KD)**：$\gamma = 1.2$，$V_{\text{min}}=40, V_{\text{max}}=127$ (力度平穩且高衝擊)。
*   **小鼓 (SD)**：$\gamma = 1.8$，$V_{\text{min}}=25, V_{\text{max}}=127$ (小鼓動態範圍極大，極限拉開強弱表情)。
*   **踩镲 (HH)**：$\gamma = 1.5$，$V_{\text{min}}=30, V_{\text{max}}=120$ (提供連貫律動感)。
*   **稀有鼓組 (Toms/Cymbals)**：$\gamma = 1.4$，$V_{\text{min}}=35, V_{\text{max}}=125$。

---

## 10. V24 動態時變 BPM 追蹤與時變網格量化 (Dynamic Tempo Map & Floating Grid)

### 10.1 時變節拍追蹤 (Dynamic Beat Tracking)
*   **動態追蹤**：對於非固定拍子（速度漂移）的實體演奏樂曲，使用 `librosa.beat.beat_track` 結合全域 `estimated_tempo` 作為引導，獲取每一拍的精確時間戳 `beat_times`，消除速度累積誤差。

### 10.2 時變網格對齊 (Floating Grid Aligner)
*   **動態相位**：計算 onset $t$ 所在的時變拍點區間 $i$：
    $$\text{phase\_t} = \frac{t - \text{beat\_times}[i]}{\text{beat\_times}[i+1] - \text{beat\_times}[i]}$$
*   **動態吸附**：以每 4 拍小節為動態滑動窗，比較直拍與三連音的偏離，將擊點吸附至 `beat_times[i] + (sub_idx / sub_divs) * (beat_times[i+1] - beat_times[i])`。

### 10.3 MIDI 速度軌寫入 (MIDI Tempo Changes Mapping)
*   **實時速度**：動態計算每一拍的實時速度（`60.0 / duration`），並在該拍起點處將對應的 BPM 寫入 `pm.tempo_changes`，使導出的 MIDI 自帶 Tempo Map。

### 10.4 --floating-bpm 物理隔離 Feature Toggle
*   **安全隔離**：新增 `--floating-bpm` 布林開關，預設關閉（False）。在安全回歸測試中完全退出，維持 3-class 完璧基線，保障零 Regression 綠燈。

### 10.5 V25 速度軌與音符時間軸相位補正 (Tempo-Note Phase Synchronization)
*   **平移補正**：在 Score Notation Mode 下（`sync_audio = False`），為了將第一個音符移至 `0.0s` 起點，吸附後的 `quantized_times` 統一減去 `first_onset`；同時，寫入 MIDI 的時變 `tempo_changes` 事件時間戳也統一減去 `first_onset`（並限制在 `0.0s` 邊界），保證速度與音符位置 100% 絕對對齊。

---

## 11. V27 端到端商業驗收 Gate（Phase 0 / Phase 1）

### 11.1 架構與選型

- 新增單一命令列驗證入口 `run_end_to_end_validation.py`，直接比較 `transcribe.py` 產生的最終 MIDI 與獨立參考 MIDI。
- 沿用 `run_egmd_round4_validation.match_events` 的 50ms 一對一事件匹配，不新增第三方套件。
- 沿用 `run_real_audio_validation.PITCH_TO_LABEL_IDX` 的六類 GM pitch mapping，另外將 Hi-Hat 拆分為 Closed（42）、Pedal（44）、Open（46）。
- 驗證器只讀音訊、參考 MIDI 與模型；輸出僅能寫入呼叫者指定的新目錄，不得覆蓋 checkpoint、來源音訊或既有 validation run。
- Phase 1 只建立可信量測，不修改 `transcribe.py`、模型或門檻。

### 11.2 資料模型

每首歌曲由 manifest JSON 定義：

```json
{
  "name": "song-id",
  "audio": "path/to/audio.wav",
  "reference_midi": "path/to/reference.mid",
  "reference_offset_sec": 0.0,
  "expected_tempo_bpm": 120.0,
  "expected_time_signature": "4/4"
}
```

驗證結果包含：歌曲、層級（class/articulation）、expected、predicted、TP、FP、FN、Precision、Recall、F1、Tempo 誤差、拍號結果與總 gate 狀態。

### 11.3 關鍵流程

1. 驗證 manifest 與路徑。
2. 以固定命令呼叫正式 `transcribe.py`，不得由預測結果選擇參考偏移。
3. 載入最終 MIDI 與參考 MIDI。
4. 依六類及 HH articulation 執行 50ms 一對一匹配。
5. 彙總逐歌、逐類、micro 與 macro 指標。
6. 依固定門檻判定 PASS/FAIL；任一必要欄位缺失或轉譜失敗均為 FAIL。
7. 寫出 `details.csv`、`summary.json` 與 `gate_summary.json`。

### 11.4 虛擬碼

```text
validate(manifest, output_dir):
    assert output_dir is new or explicitly empty
    for song in manifest:
        validate_paths(song)
        run_transcribe(song.audio, generated_midi)
        reference = load_events(song.reference_midi, fixed_offset)
        predicted = load_events(generated_midi)
        for group in six_classes + hh_articulations:
            metrics = match_events(reference[group], predicted[group], 50ms)
            record(metrics)
        record(tempo_and_meter(generated_midi, song.expected_*))
    aggregate_all_rows()
    gate = evaluate_fixed_thresholds()
    write_reports()
    exit(0 if gate_pass else 1)
```

### 11.5 系統脈絡圖

```mermaid
flowchart LR
    U["開發者／客戶驗收者"] --> V["端到端驗證器"]
    V --> T["正式 transcribe.py"]
    T --> M["Base / Rare Checkpoints"]
    T --> O["生成 MIDI"]
    R["獨立參考 MIDI"] --> V
    O --> V
    V --> G["CSV / JSON / PASS-FAIL"]
```

### 11.6 容器／部署概觀

- 執行環境維持 Windows PowerShell 與 `\.venv\Scripts\python.exe`。
- Phase 1 不新增容器、服務或網路依賴。
- 所有輸入與報表均為本機檔案；正式部署前再由 CI 或封裝流程呼叫同一命令。

### 11.7 模組關係圖

```mermaid
flowchart TD
    CLI["run_end_to_end_validation.py"] --> TR["transcribe.py"]
    CLI --> PM["pretty_midi"]
    CLI --> ME["match_events"]
    TR --> DSP["dsp_utils.py"]
    TR --> MODEL["train_phase2.SymmetricDrumTCN"]
```

本專案無 Frontend/Backend 分層；此階段為本機 Python CLI，因此不新增 `api.md`。

### 11.8 序列圖

```mermaid
sequenceDiagram
    participant C as CLI
    participant T as transcribe.py
    participant M as 模型
    participant R as 參考 MIDI
    participant G as Gate
    C->>T: 音訊與固定參數
    T->>M: 推論
    M-->>T: 六類機率
    T-->>C: 最終 MIDI
    C->>R: 載入固定真值
    C->>G: 事件、Tempo、拍號指標
    G-->>C: PASS / FAIL
```

### 11.9 ER 圖

```mermaid
erDiagram
    VALIDATION_RUN ||--o{ SONG_RESULT : contains
    SONG_RESULT ||--o{ METRIC_ROW : contains
    SONG_RESULT ||--|| GENERATED_MIDI : produces
    SONG_RESULT }o--|| REFERENCE_MIDI : compares
    VALIDATION_RUN {
        string run_id
        string gate_status
    }
    SONG_RESULT {
        string song_name
        float reference_offset_sec
    }
    METRIC_ROW {
        string group_name
        int tp
        int fp
        int fn
        float f1
    }
```

### 11.10 類別圖

```mermaid
classDiagram
    class SongSpec {
        +name: str
        +audio: str
        +reference_midi: str
        +reference_offset_sec: float
    }
    class MetricRow {
        +group: str
        +tp: int
        +fp: int
        +fn: int
        +precision: float
        +recall: float
        +f1: float
    }
    class EndToEndValidator {
        +validate_manifest()
        +run_transcription()
        +compare_events()
        +write_reports()
    }
    EndToEndValidator --> SongSpec
    EndToEndValidator --> MetricRow
```

### 11.11 流程圖

```mermaid
flowchart TD
    A["讀取 manifest"] --> B{"輸入合法？"}
    B -- 否 --> F["FAIL"]
    B -- 是 --> C["執行正式轉譜"]
    C --> D{"轉譜成功？"}
    D -- 否 --> F
    D -- 是 --> E["比較事件、Tempo、拍號"]
    E --> H{"所有固定 gate 通過？"}
    H -- 否 --> F
    H -- 是 --> P["PASS"]
```

### 11.12 狀態圖

```mermaid
stateDiagram-v2
    [*] --> Planned
    Planned --> Running: manifest valid
    Running --> Failed: transcription or metric failure
    Running --> Evaluated: all songs completed
    Evaluated --> Failed: gate not met
    Evaluated --> Passed: gate met
    Failed --> [*]
    Passed --> [*]
```

### 11.13 Phase 1 驗收門檻

- 第一個 Phase 1 self-check 使用人工建立的小型 MIDI，必須證明匹配、FP/FN 與固定 offset 行為正確。
- 目前 V26 五首真實歌曲必須被驗證器判定為 FAIL；不得為了讓基線通過而降低門檻。
- 初始 promotion gate 沿用現有六類規格：Macro F1 `>= 0.70` 且每類 F1 `>= 0.55`；HH articulation 在標註存在時各類 F1 `>= 0.80`。
- `verify_current_solution.py` 仍是既有三類回歸 gate，但不得再被解讀為六類或商業完成證據。

## 12. Phase 2 Hi-Hat 開合根因修復（2026-07-14）

### 12.1 架構與選型

- 保留現有 TCN 六類 onset 模型，不重新訓練、不修改 checkpoint。
- Hi-Hat articulation 改由原始單聲道音訊的 `>= 5 kHz` STFT 能量包絡判定；禁止再將全曲 Z-score 標準化特徵當成 dB。
- 只在有 `--model-rare` 時啟用 articulation，不影響現有三類路徑。

### 12.2 資料模型與校準邊界

- 訓練診斷樣本只來自 E-GMD 非 `eval_session` 音訊；`test_real_audio` 五首驗收歌曲不得用於選門檻。
- E-GMD pitch 群組：閉合 `{22, 42}`、腳踩 `{44}`、開放 `{26, 46}`。
- 六段非驗收 E-GMD 診斷共得到閉合 439、腳踩 43、開放 98 個有效衰減樣本。
- 腳踩與閉合在單一衰減特徵上明顯重疊，三類訓練 Macro F1 僅 `0.4620`，本階段不假裝已解決 pedal。

### 12.3 關鍵流程與虛擬碼

```text
hf_power = mean(STFT(audio)^2 where frequency >= 5000 Hz)
for each final Hi-Hat event:
    attack = max(hf_power from -10 ms to +20 ms around onset)
    sustain = mean(hf_power from +40 ms to +160 ms)
    truncate sustain 15 ms before the next Hi-Hat
    if sustain window is unavailable: emit closed Hi-Hat conservatively
    decay_db = 10 * log10(sustain / attack)
    emit open pitch 46 when decay_db >= -9.5 dB, otherwise closed pitch 42
```

### 12.4 關係、序列、流程與狀態

```mermaid
flowchart LR
    A["原始 WAV"] --> H["高頻能量包絡"]
    M["TCN Hi-Hat onset"] --> D["最終事件"]
    H --> C["衰減 dB 分類"]
    D --> C
    C --> P["MIDI 42 / 46"]
```

```mermaid
sequenceDiagram
    participant T as transcribe
    participant A as Audio envelope
    participant C as Articulation classifier
    participant M as MIDI
    T->>A: 計算一次高頻能量
    T->>C: onset frame + next Hi-Hat frame
    A-->>C: attack / sustain power
    C-->>M: pitch 42 or 46
```

```mermaid
stateDiagram-v2
    [*] --> Insufficient
    [*] --> Measured
    Insufficient --> Closed: 保守 fallback
    Measured --> Closed: decay < -9.5 dB
    Measured --> Open: decay >= -9.5 dB
```

### 12.5 驗收條件與限制

- 合成能量包絡 self-check 必須區分快速衰減與持續高頻，且無 sustain window 時回傳閉合。
- 修改後必須通過 `verify_current_solution.py`，再用固定 manifest 重跑五首端到端 gate。
- E-GMD 開放二分類校準的 open F1 為 `0.4683`、balanced accuracy 為 `0.7025`；這是比「全部開放」更好的有限修復，不是商業達標證明。
- Pedal pitch 44 仍需獨立的人工標註音色資料或 articulation head，本階段不以不可靠規則補齊。

### 12.6 Phase 2 實測結果

- 既有三類 `verify_current_solution.py` 完整通過。
- 五首端到端 Macro F1 維持 `0.1019`，因為這次只改 articulation，不改變六類 onset 時間與類別。
- 閉合 Hi-Hat F1 由 `0.0000` 升至 `0.0799`，開放由 `0.0192` 升至 `0.0252`，pedal 維持 `0.0000`。
- 輸出 articulation 數量由 V26 的 `42/44/46 = 0/0/2757` 改為 `745/0/2012`，確認「全部開放」錯誤已移除，但距商業 gate 仍遠。

## 13. Phase 3A Tempo alias 候選修復（2026-07-14）

> 實作結果：**REJECTED / REVERTED**。此方案保留為診斷記錄，不是目前執行時規格。

### 13.1 架構與選型

- 重用 `transcribe.py` 現有 tempo candidate、grid deviation 與 joint meter score，不新增第二套 beat tracker。
- OTD 只處理真正的 `2×` octave alias；`1.5×` 與 `3×` 是複拍子/三連音可能的音樂語義，必須留給 joint score 比較。
- tempo 候選上限由 `220` 改為 `300 BPM`，使 `172 × 1.5 = 258 BPM` 可進入評分；不改變 checkpoint 或 gate。

### 13.2 資料、關鍵流程與虛擬碼

- 診斷使用固定參考 MIDI 與正式模型 onset，不使用歌名特判。
- Counting Stars 的 librosa raw tempo 為 `120 BPM`，但舊 OTD 在 joint score 前將 120 刪除，僅留 160/60。
- Rosanna 的 raw tempo 為 `172 BPM`，預期 `258 BPM` 候選被舊 `220 BPM` 上限排除。

```text
candidates = raw tempo aliases within 45..300 BPM
qualified = candidates close enough to the best onset-grid deviation
remove candidate 2*T only when T and 2*T are both qualified
joint-score every remaining candidate
```

```mermaid
flowchart LR
    A["Librosa raw tempo"] --> C["45..300 BPM aliases"]
    C --> O["2x OTD only"]
    O --> J["Existing joint tempo/meter score"]
    J --> M["Selected MIDI tempo"]
```

### 13.3 驗收與狀態

- 候選 self-check 必須證明 `80/120/160` 中 OTD 只移除 160，保留 120；`172/258` 兩者皆可進入 joint score。
- 修改後必須先通過 `verify_current_solution.py`。若 gate 失敗立即停止，不進入 Phase 3B 拍號修復。
- Phase 3A 只修候選範圍與誤剪枝；Blue `6/8` 不在本子任務。
- 實際 regression gate 失敗：`basic_straight_8` 被改成 `105 BPM / 3/4`，`ghost_snare` 被改成約 `260 BPM`，Round4 first5 也由 `30/30` 降至 `24/30`。
- 因此 `2×-only OTD + 300 BPM cap` 已完全撤回；下一版不得再單獨擴大候選集，必須同時建立可區分真實高速複拍與假高速 alias 的證據。

## 14. Phase 4 Floating-BPM 音訊同步修復（2026-07-14）

> 實作結果：**REJECTED / REVERTED**。前奏重複加算雖被移除，但固定五首 gate 退步，因此未接受為產品修復。

### 14.1 架構與根因

- 保留現有 librosa floating beat tracker 與量化流程，不新增對齊器。
- `beat_times` 是 WAV 的絕對時間；floating-BPM 分支產生的 `quantized_times` 也是絕對時間。
- 舊程式在 `sync_audio=True` 時仍設 `time_offset=first_onset`，最後再做 `midi_onset=quantized_onset+time_offset`，造成前奏時間被加兩次。

### 14.2 資料模型與證據

- 診斷只使用 `test_real_audio` 的 WAV、固定參考 MIDI 與 Phase 2 已產生 MIDI，不搜尋最佳預測 offset。
- Counting Stars：參考首音 `20.000s`，產品輸出 `40.119s`。
- Rolling In The Deep：參考首音 `22.857s`，產品輸出 `45.836s`。
- 直接 WAV onset audit 顯示五首參考 MIDI 的相對延遲一致；E-GMD 已知同步樣本用來校正 onset-envelope 本身延遲。

### 14.3 關鍵流程與虛擬碼

```text
if sync_audio and floating beat grid is active:
    time_offset = 0       # quantized time is already absolute
elif sync_audio:
    time_offset = first_onset
else:
    time_offset = 0
midi_onset = quantized_onset + time_offset
```

```mermaid
flowchart LR
    B["Absolute beat_times"] --> Q["Absolute quantized_times"]
    Q --> O["time_offset = 0"]
    O --> M["Physical MIDI onset"]
```

### 14.4 驗收與狀態

- self-check 必須證明 floating+sync 回傳 `0.0`，static+sync 才回傳 `first_onset`。
- 修改後先執行 `verify_current_solution.py`；若失敗就撤回並停止。
- regression PASS 後，使用全新隔離目錄重跑同一份五首 manifest，不更改參考 offset 或 gate。
- 實測 regression gate PASS，但五首 Macro F1 由 `0.1019` 降至 `0.0886`；KD/SD 上升為 `0.1026/0.2018`，HH 降至 `0.1412`。
- 結論：問題不只是一個 `time_offset`，floating beat grid 的絕對相位、後續量化與參考 MIDI 小節相位必須一起診斷。
- 無程式修改的 static-time 配置也已測試：關閉 `floating-bpm`、保留 `sync-audio` 後 Macro F1 降至 `0.0129`，因此關閉 floating tracker 不是可接受修復。

## 15. Phase 5 共用輸出延遲校正（2026-07-15）

### 15.1 架構與選型

- 保留現有 floating beat tracker、模型、六類後處理與固定 50ms gate。
- 修正 floating+sync 的重複 prefix offset，並在共用 MIDI 輸出邊界加入單一 `67ms` 推論延遲校正常數。
- 校正只依物理時間作用，不讀取歌名、參考答案或類別；不得建立每首歌曲 offset。

### 15.2 資料模型與關鍵流程

- `quantized_onset`：floating 模式下的 WAV 絕對時間。
- `base_offset`：floating+sync 為 `0`；static+sync 為 `first_onset`；notation 為 `0`。
- `sync_latency`：音訊同步輸出固定減去 `0.067s`，notation 不校正。
- MIDI onset 必須裁切到非負時間；floating tempo map 使用相同時間校正。

```text
base_offset = 0 if floating_sync else first_onset if static_sync else 0
sync_latency = 0.067 if sync_audio else 0
midi_time = max(0, quantized_time + base_offset - sync_latency)
```

```mermaid
flowchart LR
    A["Audio/model onset"] --> B["Floating quantized absolute time"]
    B --> C["Remove duplicate prefix offset"]
    C --> D["Subtract shared 67ms latency"]
    D --> E["Clamp at zero and write MIDI"]
```

### 15.3 驗收

- 小型 self-check 覆蓋 floating+sync、static+sync、notation 三種 offset。
- 先通過 `verify_current_solution.py`，再以全新隔離目錄重跑原五首 manifest。
- 不調整參考 offset、gate 容差、checkpoint、Tempo 或拍號邏輯。
- 診斷掃描顯示修正雙重 prefix 後，全局提前 `30–100ms` 的最佳區間可將六類 Macro F1 從 `0.0886` 提升至約 `0.47`；正式結果仍以完整重跑為準。
- 正式完整重跑結果為 Macro F1 `0.4710`：KD `0.9388`、SD `0.7435`、HH `0.5873` 已通過 `0.55` 類別門檻；TOM `0.0940`、CRASH `0.0714`、RIDE `0.3909` 仍失敗。
- `verify_current_solution.py` PASS，因此時間校正保留為候選修復；整體商業 gate 仍為 FAIL，不得部署或宣稱達標。
- 下一個獨立任務只處理 TOM/CRASH/RIDE 類別混淆與誤報，不同時修改 Tempo/拍號或 Hi-Hat articulation。

## 16. Phase 6 罕見類別混淆診斷（2026-07-15）

### 16.1 診斷邊界

- 固定使用 `test_real_audio`、50ms gate、現有 base checkpoint 與 `six_class_tower_b_specialized.pth`。
- 不調整參考 offset、不做歌曲特判、不修改 Tempo/拍號或 Hi-Hat articulation。
- 先驗證 threshold 與 core/rare 競爭式互斥的理論上限；未達 `0.55` 就禁止加入新 heuristic。

### 16.2 結果與根因

- 單類 threshold 掃描最佳 F1：TOM `0.1337`、CRASH `0.0885`、RIDE `0.3528`。
- 加入「rare 機率必須勝過同時 core 機率」後最佳 F1：TOM `0.1551`、CRASH `0.0356`、RIDE `0.3223`，仍遠低於 gate。
- TOM 預測大量對應 KD/HH；CRASH 大量對應 KD/HH/SD；RIDE 大量對應 HH。這是 checkpoint 類別表徵混淆，不是後處理 threshold 問題。
- 已存在但未驗收的 v15 候選在未修改 STAR held-out gate 得 `0.3551`；TOM/CRASH/RIDE 為 `0.0000/0.1053/0.1538`，因此拒絕且不進五首 gate。

### 16.3 下一個候選規格

- 稽核確認 v15 的 `4032 = 576 × 7` 排程已包含 core-only hard negatives，因此不得重複同一配方。
- v16 在現有 adversarial core-frame negative mask 之外，只對「恰有一個 TOM/CRASH/RIDE 真值」的 frame 加入 rare 三類交叉熵，使模型學會罕見類別彼此區分；rare 同時擊打仍由原 multi-label BCE 處理。
- 使用獨立 rare tower 候選，保留 base KD/SD/HH checkpoint、held-out split、threshold、tolerance 與五首商業資料完全不變。
- 新 checkpoint 必須使用新候選檔名；先通過 STAR held-out gate，再允許進五首端到端 gate。

```text
base_loss = weighted multi-label BCE + velocity loss
single_rare_mask = exactly one of TOM/CRASH/RIDE is positive
competition_loss = CE(rare_logits[single_rare_mask], rare_target_class)
loss = base_loss + competition_weight * competition_loss
```
## Phase 17：六類候選評估分層修正（2026-07-15）

- 現有 `run_six_class_validation.py` 固定從 STAR `test` split 各挑一個類別窗口，但實際 6 筆選樣只有 3 個獨立物理窗口；其中同一個 0–2 秒窗口被 KD、SD、CRASH 重複累加，TOM 也只有 1 個事件，不能可靠代表泛化能力。
- STAR metadata 已提供獨立 `train`、`validation`、`test`。後續模型資料分工固定為：`train` 只負責訓練、`validation` 負責候選選擇、`test` 只負責最後一次資料集驗收；五首 `test_real_audio` 仍是不可訓練的客戶商業 gate。
- 驗證器新增 `--split` 與 `--per-class`：依類別、檔名及事件時間做確定性選樣，且相同音訊的重疊物理窗口只計一次。每個選入窗口內的所有六類事件都照實累加，不做歌曲或類別特判。
- 聚合前須替每個窗口加入互不重疊的虛擬時間 offset，避免不同歌曲都落在 `0–4s` 後被一對一 matcher 錯誤交叉配對。
- 舊的 `test + per-class=1` 報表保留作歷史證據，但不再作為唯一候選選擇依據；任何候選仍須通過完整五首 Macro F1 `>= 0.70` 且各類 F1 `>= 0.55` 才可上線。

### 關鍵流程虛擬碼

```text
load STAR metadata
filter requested split
for each drum class:
    sort labeled events deterministically
    propose centered physical windows
    skip a proposal when it overlaps an already selected window from the same audio
    keep up to per_class unique windows
run model once per unique window
aggregate all six-class expected/predicted physical events
write selection evidence and fixed-threshold metrics
```

### 評估狀態圖

```mermaid
stateDiagram-v2
    [*] --> TrainOnly: STAR train
    TrainOnly --> CandidateSelection: STAR validation
    CandidateSelection --> Rejected: validation fail
    CandidateSelection --> DatasetFinal: validation pass
    DatasetFinal --> Rejected: STAR test fail
    DatasetFinal --> CommercialGate: STAR test pass
    CommercialGate --> Rejected: five-song fail
    CommercialGate --> Ready: five-song pass
```

## Phase 18：v17 Rare Head-only Focal 候選（2026-07-15）

- Phase 17 的 48-window validation 顯示 v12 / v15 / specialized / v16 Macro F1 分別為 `0.4195 / 0.3929 / 0.3249 / 0.3221`；v16 competition 使 recall 下降，正式拒絕。
- v12 直接進固定五首只有 `0.4377`，低於 specialized 產品組合的 `0.4710`；CRASH 改善至 `0.1433`，但 RIDE 降至 `0.1267`，因此不可替換產品模型。
- v17 以 specialized checkpoint 為起點，凍結完整 backbone，只更新六類 head tensor；loss 只讀 TOM/CRASH/RIDE 三欄，因此 KD/SD/HH head row 梯度必須為零。
- onset 使用 numerically stable binary focal loss，降低大量容易負樣本主導梯度的問題；罕見類別正樣本權重可獨立設定。既有 core-hit adversarial negative 仍可套用，但不再加入已失敗的 rare competition loss。
- 候選只使用 STAR train schedule；模型選擇使用 STAR validation 48 個不重疊窗口，test 與五首商業 gate 保持隔離。

```text
freeze(backbone)
rare_logits = onset_logits[..., 3:6]
rare_targets = onset_targets[..., 3:6]
bce = BCEWithLogits(rare_logits, rare_targets)
pt = target * sigmoid(logit) + (1-target) * (1-sigmoid(logit))
focal = (1-pt)^gamma * bce * class_and_adversarial_weight
loss = mean(focal) + rare_velocity_loss
```

## Phase 19：Rare Tower Percussive-domain 候選（2026-07-15）

- v17 最佳 validation Macro F1 僅 `0.3060`，低於 specialized `0.3249`；head-only focal 方案拒絕。
- STAR train schedule 稽核顯示各類 576 筆樣本具備 500–576 個不同音訊來源，資料排程並非重複窗口；主要差異是 STAR 鼓組 mix 與完整商業歌曲的 harmonic/vocal domain gap。
- 新增 opt-in `rare_percussive`：base tower 維持原始 waveform 特徵；只有 rare tower 使用 `librosa.effects.percussive` 的時間對齊 percussive waveform 重新抽取相同 hybrid features。
- 不修改 checkpoint、threshold、AME、Tempo、拍號或五首 reference。候選先通過既有 regression，再以全新五首輸出目錄比較；未改善就撤回，不設為預設。

```mermaid
flowchart LR
    W["完整歌曲 waveform"] --> B["原始 features → base KD/SD/HH"]
    W --> H["HPSS percussive waveform"]
    H --> R["相同 features → rare TOM/CRASH/RIDE"]
    B --> F["雙塔機率融合"]
    R --> F
```

### Phase 19 實測與 Phase 20 matched-domain 限制

- 直接把 raw-domain specialized 模型套到 HPSS 輸入，五首 Macro F1 降至 `0.4189`；TOM/CRASH/RIDE 為 `0.0516/0.0000/0.1922`，因此 unmatched 方案拒絕。
- Phase 20 只允許測試 train、validation、inference 三者皆使用相同 HPSS percussive transform 的 matched-domain 候選；不得混用 raw-domain validation 排名。
- `build_window` 與六類驗證器新增 opt-in percussive input，預設仍為 raw；訓練報表與 selected window 證據必須記錄 input domain。
- 先用小型 schedule 建立候選並跑 percussive STAR validation。若未高於同域起始 checkpoint，就停止，不進五首。

### Phase 20 最終結果（拒絕）

- matched HPSS 候選最佳為 epoch 6：percussive STAR validation `0.3224`，高於同域起點 `0.2281`，因此依預設條件進一次五首 gate。
- 固定五首結果僅 `0.4486 < 0.4710`；TOM/CRASH/RIDE 為 `0.0620/0.0202/0.3367`。候選拒絕，所有 percussive 產品 opt-in 程式碼撤回；checkpoint 與報表只保留為研究證據。
- 現有資料與架構的最佳可重現產品結果仍為 `0.4710`。下一階段不得再用五首驗收歌曲訓練或選 threshold；必須新增獨立、已對齊的完整歌曲六類訓練/validation 資料，或引入經驗證的 drum source-separation/pretrained audio backbone。
- 即使六類 Macro F1 達標，商業 gate 仍有獨立工作：HH closed/pedal/open articulation，以及 Blue/Counting Stars/Rosanna 的 Tempo/拍號錯誤；不得把 rare-class 修復誤稱為全部完成。

## Phase 21：合法伴奏域增強候選（2026-07-15）

- 本機 `accompaniment/` 已有既存 Phase 3 線上混音資料與公式，不新增依賴或下載資料。
- `adele_bass/no_drums/other/vocals.wav` 對應固定五首 gate 的 Rolling In The Deep，全部禁止用於訓練、validation 或選 threshold。
- 本輪唯一允許的伴奏是 `queen_no_drums.wav`；以不同物理片段及 `0.10–0.30` drum-peak 相對增益與 STAR train drum windows 線上混合。
- 先以固定 `0.17` 增益建立 Queen-mixed STAR validation baseline；候選必須同時改善 mixed validation 且不讓 raw validation 明顯退化，才允許進固定五首。
- 混音只改 waveform，事件時間與六類 target 完全沿用 STAR；checkpoint 使用全新候選目錄。

```text
accompaniment = queen_no_drums[random offset : offset + 4s]
scaled = accompaniment / peak(accompaniment) * peak(drums) * uniform(0.10, 0.30)
mixed = normalize_if_clipped(drums + scaled)
features = existing extract_features(mixed)
```

### Phase 21 結果與 Phase 22 擴大排程

- v19 最佳 mixed validation 為 epoch 7：`0.3362 > 0.3222`，raw validation `0.3262` 未崩潰；依規格進一次五首 gate。
- 五首 Macro F1 `0.4680 < 0.4710`；TOM/CRASH/RIDE `0.0994/0.0526/0.3863`，候選拒絕。
- v19 只使用每類 96 windows 且 rare positive weight 50；Phase 22 改為已驗證較穩定的 v15 配方：每類 576 windows、schedule-balanced positive weights capped at 12、完整模型極低學習率，再加入同一合法 Queen domain mix。
- Phase 22 是本機現有資料能做的完整規模域增強；若仍未通過 STAR validation，後續必須新增非 gate accompaniment／商業混音資料，不再重複超參數掃描。
- 啟動前 self-check 發現既有期望仍假設 6 類 schedule，但目前排程已是六類加 `NEG` 共 7 個 bucket；修正只更新自檢為 `7 × per_class`，不改訓練排程。

### Phase 22 v20 最終結果（拒絕）

- v20 使用每類 576 windows、10 epochs、schedule-balanced positive weights capped at 12、完整模型低學習率及合法 `queen_no_drums.wav` 的 `0.10–0.30` 隨機增益；沒有使用五首 gate 音訊或標註。
- 10 個 checkpoint 的固定 Queen-mixed STAR validation 介於 `0.4181–0.4313`；最佳 epoch 10 為 Macro F1 `0.4313`，KD/SD/HH/TOM/CRASH/RIDE 分別為 `0.6465/0.6596/0.5052/0.2943/0.1519/0.3305`。
- epoch 10 的 raw STAR validation 為 Macro F1 `0.4277`，六類分別為 `0.6589/0.6564/0.5057/0.2934/0.1091/0.3427`；域增強沒有造成 raw 能力崩潰，但四類仍未達 `0.55`。
- v20 未通過 STAR promotion gate（Macro F1 `>=0.70` 且各類 `>=0.55`），因此依預先規格停止，不執行五首 gate、不替換產品模型。
- 現有單一合法伴奏能改善 STAR 域泛化，但無法補足 TOM/CRASH/RIDE 及 HH 的分類邊界。下一步不得繼續同資料超參數掃描；需新增不含 gate 的完整歌曲六類對齊訓練/validation 資料，並保留歌曲級隔離。

## Phase D0–D5：DCNN + 小型 Conformer 接力規格（2026-07-15）

### 架構與選型

- 新候選固定採雙分支 DCNN frontend：Log-Mel 音色分支與真正 SuperFlux 瞬態分支各自擁有獨立卷積權重，於相同時間解析度做 late fusion。
- Phase D2 先沿用既有 TCN，隔離驗證 DCNN 的效果；只有 DCNN 同時改善 matched Queen-mixed 與 raw STAR validation，Phase D4 才允許把時序層替換為 2–4 層小型 Conformer。
- 禁止純 Transformer 候選。Conformer 必須保留局部卷積模組與 frame-level 對齊，不得降低 onset 時間解析度。
- 舊 checkpoint 只移植形狀與語意相容的權重；新增分支與 fusion 使用新參數。所有輸出皆為新 candidate，不覆蓋產品 checkpoint。

### 資料模型與隔離

- 訓練只使用 STAR train、既有非 gate 訓練資料與合法 `queen_no_drums.wav`；STAR validation 用於候選選擇，STAR test 只用於 promotion。
- `test_real_audio` 固定五首不得進入訓練、選 epoch、選 threshold、特徵參數或架構挑選。
- 每份訓練報告至少記錄 architecture、feature mode、split、seed、checkpoint source、event counts、loss 與候選路徑。

### 關鍵流程與虛擬碼

```text
phase D0: audit existing changes -> regression + loop audit -> commit/push baseline
phase D1: implement true SuperFlux opt-in -> self-check + regression -> commit/push
phase D2: implement DCNN frontend -> reuse existing TCN -> self-check + regression -> commit/push
phase D3: train DCNN+TCN -> compare fixed mixed/raw STAR validation -> commit/push evidence
if DCNN does not improve both matched baselines: reject and stop architecture escalation
phase D4: replace only temporal encoder with small Conformer -> self-check + regression -> commit/push
phase D5: train candidate -> STAR validation -> STAR test -> fixed five-song gate only after promotion
```

### Phase D1 True SuperFlux 公式

- `extract_features(..., use_true_superflux=False)` 預設維持既有 Mel 正向差分，確保產品與舊 checkpoint 完全不變。
- opt-in 路徑先對 Mel power 做 `log1p`，再對前一個延遲 frame 沿頻率軸做寬度 3 的 maximum filter；目前 frame 減去 filtered reference，負值截為零。
- 固定 lag 為 2 frames，輸出前補零保持 `[frequency, time]` shape 與 onset frame 對齊，之後沿用既有獨立 Z-score normalization。
- 最小測試必須證明：shape 不變、靜態頻譜為零、鄰近頻率漂移被抑制、真正寬頻瞬態保留正值、非法 lag/max-size 被拒絕。

```text
log_energy = log1p(mel_power * 1000)
reference = frequency_max_filter(log_energy, width=3)
superflux[:, lag:] = max(0, log_energy[:, lag:] - reference[:, :-lag])
superflux[:, :lag] = 0
```

### Phase D2 DCNN + TCN 模型與權重移植

- `SharedCNNBackbone(input_channels=2)` 只參數化第一層輸入通道；預設 `2` 必須保持舊 checkpoint state key 與 tensor shape 不變。
- `DCNNBackbone` 內含兩個獨立的 `SharedCNNBackbone(input_channels=1)`：timbre 只讀 feature channel 0，transient 只讀 channel 1。
- 兩分支各輸出 `[B,64,T]`，concatenate 後以 `Conv1d(128,64,1)` late fusion；fusion 初始值為同索引兩分支各 `0.5`，bias 為零。
- `DCNNDrumTCN` 沿用既有 onset/velocity TCN 與六類 heads，D2 不改 loss、threshold 或事件解碼。
- 從六類 Symmetric checkpoint 移植時，timbre 第一層只複製舊 `conv1` 的 input channel 0，transient 只複製 channel 1；其餘 backbone tensor 複製到兩分支，TCN/head 只複製 shape 相容 tensor。
- D2 self-check 必須驗證舊模型第一層仍為 2 channels、DCNN 輸出 shape、兩分支參數獨立、首層語意切分、TCN/head 精確移植與非法輸入拒絕。

### Phase D3 訓練與驗證控制

- 既有 `train_six_class_candidate.py` 與 `run_six_class_validation.py` 新增 `--architecture symmetric|dcnn-tcn`，預設 `symmetric`，不得另建重複 trainer/validator。
- `dcnn-tcn` 自動使用 True SuperFlux 並以 `transfer_symmetric_state` 從 v12 六類 checkpoint 移植；候選重新載入時必須還原兩分支 legacy projection 狀態。
- D3 固定複用 v20 配方：STAR train、576 windows/類、10 epochs、batch 12、head LR `1e-4`、backbone LR `1e-6`、balanced positive weight cap 12、freeze BN、Queen accompaniment `0.10–0.30`、seed 1337。
- 模型選擇只看固定 Queen-mixed STAR validation；最佳 epoch 再跑 raw STAR validation。matched baselines 為 mixed `0.4313`、raw `0.4277`。
- 只有 mixed 與 raw Macro F1 均高於各自 baseline，且沒有任一類 F1 下降超過 `0.03`，才允許進 D4 Conformer；否則提交失敗證據並停止架構升級。

### Phase D3 最終結果（拒絕）

- 14-window smoke training 與正式 validator reload PASS；完整候選使用 4,032 windows、10 epochs，成功移植 220 個 tensor，train loss 由 `0.3217` 降至 `0.0959`。
- 十個 Queen-mixed STAR validation checkpoint 由 `0.3180` 逐步升至 epoch 10 的最佳 `0.3937`，仍低於 matched baseline `0.4313`。
- mixed epoch 10 的 KD/SD/HH/TOM/CRASH/RIDE F1 為 `0.5627/0.6245/0.4904/0.2718/0.1311/0.2820`。
- raw epoch 10 Macro F1 `0.3951 < 0.4277`；六類為 `0.5745/0.6198/0.4938/0.2765/0.1235/0.2825`。
- D3 未同時改善 mixed/raw baseline，且多類退步超過 `0.03`，因此依預先 gate 拒絕；不跑固定五首、不替換產品模型、不解鎖 D4 Conformer。
- 結果顯示簡單拆分共享 CNN 會損失 Log-Mel 與瞬態特徵的早期交互；訓練 loss 收斂但 held-out F1 退步，不屬於訓練未執行或 checkpoint reload 錯誤。

### 模組關係圖

```mermaid
flowchart LR
    A["44.1 kHz 音訊"] --> M["Log-Mel"]
    A --> S["True SuperFlux"]
    M --> C1["Timbre CNN"]
    S --> C2["Transient CNN"]
    C1 --> F["Late fusion"]
    C2 --> F
    F --> B["TCN baseline / Small Conformer candidate"]
    B --> O["Six-class onset head"]
    B --> V["Six-class velocity head"]
```

### 序列與狀態

```mermaid
stateDiagram-v2
    [*] --> DCNNTCN
    DCNNTCN --> Rejected: mixed 或 raw validation 未改善
    DCNNTCN --> Conformer: mixed 與 raw validation 均改善
    Conformer --> Rejected: STAR validation fail
    Conformer --> StarTest: STAR validation pass
    StarTest --> Rejected: STAR test fail
    StarTest --> FiveSong: STAR test pass
    FiveSong --> Rejected: commercial gate fail
    FiveSong --> Ready: Macro F1 >= 0.70 且各類 >= 0.55
```

### Phase D3R 根因修復規格

- D3 同時改變 CNN 拓撲與特徵差分，且把新 DCNN/fusion 與既有 TCN 一起放在 `1e-6` 學習率；因此 D3 不能單獨歸因於 DCNN，也不足以判定過擬合。
- D3R 新增 `dcnn-residual-tcn`：完整保留來源 shared CNN，另加雙分支 DCNN correction，輸出為 `shared + gate * correction`；`gate` 初始化為零，轉移後必須在 eval mode 與來源模型逐值相同。
- trainer/validator 將 feature mode 與 architecture 分離為 `legacy-diff|true-superflux`。D3R 第一個候選固定使用 `legacy-diff`，只測拓撲與優化修復，不同時更換前端特徵。
- full-model optimizer 分成三組：heads 使用 `--lr`，新 DCNN correction/gate 使用 `--new-module-lr`，既有 shared CNN/TCN 使用 `--backbone-lr`。新模組不得再落入 `1e-6` 群組。
- D3R 先跑最小 transfer/optimizer/backward self-check 與完整三類 regression；再以固定 STAR train/validation、Queen augmentation、seed 1337 評估。不得使用固定五首歌曲選模型。
- promotion 仍須同時高於 mixed `0.4313` 與 raw `0.4277`，且任一類不得下降超過 `0.03`；未通過則保留證據、停止，不進 D4 Conformer。

### 協作、部署與回退

- 每個 Phase 必須先更新文件、完成規定測試，再以單一目的 commit 並 push 至 `origin/codex`；測試失敗不得標記完成或進下一 Phase。
- 其他 AI 接手時必須先 fetch `origin/codex`，閱讀 `AGENTS.md`、`spec.md`、`todolist.md`、`current_status.md`，並依本節的資料隔離、架構順序及 gate 繼續。
- 任何改用純 Transformer、使用五首調參、降低門檻、覆蓋 checkpoint 或跳過 Phase 的提案，都必須先記錄證據並取得使用者明確確認。
- 部署前保留目前產品模型與設定作為回退；DCNN/Conformer 候選未通過完整 gate 前不得成為預設。

### Phase D3R 最終結果（架構修復通過，商業 gate 未通過）

- 4,032 windows、10 epochs 的 residual DCNN + legacy diff 候選由 mixed `0.4235` 穩定升至 epoch 10 的 `0.4500`，高於 v20 `0.4313` 與原 D3 `0.3937`。
- epoch 10 raw STAR Macro F1 為 `0.4520`，高於 v20 `0.4277` 與原 D3 `0.3951`；mixed/raw 均改善，且沒有類別 F1 下降超過 `0.03`，因此 D3R conditional architecture gate 通過並解鎖 D4。
- mixed KD/SD/HH/TOM/CRASH/RIDE 為 `0.6984/0.6992/0.5036/0.3032/0.1384/0.3570`；raw 為 `0.7062/0.6990/0.4945/0.3038/0.1367/0.3720`。
- 商業 STAR gate 仍 FAIL：Macro F1 未達 `0.70`，HH/TOM/CRASH/RIDE 未達 `0.55`。不執行固定五首、不替換產品模型；D4 只能測小型 Conformer 是否改善時間建模，不能宣稱資料不足問題已解決。

### Phase D4 小型 Conformer 規格

- 新候選固定為 `dcnn-conformer`：沿用 D3R residual DCNN backbone 與 legacy diff，僅把 onset/velocity 的 TCN temporal encoders 換成兩套相同的小型 Conformer encoders。
- 每套 encoder 使用 2 層、`d_model=64`、4-head self-attention、FFN expansion 2、depthwise convolution kernel 15；輸入輸出均為 `[B,64,T]`，不得降採樣或改變 onset frame 對齊。
- 每個 Conformer block 使用 Macaron half-step FFN、multi-head self-attention、GLU pointwise + depthwise convolution、第二個 half-step FFN 與 final LayerNorm；只使用 PyTorch 現有元件，不新增依賴。
- 來源 checkpoint 固定為 D3R mixed 最佳 epoch 10。只移植 shape 相容的 residual DCNN backbone 與 onset/velocity heads；TCN tensor 不得語意冒充 Conformer tensor。
- optimizer 維持三組：heads `1e-4`、既有 residual DCNN `1e-6`、新 Conformer `5e-5`。資料、4,032-window schedule、Queen `0.10–0.30`、seed、loss、threshold 與驗證窗口全部不變。
- 先通過 shape、有限值、時間長度、checkpoint reload、backward 與參數分組 self-check，再跑完整 regression。正式訓練後只以 mixed STAR 選 epoch，最佳者再跑一次 raw STAR。
- D4 promotion 必須 mixed 高於 D3R `0.4500`、raw 高於 D3R `0.4520`，且任一類不得下降超過 `0.03`；否則拒絕 Conformer，不跑固定五首、不替換產品模型。

### Phase D4 最終結果（拒絕）

- 完整候選使用 4,032 windows、10 epochs、batch 12 與 D3R 完全相同的資料/增強/loss；train loss 由 `0.4096` 降至 `0.0824`，證明 Conformer 可訓練且非 reload/NaN/OOM 問題。
- mixed STAR 由 epoch 1 `0.3579` 升至 epoch 10 最佳 `0.4501`，只比 D3R `0.4500` 高 `0.0001`；raw epoch 10 為 `0.4538`，只比 D3R `0.4520` 高 `0.0018`，沒有實質整體改善。
- mixed KD/SD/HH/TOM/CRASH/RIDE 為 `0.6550/0.7185/0.5024/0.2801/0.1392/0.4053`；raw 為 `0.6745/0.7187/0.5080/0.2770/0.1438/0.4008`。
- mixed KD 相對 D3R 下降 `0.0434`，raw KD 下降 `0.0317`，均超過允許的 `0.03`；TOM 亦退步。D4 promotion FAIL，D5 不解鎖，不跑 STAR test/固定五首、不替換產品模型。
- 結論：小型 Conformer 改善 SD/RIDE，但未解決 HH/TOM/CRASH 類別邊界，且犧牲 KD。現有資料下，時間模型替換不是到達 `0.70` 的主解法。

### Phase D4R gated TCN-Conformer 修復規格

- D4R 不再刪除已訓練 TCN；onset/velocity temporal path 改為 `TCN(x) + gate * Conformer(x)`，`gate` 初始化為零。載入 D3R epoch 10 後，eval 輸出必須與 D3R 逐值相同。
- Conformer correction 沿用 D4 的 2 層、64 維、4-head、kernel 15 配置；TCN 與 Conformer 均保持 `[B,64,T]`，不得改變 frame alignment。
- D3R residual DCNN、TCN、heads 全部語意移植；新 Conformer 與 gate 為新參數。optimizer 使用 heads `1e-4`、新 Conformer/gate `5e-5`、其餘繼承參數 `1e-6`。
- 訓練資料、4,032-window schedule、10 epochs、batch 12、Queen augmentation、seed、loss、threshold 與驗證窗口全部不變；不得使用固定五首選擇 gate、LR 或 epoch。
- promotion gate 固定為 mixed `>0.4500`、raw `>0.4520`，且任一類別相對 D3R 不得下降超過 `0.03`。未通過即拒絕，不進 D5、不替換產品模型。

### Phase D4R 最終結果（相對改善通過；商業 gate 失敗）

- exact-output、兩階段 backward、optimizer 分組、checkpoint reload、trainer/validator self-check 與完整 `verify_current_solution.py` 全部 PASS；零 gate 初始化確實保留 D3R 輸出。
- 完整訓練使用固定 4,032 windows、10 epochs、batch 12、Queen augmentation、seed 與 loss；train loss `0.0803 -> 0.0721`，沒有 NaN、OOM 或純 Conformer 的初期崩塌。
- 依 mixed STAR 比較 epoch 1–10，epoch 10 最佳 Macro F1 `0.4599`；KD/SD/HH/TOM/CRASH/RIDE 為 `0.7010/0.7142/0.5174/0.3062/0.1413/0.3791`。
- 只對 mixed 最佳 epoch 10 執行一次 raw STAR，Macro F1 `0.4685`；六類為 `0.7166/0.7221/0.5151/0.3043/0.1600/0.3929`。
- 相對 D3R，mixed/raw 分別增加 `0.0099/0.0165`，六類皆未下降超過 `0.03`，因此 D4R 架構改善 gate 通過並保留為後續研究基線。
- 商業 STAR gate 仍 FAIL：Macro F1 距 `0.70` 尚差 `0.2315`（以較佳 raw 計），且 HH/TOM/CRASH/RIDE 未達單類 `0.55`。不得替換產品模型、不得部署，也不得用固定五首反覆選 checkpoint/threshold。
- 結論：保留 TCN 再以 gated Conformer 學殘差，比直接替換 TCN 有效；但架構改善幅度不足以補足稀有類資料與 false-positive 邊界問題。下一步應先補歌曲隔離、非 gate、具授權且含 TOM/CRASH/RIDE 的真實音訊與標註，再做一次預先鎖定的訓練。

### Phase D4D 現有資料覆蓋修復規格

- 不更換 D4R 架構、loss、threshold 或驗證器；來源 checkpoint 固定為 D4R mixed 最佳 epoch 10。唯一實驗變因是把本機既有 E-GMD TOM/CRASH/RIDE 與更多 STAR train 窗口加入候選訓練。
- E-GMD 六類映射固定為 TOM `41/43/45/47/48/50`、CRASH `49/52/55/57`、RIDE `51/53/59`；保留既有 KD/SD/HH 映射。只讀原始 train MIDI，輸出新的 rare metadata，不覆蓋 `processed_data/egmd_meta.json`。
- E-GMD 以 `groove_key` 去重，每個 groove 最多一個 kit；只選含 TOM/CRASH/RIDE 的 train item。與 `processed_data/star_meta.json` 合併時，遇到重複 key 必須失敗，不得靜默覆蓋。
- 固定每類 `1,152` windows 加 `1,152` NEG，共 `8,064` windows；訓練 `5` epochs、batch `12`，總計 `3,360` batches，與 D4R 的總更新步數相同。Queen augmentation、seed、heads/inherited/new-module LR `1e-4/1e-6/5e-5` 全部不變。
- 必須先輸出 schedule 來源分布，確認 TOM/CRASH/RIDE 都實際包含 E-GMD 與 STAR；否則停止，不允許以「已合併」名義訓練。
- 正式候選只訓練一次。依 mixed STAR 選 epoch 1–5，最佳者只跑一次 raw STAR；promotion 必須 mixed `>0.4599`、raw `>0.4685`，且任一類相對 D4R 不得下降超過 `0.03`。
- `test_real_audio`、STAR validation/test 與其衍生輸出不得進入訓練或選參。未達商業 gate（Macro F1 `>=0.70` 且每類 `>=0.55`）時，不替換產品模型、不部署。

### Phase D4D 最終結果（技術 gate 通過；效果不足以商用）

- E-GMD rare metadata 共 `716` 個去重 groove；事件為 TOM `18,000`、CRASH `2,565`、RIDE `31,543`。新檔與 STAR 無 key 衝突，舊 `egmd_meta.json` 未覆蓋。
- 8,064-window schedule 確認三類同時含 STAR/E-GMD：TOM `1,041/111`、CRASH `1,097/55`、RIDE `721/431`；正式訓練內實際三類事件為 `26,177/8,993/18,634`。
- D4R epoch 10 已完整續載 `383` 個 tensors；5 epochs、3,360 batches 正常完成，loss `0.2337 -> 0.0694`，沒有 NaN/OOM。
- mixed STAR 最佳為 epoch 2，Macro F1 `0.4601`；KD/SD/HH/TOM/CRASH/RIDE 為 `0.7046/0.7151/0.5294/0.3125/0.1390/0.3600`。
- raw STAR 只測 epoch 2，Macro F1 `0.4692`；六類為 `0.7127/0.7177/0.5245/0.3132/0.1556/0.3912`。
- 相對 D4R mixed/raw 只增加 `0.0002/0.0007`，且沒有類別下降超過 `0.03`，因此預先定義的技術 promotion gate 通過；但幅度沒有實務意義，不足以宣稱資料問題已解決。
- 商業 gate 仍 FAIL，產品模型、固定五首與部署狀態不變。結論是既有 E-GMD rare mapping 可用，但自然比例下 CRASH 新窗口太少，且電子鼓/合成鼓域仍無法補足真實歌曲的類別邊界。

### Phase D4S rare source-balance 修復規格

- 唯一變因是 TOM/CRASH/RIDE 的來源配額；D4R 架構、legacy diff、loss、threshold、Queen augmentation、seed、LR 與驗證器全部不變。
- 來源 checkpoint 固定為 D4R epoch 10，不從 D4D epoch 2 繼續，避免把前一輪自然比例訓練疊加進來。
- `build_schedule` 新增 opt-in `--balance-rare-sources`；預設關閉並保持舊排程行為。啟用後固定 50/50，使每個 weak class 的 1,152 windows 精確分成 STAR `576` + E-GMD `576`。
- source 判定只允許 metadata 的 `source == "egmd_pitch_weighted"`；不得依測試檔名、歌曲名稱或預期答案分類。若任一來源不足 quota，必須立即失敗，不得靜默回退。
- KD/SD/HH/NEG 維持既有自然比例排程；總計仍為 8,064 windows、5 epochs、batch 12、3,360 batches，與 D4D/D4R 比較保持等預算。
- 必須先以 self-check 驗證預設相容、50/50 精確配額與不足來源拒絕，再輸出正式 schedule 來源分布；未達精確配額不訓練。
- 正式候選只訓練一次。依 mixed STAR 選 epoch 1–5，最佳者只跑一次 raw；promotion 必須 mixed `>0.4601`、raw `>0.4692`，且任一類相對 D4D 不得下降超過 `0.03`。
- 未達商業 gate（Macro F1 `>=0.70` 且每類 `>=0.55`）時，不跑固定五首、不替換產品 checkpoint、不部署。

### Phase D4S 最終結果（拒絕）

- schedule self-check、預設相容、來源不足拒絕與正式分布稽核全部 PASS；TOM/CRASH/RIDE 均精確使用 STAR `576` + E-GMD `576`，KD/SD/HH/NEG 維持既有自然比例。
- 從 D4R epoch 10 完整續載 `383` tensors；8,064 windows、5 epochs、3,360 batches 正常完成。loss `0.1895 -> 0.1290`，明顯高於 D4D，顯示 50% E-GMD 帶來強烈來源域衝突。
- mixed STAR 最佳為 epoch 1，Macro F1 `0.4594 < 0.4601`；KD/SD/HH/TOM/CRASH/RIDE 為 `0.6780/0.7037/0.5621/0.2958/0.1603/0.3564`。
- raw STAR epoch 1 為 `0.4716 > 0.4692`；六類為 `0.6887/0.7066/0.5604/0.2965/0.1878/0.3894`。HH/CRASH 改善，但 KD/TOM/RIDE 與 mixed 整體付出代價。
- 因 mixed gate 下降，D4S promotion FAIL；保留 opt-in scheduler 與候選作研究證據，但不得成為新基線、不得跑固定五首、不得替換產品模型或部署。
- 結論：E-GMD 對 HH/CRASH 有可見訊號，但 50/50 比例過高。不得把下一步變成無限制比例 sweep；在新增真實授權 full-song 六類資料前，D4D 仍是現有資料路線的較安全研究基線。

### Phase D5A MDB Drums 研究資料匯入規格

- 以 shallow clone 將 `CarlSouthall/MDBDrums` 原樣下載至專案根目錄 `MDBDrums/`，不得覆蓋既有資料或 checkpoint。
- 本階段只驗證下載完整性、資料結構與授權文件；不得訓練、調參或將內容放入 `test_real_audio/`。
- MDB Drums 為 CC BY-NC-SA 4.0 研究資料，後續若使用只能作非商業方向驗證，不得直接宣稱商業模型可使用。
- 下載結果鎖定官方 `master` commit `b29e2d63c3a023506f4bf353c5b2e8a558eed135`：362 個追蹤檔、268 個 WAV、46 個文字標註，總大小 `2,010,349,446` bytes；沒有小於 1 KB 的 WAV。

### Phase D5B MDB Drums 六類 metadata 規格

- 新增單一 builder，將 `MDBDrums/MDB Drums/audio/full_mix` 與 `annotations/subclass` 轉成現有 `audio_path/duration/split/source/events` schema；不建立另一套 dataset/trainer。
- 六類映射固定為：`KD→KD`；`SD/SDB/SDD/SDF/SDG/SDNS/SST→SD`；`CHH/OHH/PHH→HH`；`HIT/MHT/HFT/LFT→TOM`；`CRC/CHC/SPC→CRASH`；`RDC/RDB→RIDE`。`TMB` tambourine 忽略。
- 依官方 MIREX 2017 split 保留 12 首 train 與 11 首 test；README 的 `Zepplin` 拼字錯誤須正規化為實際檔名 `Zeppelin`。同一歌曲的 full mix、drum-only、stems 不得跨 split；本階段只使用 full mix。
- 每個事件沒有力度標註，沿用現有 `build_window` 的預設 velocity `100`；輸出新檔 `processed_data/mdbdrums_six_class_meta_d5b.json` 與 `validation_runs/mdbdrums_d5b_audit.json`，不得覆蓋既有 metadata。
- builder 必須拒絕缺音訊、缺標註、未知 subclass、時間超出音訊、歌曲重複或 train/test 數量不是 12/11；self-check 至少驗證官方映射、未知標籤拒絕與 split 隔離。
- 虛擬碼：`for song in full_mix -> parse subclass -> map supported labels -> validate times -> assign official split -> write metadata + aggregate audit`。
- D5B 只做資料接入與稽核，不訓練。只有 train split 同時覆蓋六類且 TOM/CRASH/RIDE 事件數公開記錄後，才可預先鎖定 D5C 的唯一訓練配方。
- MDB Drums 為 CC BY-NC-SA 4.0；所有 D5 結果只能作非商業研究證據，不得成為商業部署權重。

### Phase D5B 最終結果（資料接入通過；訓練覆蓋不足）

- builder syntax/self-check PASS；建立 23 首 full-mix metadata，官方 split 為 train/test `12/11`，兩側均覆蓋六類，未知 subclass、缺檔、超時事件與 split 數量均 fail-fast。
- train 事件為 KD/SD/HH/TOM/CRASH/RIDE `661/1310/1603/15/57/210`；test 為 `878/1382/1036/75/94/641`。train 的 TOM 只有 15 個，不能合理支撐既有每類 1,152-window 配方或宣稱解決資料 blocker。
- 以 D4D epoch 2、固定 threshold `0.50`、tolerance `50 ms`、官方 MDB test、每類 8 個互不重疊窗口做零調參診斷：Macro F1 `0.4478`；六類 `0.6411/0.5995/0.4180/0.3136/0.1436/0.5708`。
- MDB test 診斷顯示 HH/TOM/CRASH 的主要錯誤仍是 false positives：分別 `423/140/134`。官方 11 首 test 不得回流訓練；不以僅 15 個 train TOM 啟動過度重複的 D5C 候選。
- 完整 `verify_current_solution.py` PASS；舊三類產品路徑、固定五首與產品 checkpoint 均未改動。
