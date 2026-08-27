# loop-run-log.md

Append one entry per loop run. Keep entries concise.

## Recent Runs

```json
{
  "run_id": "2026-07-12T03:20:00+08:00",
  "pattern": "manual-validation",
  "level": "L1",
  "outcome": "blocked",
  "commands": [
    ".\\.venv\\Scripts\\python.exe -u train_six_class_candidate.py --meta validation_runs\\six_class_smoke\\star_six_class_meta.json --output-dir validation_runs\\six_class_candidate_v7 --candidate-name six_class_candidate_v7.pth --per-class 96 --batch-size 16 --epochs 30 --lr 0.001 --backbone-lr 0.00002 --full-model --gaussian-targets --schedule-balanced-weights --freeze-bn --log-every 36",
    ".\\.venv\\Scripts\\python.exe -u run_six_class_validation.py --meta validation_runs\\six_class_smoke\\star_six_class_meta.json --model validation_runs\\six_class_candidate_v7\\six_class_candidate_v7.pth --output-dir validation_runs\\six_class_candidate_v7\\heldout_validation"
  ],
  "notes": "Candidate v7 used only STAR split=train and did not read Round5 or test_real_audio. It trained 576 balanced windows for 30 epochs but the fixed STAR split=test gate failed at macro F1 0.0000 with zero predicted events for all six labels at the predeclared 0.50 threshold. Candidate is rejected; automatic further training is stopped pending a documented and approved different plan."
}
```

```json
{
  "run_id": "2026-07-17T22:40:00+08:00",
  "pattern": "manual-d10-safe-true-superflux-specaugment",
  "level": "human-approved",
  "duration_s": 7200,
  "items_found": 2,
  "actions_taken": 5,
  "escalations": 0,
  "tokens_estimate": 30000,
  "outcome": "rejected",
  "notes": "Added opt-in synchronized 0-12-bin training-only frequency masking, completed 20 epochs with 2048 FFT Log-Mel plus True SuperFlux on 6GB VRAM, independently reproduced best macro F1 0.4584, and rejected D10 because it stayed below D7 0.4601 while KD fell by 0.0737. Product and fixed-five gates were untouched."
}
```

```json
{
  "run_id": "2026-07-17T03:40:00+08:00",
  "pattern": "manual-d7-d4d-earlystop20",
  "level": "human-approved",
  "duration_s": 2100,
  "items_found": 1,
  "actions_taken": 4,
  "escalations": 0,
  "tokens_estimate": 30000,
  "outcome": "no-improvement",
  "notes": "Added per-epoch six-class validation and patience-5 early stopping, completed 7 of 20 epochs, and stopped after epochs 3-7 failed to beat epoch 2. Best mixed STAR remained 0.4601 with KD/SD/HH/TOM/CRASH/RIDE 0.7046/0.7151/0.5294/0.3125/0.1390/0.3600. Product and fixed-five gates were untouched."
}
```

```json
{
  "run_id": "2026-07-17T04:10:00+08:00",
  "pattern": "manual-d8-six-class-confusion",
  "level": "human-approved",
  "duration_s": 600,
  "items_found": 1,
  "actions_taken": 3,
  "escalations": 0,
  "tokens_estimate": 12000,
  "outcome": "diagnostic-complete",
  "notes": "Generated a row-normalized 6x6 confusion matrix for D7 best on the unchanged STAR mixed validation. Largest within-class confusions were CRASH-to-SD 20.00%, CRASH-to-HH 20.00%, RIDE-to-HH 16.28%, and TOM-to-KD 13.46%; rare-class extra prediction rates remained 61.28%-83.33%. No training or product changes."
}
```

```json
{
  "run_id": "2026-07-17T04:35:00+08:00",
  "pattern": "manual-d9-auto-confusion-report",
  "level": "human-approved",
  "duration_s": 900,
  "items_found": 1,
  "actions_taken": 4,
  "escalations": 0,
  "tokens_estimate": 16000,
  "outcome": "implemented",
  "notes": "Integrated best-checkpoint confusion reporting into every six-class fine-tune with held-out validation. Added F1-sorted class_health.csv and verified the full path with an isolated one-batch candidate. Runs without validation metadata do not claim a quality report."
}
```

```json
{
  "run_id": "2026-07-15T23:58:11+08:00",
  "pattern": "manual-d5b-mdbdrums-ingest",
  "level": "human-approved",
  "duration_s": 900,
  "items_found": 2,
  "actions_taken": 4,
  "escalations": 0,
  "tokens_estimate": 14000,
  "outcome": "fix-proposed",
  "notes": "Built isolated MDBDrums six-class metadata with official 12/11 song split. Train rare counts were TOM 15, CRASH 57, RIDE 210, so training was not started. D4D zero-tune MDB test macro F1 was 0.4478; full regression passed."
}
```

```json
{
  "run_id": "2026-07-15T23:35:27+08:00",
  "pattern": "manual-mdbdrums-download",
  "level": "human-approved",
  "duration_s": 360,
  "items_found": 1,
  "actions_taken": 1,
  "escalations": 0,
  "tokens_estimate": 7000,
  "outcome": "fix-proposed",
  "notes": "Shallow-cloned MDBDrums at b29e2d6 and verified 362 tracked files, 268 WAV files, 46 text annotations, and 2.01 GB total size. No training performed."
}
```

```json
{
  "run_id": "2026-07-15T20:30:00+08:00",
  "pattern": "manual-data-audit",
  "level": "L1",
  "duration_s": 180,
  "items_found": 2,
  "actions_taken": 0,
  "escalations": 0,
  "tokens_estimate": 7000,
  "outcome": "report-only",
  "notes": "Confirmed STAR train already contains TOM/CRASH/RIDE; raw E-GMD MIDI also contains those pitches, but current egmd_meta preprocessing keeps only KD/SD/HH. No code, data, checkpoint, or gate changes."
}
```

```json
{
  "run_id": "2026-07-15T01:00:00+08:00",
  "pattern": "manual-validation",
  "duration_s": 360,
  "items_found": 1,
  "actions_taken": 0,
  "escalations": 1,
  "tokens_estimate": 14000,
  "outcome": "escalated",
  "notes": "Rare-class threshold and core-competition sweeps proved TOM/CRASH/RIDE are model class-confusion errors. Existing v15 failed unchanged STAR held-out gate at macro F1 0.3551, so no product code, five-song run, training, checkpoint replacement, push, or deployment followed."
}
```

```json
{
  "run_id": "2026-07-15T00:00:00+08:00",
  "pattern": "manual-validation",
  "duration_s": 420,
  "items_found": 1,
  "actions_taken": 1,
  "escalations": 1,
  "tokens_estimate": 18000,
  "outcome": "fix-proposed",
  "notes": "Removed duplicate floating sync prefix offset and added one shared 67ms output-latency correction. Existing verifier passed; unchanged five-song gate improved to macro F1 0.4710 but remains below 0.70, so no deployment or further runtime fix was attempted."
}
```

```json
{
  "run_id": "2026-07-06T00:00:00+08:00",
  "pattern": "daily-triage",
  "level": "L1",
  "outcome": "loop-ready",
  "commands": [
    "loop-audit.cmd . --suggest",
    "loop-cost.cmd --pattern daily-triage --level L1",
    "loop-init.cmd . --pattern daily-triage --tool codex"
  ],
  "notes": "Initial audit was 10/100 L0. Final audit passed at 100/100 L3. Kept operation policy at L1 report-only."
}
```

```json
{
  "run_id": "2026-07-12T03:25:00+08:00",
  "pattern": "manual-validation",
  "level": "L1",
  "outcome": "escalated",
  "commands": [
    ".\\.venv\\Scripts\\python.exe train_six_class_candidate.py --meta validation_runs\\six_class_smoke\\star_six_class_meta.json --output-dir validation_runs\\six_class_candidate_v1",
    ".\\.venv\\Scripts\\python.exe run_six_class_validation.py --meta validation_runs\\six_class_smoke\\star_six_class_meta.json --model validation_runs\\six_class_candidate_v1\\six_class_candidate_v1.pth --output-dir validation_runs\\six_class_candidate_v1\\heldout_validation"
  ],
  "notes": "Candidate-v1 completed the fixed 144-window head-only schedule. Training loss fell, but held-out macro F1 was 0.0056, so the candidate was rejected and the run stopped without threshold tuning, Round5 use, promotion, replacement, or deletion."
}
```

```json
{
  "run_id": "2026-07-12T03:05:00+08:00",
  "pattern": "manual-validation",
  "level": "L1",
  "outcome": "escalated",
  "commands": [
    ".\\.venv\\Scripts\\python.exe run_six_class_validation.py --self-check",
    ".\\.venv\\Scripts\\python.exe run_six_class_validation.py --meta validation_runs\\six_class_smoke\\star_six_class_meta.json --model validation_runs\\six_class_smoke\\six_class_smoke_candidate.pth --output-dir validation_runs\\six_class_smoke\\heldout_baseline"
  ],
  "notes": "Built a six-class STAR test gate and recorded the expected smoke baseline failure: macro F1 0.0332. Stopped before formal training; no threshold tuning, test_real_audio use, candidate promotion, checkpoint replacement, or deletion occurred."
}
```

```json
{
  "run_id": "2026-07-12T02:45:00+08:00",
  "pattern": "manual-validation",
  "level": "L1",
  "outcome": "fix-proposed",
  "commands": [
    ".\\.venv\\Scripts\\python.exe preprocess_star.py --label-scheme six-class --output validation_runs\\six_class_smoke\\star_six_class_meta.json",
    ".\\.venv\\Scripts\\python.exe run_six_class_smoke.py --meta validation_runs\\six_class_smoke\\star_six_class_meta.json --output-dir validation_runs\\six_class_smoke",
    "accepted three-class blind, hard, and six single-clip Round4 gates",
    "loop-audit.cmd . --suggest"
  ],
  "notes": "Created an isolated six-class STAR smoke path. Smoke candidate passed metadata coverage, one update, reload, and [1,688,6] shape checks. Existing three-class gates remained green; no held-out real-audio input was read by the six-class path."
}
```

```json
{
  "run_id": "2026-07-12T02:20:42+08:00",
  "pattern": "manual-validation",
  "level": "L1",
  "outcome": "report-only",
  "commands": [
    "Round5 held-out raw-event to MIDI pitch audit (read-only)",
    "E-GMD/STAR label coverage audit (read-only)",
    "loop-audit.cmd . --suggest"
  ],
  "notes": "Confirmed the dominant held-out HH error is unsupported Ride/Crash/Tom proxying, not a threshold/NMS defect. STAR provides labels for a bounded six-class next path; no code, training, checkpoint replacement, push, or deletion was performed."
}
```

```json
{
  "run_id": "2026-07-10T00:00:00+08:00",
  "pattern": "manual-validation",
  "level": "L1",
  "outcome": "report-only",
  "commands": [
    ".\\.venv\\Scripts\\python.exe verify_current_solution.py --output-dir validation_runs\\current_solution_verification_20260710_recheck",
    "loop-audit.cmd . --suggest"
  ],
  "notes": "Current accepted verifier passed blind raw 5/5, notation 5/5, hard 4/4, and Round4 30/30 plus 6/6."
}
```

```json
{
  "run_id": "2026-07-10T00:00:00+08:00",
  "pattern": "manual-validation",
  "level": "L1",
  "outcome": "report-only",
  "commands": [
    ".\\.venv\\Scripts\\python.exe preprocess_egmd.py --self-check",
    ".\\.venv\\Scripts\\python.exe run_egmd_round4_validation.py --self-check",
    "loop-audit.cmd . --suggest",
    "loop-cost.cmd --pattern daily-triage --level L1"
  ],
  "notes": "Confirmed E-GMD pitch 22 and 26 are already in the shared HH mapping; no model or runtime change."
}
```

```json
{
  "run_id": "2026-07-15T21:15:00+08:00",
  "pattern": "manual-d4d-existing-data",
  "level": "human-approved",
  "duration_s": 1800,
  "items_found": 2,
  "actions_taken": 4,
  "escalations": 0,
  "tokens_estimate": 22000,
  "outcome": "fix-proposed",
  "notes": "Added six-class E-GMD mapping, built non-destructive rare/combined metadata, fixed exact D4R resume, trained one equal-budget D4D candidate, and recorded mixed/raw 0.4601/0.4692. Commercial gate remains FAIL."
}
```

```json
{
  "run_id": "2026-07-15T22:00:00+08:00",
  "pattern": "manual-d4s-source-balance",
  "level": "human-approved",
  "duration_s": 1800,
  "items_found": 1,
  "actions_taken": 3,
  "escalations": 0,
  "tokens_estimate": 22000,
  "outcome": "fix-proposed",
  "notes": "Added opt-in 50/50 STAR-EGMD rare scheduling, trained one equal-budget candidate, and rejected it because mixed STAR fell to 0.4594 despite raw rising to 0.4716. Commercial gate remains FAIL."
}
```

```json
{
  "run_id": "2026-07-16T00:50:00+08:00",
  "pattern": "manual-d5c-mdb-hard-negative",
  "level": "human-approved",
  "duration_s": 2400,
  "items_found": 1,
  "actions_taken": 4,
  "escalations": 0,
  "tokens_estimate": 20000,
  "outcome": "rejected",
  "notes": "Added opt-in MDB full-mix hard-negative scheduling, trained one equal-budget five-epoch candidate, and rejected it because mixed/raw/MDB were 0.4503/0.4570/0.4390 while HH-TOM-CRASH false positives increased from 697 to 790. Product and fixed-five gates were untouched."
}
```

```json
{
  "run_id": "2026-07-16T15:10:00+08:00",
  "pattern": "manual-d6-star-original-mix",
  "level": "human-approved",
  "duration_s": 5400,
  "items_found": 1,
  "actions_taken": 5,
  "escalations": 0,
  "tokens_estimate": 30000,
  "outcome": "rejected",
  "notes": "Added opt-in STAR original_mix metadata, measured the locked D4D real-mix baseline, completed one equal-budget five-epoch restart after an external terminal interruption, and rejected D6 because mixed/raw/original_mix/MDB were 0.4282/0.4240/0.3961/0.4185. Product and fixed-five gates were untouched."
}
```

```json
{
  "run_id": "2026-07-22T00:00:00+08:00",
  "pattern": "manual-d72-d73-sd-kd-decision",
  "level": "human-approved",
  "duration_s": 900,
  "items_found": 1,
  "actions_taken": 4,
  "escalations": 0,
  "tokens_estimate": 12000,
  "outcome": "rejected",
  "notes": "Read-only D72 compared existing fixed 48-window D61 and D70+D64 event reports. SD TP fell by 12 while FP and FN rose by 7 and 12, so D73 SD-vs-KD training was stopped; no checkpoint, test, threshold, or product inference change."
}
```

```json
{
  "run_id": "2026-07-22T00:00:00+08:00",
  "pattern": "manual-d74-d75-crash-kd-feasibility",
  "level": "human-approved",
  "duration_s": 1200,
  "items_found": 1,
  "actions_taken": 5,
  "escalations": 0,
  "tokens_estimate": 16000,
  "outcome": "fix-proposed",
  "notes": "Read-only D74 found KD as 60 of 102 fixed-window CRASH false-negative alternatives. D75 found 6,169 centered CRASH+KD train candidates and all existing source quotas sufficient. No schedule, training, checkpoint, test, threshold, data, or product inference change."
}
```

```json
{
  "run_id": "2026-07-22T00:00:00+08:00",
  "pattern": "manual-d76-d77-crash-kd-candidate",
  "level": "human-approved",
  "duration_s": 1500,
  "items_found": 1,
  "actions_taken": 8,
  "escalations": 0,
  "tokens_estimate": 30000,
  "outcome": "fix-proposed",
  "notes": "Added opt-in CRASH-vs-KD scheduling, completed a fresh five-epoch retry after the first attempt hit the executor timeout, and accepted D76+D64 as the new research baseline at fixed macro F1 0.5386 over D67 0.5356. Full six-class release remains rejected; no test, threshold, product checkpoint, or deployment change."
}
```

```json
{
  "run_id": "2026-07-22T00:00:00+08:00",
  "pattern": "manual-d78-d77-crash-residual-audit",
  "level": "human-approved",
  "duration_s": 900,
  "items_found": 1,
  "actions_taken": 6,
  "escalations": 0,
  "tokens_estimate": 12000,
  "outcome": "report-only",
  "notes": "Read-only D78 reran the fixed 48-window CRASH FN audit for D77 and compared it with D67. CRASH FP fell by 40 but TP fell by 7 and FN rose by 7; KD remained the strict-majority residual alternative at 62/109. The repeated KD route was stopped; no training, data, checkpoint, threshold, test, or product inference change."
}
```

```json
{
  "run_id": "2026-07-22T00:00:00+08:00",
  "pattern": "manual-d79-d77-hh-residual-audit",
  "level": "human-approved",
  "duration_s": 900,
  "items_found": 1,
  "actions_taken": 5,
  "escalations": 0,
  "tokens_estimate": 12000,
  "outcome": "report-only",
  "notes": "Read-only D79 audited fixed D77 HH errors. HH FN alternatives were split across KD 32, SD 30, TOM 12, CRASH 8, and RIDE 7; no strict-majority root cause exists. D80 feasibility and D81 training were not created; no data, checkpoint, threshold, test, or product inference change."
}
```

```json
{
  "run_id": "2026-07-26T13:05:57+08:00",
  "pattern": "daily-triage",
  "level": "L1",
  "duration_s": 180,
  "items_found": 2,
  "actions_taken": 0,
  "escalations": 0,
  "tokens_estimate": 7000,
  "outcome": "report-only",
  "notes": "D89 completed as research-only: fixed STAR macro F1 0.5545, release gate still failed. The only current unchecked task is D80 workspace cleanup; dirty worktree and protected data remain untouched. Next action is a read-only cleanup inventory, with human approval required before deletion."
}
```

```json
{
  "run_id": "2026-07-26T17:18:57+08:00",
  "pattern": "daily-triage",
  "level": "L1",
  "duration_s": 900,
  "items_found": 3,
  "actions_taken": 0,
  "escalations": 1,
  "tokens_estimate": 12000,
  "outcome": "escalated",
  "notes": "D80 read-only inventory completed. Candidate cleanup is limited to __MACOSX (0.000332 GiB), __pycache__ (0.000635 GiB), and D47 smoke input/output (~0.0645 GiB), with D47 audit retained. D48/D52/D53 and D27 audio remain required by D54 or hard links. Deletion awaits explicit human approval."
}
```

```json
{
  "run_id": "2026-07-31T03:10:42+08:00",
  "pattern": "human-approved-d110",
  "level": "human-approved",
  "duration_s": 720,
  "items_found": 3,
  "actions_taken": 5,
  "escalations": 0,
  "tokens_estimate": 18000,
  "outcome": "report-only",
  "notes": "D110 built a same-size 168-window full-coverage ENST schedule covering 97/97 tracks, including 94 six-class positive tracks and 3 cowbell-only NEG tracks. Window targets and D89 no-step gradient smoke passed, but four train tracks had residual offsets from -0.3715s to -0.5108s and already occupied seven D108 rows. D111 remains blocked; no optimizer step, checkpoint, sealed validation/test read, training, push, merge, or deletion occurred."
}
```

```json
{
  "run_id": "2026-07-31T04:00:00+08:00",
  "pattern": "human-approved-d110a-d110b",
  "level": "human-approved",
  "duration_s": 600,
  "items_found": 2,
  "actions_taken": 6,
  "escalations": 0,
  "tokens_estimate": 14000,
  "outcome": "report-only",
  "notes": "D110A adjudicated all four apparent ENST offsets as periodic-correlation aliases: local offsets were unstable and shifting the same event denominator improved support by at most 0.0104. A first D110B run exposed and safely preserved an out-of-bounds denominator bug; the corrected immutable v2 passed 97/97 track coverage, 168 windows, window targets, alignment gates, and D89 no-step gradients. ready_for_d111=true, but no training, optimizer step, checkpoint, sealed validation/test read, push, merge, or deletion occurred."
}
```

```json
{
  "run_id": "2026-07-31T10:15:00+08:00",
  "pattern": "human-approved-d111",
  "level": "human-approved",
  "duration_s": 480,
  "items_found": 1,
  "actions_taken": 7,
  "escalations": 0,
  "tokens_estimate": 16000,
  "outcome": "fix-proposed",
  "notes": "D111 reused the existing fused-LoRA trainer with one generic fixed-extra-schedule input. D89 reproduced at D56 macro 0.5545; the only 2968-window, one-epoch run reached 0.5526, improving 0.0037 over D108 but remaining 0.0019 below D89 with KD, TOM, CRASH, and RIDE regressions. Promotion failed, ENST validation was not read, no retry occurred, and the full current-solution verifier passed. D89 and the product checkpoint remain unchanged."
}
```

```json
{
  "run_id": "2026-07-31T10:45:00+08:00",
  "pattern": "human-approved-d112",
  "level": "human-approved",
  "duration_s": 240,
  "items_found": 1,
  "actions_taken": 5,
  "escalations": 0,
  "tokens_estimate": 10000,
  "outcome": "report-only",
  "notes": "D112 reused the D109 evaluator on the byte-identical 48-window ENST validation selection. D111 fell from D89 ENST macro 0.0535 to 0.0428 (-0.0107) while D56 was also -0.0019; diagnosis is candidate_did_not_improve_enst, not domain tradeoff. Teacher distillation, extra epochs, and same-recipe retraining remain blocked. No training, optimizer, checkpoint, sealed test, product, push, merge, or deletion action occurred."
}
```

```json
{
  "run_id": "2026-07-31T11:15:00+08:00",
  "pattern": "human-approved-d113",
  "level": "human-approved",
  "duration_s": 420,
  "items_found": 1,
  "actions_taken": 5,
  "escalations": 0,
  "tokens_estimate": 12000,
  "outcome": "report-only",
  "notes": "D113 compared D89 and D111 on the byte-identical 48-window ENST validation selection. All TP/FP/FN counts reproduce D112. D111 introduced 43 event-level errors across 10 error-type/class groups; the largest was 12 RIDE false positives (27.91%), so no strict-majority root cause exists. It removed 112 old false positives but added 32 elsewhere, and added 11 false negatives while recovering one. ready_for_d114_proposal=false; no training, threshold change, sealed test read, checkpoint, product, push, merge, or deletion occurred."
}
```

```json
{
  "run_id": "2026-07-31T11:23:31+08:00",
  "pattern": "human-requested-training-diagnosis",
  "level": "L1",
  "duration_s": 240,
  "items_found": 3,
  "actions_taken": 0,
  "escalations": 0,
  "tokens_estimate": 7000,
  "outcome": "report-only",
  "notes": "Read-only diagnosis found that D99, D104, and D111 each compressed the added domain to 168 of 2968 windows (5.66%) for one epoch, while freezing the entire DCNN/TCN/Conformer and training only 560 onset-head LoRA parameters. The trainer records only mixed mean loss, not per-domain learnability. No more data acquisition is justified before a tiny-set overfit gate proves the current pipeline can learn existing real-song and ENST examples. No code, model, checkpoint, data, threshold, push, merge, or deletion changed."
}
```

```json
{
  "run_id": "2026-07-31T11:38:34+08:00",
  "pattern": "human-approved-d114-tiny-overfit",
  "level": "human-approved",
  "duration_s": 720,
  "items_found": 1,
  "actions_taken": 6,
  "escalations": 0,
  "tokens_estimate": 16000,
  "outcome": "report-only",
  "notes": "D114 ran the only authorized 200-step tiny-set learnability diagnostic from D89 on 14 corrected real-song train plus 14 ENST drummer_1 train windows, with no replay, validation/test read, or checkpoint. Combined/real/ENST loss fell from 0.71607/0.79174/0.64040 to 0.22473/0.19234/0.25712, but train Macro F1 reached only 0.30568 and TOM/CRASH/RIDE 0.26190/0.07179/0.03243, failing the predeclared 0.90 macro and 0.80 per-class gate. current_lora_can_learn_tiny_set=false; ratio training, more data, extra steps, and automatic unfreezing remain blocked. Full current-solution verification passed; D89 and product behavior are unchanged."
}
```

```json
{
  "run_id": "2026-08-03T17:51:41+08:00",
  "pattern": "human-approved-d116-drumsep-preparation",
  "level": "human-approved",
  "duration_s": 600,
  "items_found": 5,
  "actions_taken": 8,
  "escalations": 3,
  "tokens_estimate": 12000,
  "outcome": "complete-data-preparation-no-training",
  "notes": "D116 created a new five-song D103 hard-link input set and used the D48-compatible MDX23C checkpoint on GPU with batch=1, no TTA, and no LoRA. Inference completed in 105.78s and produced 30/30 44.1kHz stereo stems (1.865 GiB). The audit passed against the same librosa 44.1kHz decode used by inference; MP3 metadata duration differences of 0.2528-0.5881s were documented and the initial metadata-only rejected audit was preserved. The new drumsep-mix manifest and 5/5 existing-preprocessor consumer smoke passed. No MIDI event changed, no sealed gate/test was read, no model/checkpoint/decoder changed, and no training, push, merge, or deletion of project data occurred."
}
```

```json
{
  "run_id": "2026-08-03T18:00:00+08:00",
  "pattern": "human-approved-d117-physical-alignment-audit",
  "level": "human-approved",
  "duration_s": 900,
  "items_found": 5,
  "actions_taken": 8,
  "escalations": 1,
  "tokens_estimate": 16000,
  "outcome": "report-only",
  "notes": "D117 read only the D103 reference events and the D116 six-stem output. It created a new high-resolution 64-sample-hop same-class stem onset evidence pack: 4876 event measurements, per-song/per-class summaries, and 30 bounded review clips. Every event had a nearest same-class local peak within 25ms, with class median absolute deltas of 3.662-5.278ms. This only rules out gross timeline misalignment; it is not human label verification or a model-quality result. No MIDI event, split, model, checkpoint, decoder, threshold, gate/test, or training state changed."
}
```
