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
