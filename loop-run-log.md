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
