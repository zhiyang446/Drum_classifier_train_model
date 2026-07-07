# docs/safety.md

## Safety Policy

This project uses loop engineering only for L1 report-only triage unless a human explicitly approves a higher level.

## Denylist

- Checkpoints: `*.pth`
- Large/source datasets: `audio/`, `e-gmd-v1.0.0/`, `STAR_Drums_full/`
- Ground truth and annotations: `annotations/`, `annotation_xml/`, `processed_data/`
- Evidence: `validation_runs/`, `blind_user_tests*/`
- Secrets: `.env`, credentials, tokens

## Human Approval Required

- Training or fine-tuning.
- Deleting generated or source artifacts.
- Installing dependencies or downloading data.
- Changing accepted checkpoint behavior.
- Git push, PR creation, merge, or deployment.

## Acceptance Gates

- Loop infrastructure: `loop-audit.cmd . --suggest`
- Cost review: `loop-cost.cmd --pattern daily-triage --level L1`
- Current ADT behavior: `.\.venv\Scripts\python.exe verify_current_solution.py`

