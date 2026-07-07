# loop-constraints.md

These rules are binding for every loop run.

## Report-only L1

- L1 daily-triage may inspect, summarize, and update loop state files.
- L1 must not auto-fix code, train models, open PRs, push, merge, deploy, or delete files.

## Protected Paths

- Never delete or overwrite `*.pth`, `audio/`, `e-gmd-v1.0.0/`, `STAR_Drums_full/`, `processed_data/`, `annotations/`, `annotation_xml/`, `blind_user_tests*/`, or `validation_runs/` without human approval.
- Never edit `.env`, secrets, credentials, or external config files.
- Treat `.codex/` skill and agent changes as loop infrastructure changes; verify with `loop-audit`.

## Model Safety

- Do not replace `mixed_formal_kick375_snare18_hh12_candidate.pth`.
- New checkpoints must be candidate files until all gates pass.
- Do not use path-based model routing, per-file expected answers, or file-name special cases.

## Required Gates

- Documentation-only loop changes: run `loop-audit.cmd . --suggest`.
- Runtime/model/transcription changes: run `.\.venv\Scripts\python.exe verify_current_solution.py`.
- If a gate fails, stop and record the blocker.

## Communication

- Explain intended edits before editing.
- Report exact commands run and whether they passed.

