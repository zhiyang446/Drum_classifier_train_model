# loop-budget.md

## Daily Limits

| Loop | Max runs/day | Max tokens/day | Max sub-agents/run |
|------|--------------|----------------|--------------------|
| daily-triage | 2 | 100k | 0 |

## Cost Baseline

`loop-cost.cmd --pattern daily-triage --level L1` reported:

- no-op: `60k/day`
- full triage: `600k/day`
- action every run: `2.4M/day`
- realistic blend at 12 runs/day: `276k/day`

Therefore this project caps cadence at 1-2 manual/report-only runs per day.

## Kill Switch

If `STATE.md` contains `loop-pause-all`, stop immediately and only report that the loop is paused.

## On Budget Exceed

1. Stop new checks.
2. Append the budget event to `loop-run-log.md`.
3. Ask for human direction before continuing.

