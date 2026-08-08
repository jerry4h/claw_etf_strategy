---
name: weekly-refresh
description: Execute the weekly data refresh pipeline for the ETF rotation strategy. Updates NAV data, tushare cache, national team share tracking, regenerates the dashboard, runs 14-item pre-push gate checks, and commits+pushes if all pass. Use when the user asks to refresh data, update the dashboard, or run the weekly pipeline.
---

# Weekly Data Refresh

## When to Use

- User says "刷新数据" / "周度刷新" / "更新看板" / "refresh"
- Every Saturday after Friday market close
- When user wants latest NAV/dashboard on GitHub Pages

## Execution

```bash
cd /home/ubuntu/claw_etf_strategy
.venv/bin/python scripts/weekly_refresh.py
```

## What It Does (6 steps)

1. **NAV 更新**: Pulls latest weekly close from tushare → appends to `data/all_etfs_nav_latest.csv`
2. **日频 cache 增量**: Appends new daily OHLC+amount rows to `data/experiments/tushare_cache/fund_daily_*.csv`
3. **主力份额追踪** (soft failure): Updates `data/national_team/fund_share/` — failure doesn't block pipeline
4. **数据校验**: NaN check, date monotonicity, no >20% tail jumps
5. **看板生成 + 门禁**: `gen_dashboard.py` → `check_dashboard.py` (14 items, FAIL = abort)
6. **git commit + push**: Only if steps 1-5 all pass; no partial commits

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success (or nothing to do) | None |
| 2 | Data source failure (token expired) | Update .env TUSHARE_TOKEN |
| 3 | Concurrent lock | Wait and retry |
| 4 | Data quality anomaly | Inspect data manually |
| 5 | Dashboard generation/gate failure | Fix template then retry |
| 6 | Git push failure | Local commit preserved, will auto-push next run |

## Dry Run Mode

```bash
.venv/bin/python scripts/weekly_refresh.py --dry-run
```
Runs full pipeline but skips git commit/push at the end. Use to verify before real execution.

## Key Constraints (from project rules)

- Token never appears in logs (masked as `***TOKEN***`)
- Data files in `data/national_team/` are gitignored (sensitive)
- `check_dashboard.py` must PASS all 14 items before commit is allowed
- Idempotent: safe to run multiple times
