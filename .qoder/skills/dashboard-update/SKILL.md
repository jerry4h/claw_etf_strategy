---
name: dashboard-update
description: Modify, preview, and validate the strategy dashboard (index.html). Handles template changes in gen_dashboard.py, local preview with --preview flag, multi-resolution testing, and 14-item pre-push gate validation. Use when the user asks to change the dashboard, fix visual issues, add new dashboard sections, or troubleshoot display problems.
---

# Dashboard Update Workflow

## When to Use

- User reports visual bugs (overflow, missing charts, layout issues)
- Adding new dashboard sections or modifying existing ones
- Fixing CSS/JS in the dashboard template
- Any "看板"/"网页"/"显示" related requests

## Key Architecture

```
scripts/gen_dashboard.py (Python template, ~500 lines)
  → generates index.html (single-file, embedded JSON data)
  → generates dashboard/data.json (structured backup)
```

- Chart.js 4.4.7 via CDN (jsdelivr)
- Data embedded as `const DATA = {...}` in `<script>`
- GitHub Pages deploys from main branch root

## Modification Workflow

```bash
# 1. Edit template
vi scripts/gen_dashboard.py

# 2. Generate + local preview
python scripts/gen_dashboard.py --preview
# Opens http://localhost:8000, Ctrl+C to stop

# 3. Multi-resolution check
python scripts/test_dashboard_responsive.py
# Outputs checklist for 5 viewports (375px ~ 1920px)

# 4. Run gate validation (MUST PASS before commit)
python scripts/check_dashboard.py
# 14 items, exit 0 = safe to commit

# 5. Commit + push
git add scripts/gen_dashboard.py index.html dashboard/data.json
git commit -m "fix/feat(dashboard): ..."
git push
```

## 14-Item Gate Checklist (check_dashboard.py)

| # | Check | Fail Condition |
|---|-------|----------------|
| 1 | viewport meta | Missing `<meta name="viewport">` |
| 2 | Chart.js CDN | No chart.js script tag |
| 3 | CDN fallback | No `typeof Chart` check |
| 4 | Three main canvases | navChart/defChart/annualChart missing |
| 5 | National team section | ntGridCharts/nt-section missing |
| 6 | No fixed-column grid | `repeat(2,1fr)` without auto-fit/minmax |
| 7 | Data integrity | sharpe≤0 or nav points<100 |
| 8 | JS syntax | `node --check` fails |
| 9 | File size | index.html > 500KB |
| 10 | Chart data present | nav/drawdown/defense arrays < 50 points |
| 11 | Chart init code | Missing `new Chart(...)` for main charts |
| 12 | National team data | share_trends empty when available=true |
| 13 | No hidden sections | Critical containers with display:none |
| 14 | minmax ≤ 300px | Any `minmax(Xpx,...)` where X > 300 |

## CSS Responsive Constraints (from project rules)

- All layout grids: `repeat(auto-fit, minmax(280px, 1fr))` — auto columns based on width
- Never use fixed `repeat(2,1fr)` or `repeat(3,1fr)` — will overflow on mobile
- `.panel { max-width:100%; overflow:hidden; }` — prevent child overflow
- `.chart-stack { min-height:480px; max-width:100%; }` — prevent chart collapse
- Body: `overflow-x:hidden` — ultimate overflow guard
- Mobile breakpoints: 900px (tablet) and 600px (phone)
- Minimum supported viewport: 375px (iPhone SE)

## Common Issues & Fixes

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Charts disappear on narrow screen | Parent flex container collapses to 0 height | Add `min-height` to `.chart-stack` and children |
| Grid columns overflow right | `minmax` value > available width | Lower minmax to ≤280px |
| Tooltip flickers | Missing `interaction` config on Chart.js | Add `{mode:'nearest', axis:'x', intersect:false}` + `tooltip.animation.duration=0` |
| Mobile layout still 2-column | Missing viewport meta tag | Add `<meta name="viewport" content="width=device-width, initial-scale=1">` |

## Data Flow (don't modify _build_data logic casually)

```
run_backtest(v4_5_pvd config) → metrics/nav/holdings/defense/annual
build_position_model() → national_team Layer C overview + share trends
_html_template() → embeds both into single HTML
```
