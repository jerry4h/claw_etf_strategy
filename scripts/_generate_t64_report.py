#!/usr/bin/env python3
"""Generate Phase 6 Full-Grid comparison report for T64."""
import sys, os, json, csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.robustness import (
    _generate_grid_per_param_table,
    _generate_cliff_summary,
    _generate_sharpe_heatmap_overview,
    _write_grid_csv,
    _traffic_light,
    GRID_CONTINUOUS_PARAMS,
    GRID_DISCRETE_PARAMS,
    GRID_D4_PARAMS,
    PERTURBATION_LEVELS,
    GridPointResult,
)

output_dir = Path("output/robustness_phase6")
output_dir.mkdir(parents=True, exist_ok=True)

# Load intermediate results
with open(output_dir / "baseline_intermediate.json") as f:
    baseline_raw = json.load(f)
with open(output_dir / "d4tuned_intermediate.json") as f:
    d4tuned_raw = json.load(f)

def dict_to_grid_result(d):
    return GridPointResult(
        param_name=d["param_name"],
        level=d["level"],
        actual_value=d["actual_value"],
        sharpe=d["sharpe"] if d["sharpe"] is not None else float("nan"),
        annual_return=d["annual_return"] if d["annual_return"] is not None else float("nan"),
        max_drawdown=d["max_drawdown"] if d["max_drawdown"] is not None else float("nan"),
        relative_sharpe=d["relative_sharpe"] if d["relative_sharpe"] is not None else float("nan"),
        mc_survival_rate=d["mc_survival_rate"],
    )

def rebuild_grid(raw):
    grid = {}
    for pname, glist in raw["full_grid"].items():
        grid[pname] = [dict_to_grid_result(g) for g in glist]
        grid[pname].sort(key=lambda x: x.level)
    return grid

baseline_grid = rebuild_grid(baseline_raw)
d4tuned_grid = rebuild_grid(d4tuned_raw)

def _format_grid_level(param_name, level):
    if param_name == "top_n":
        return f"n={int(level)}"
    return f"{level:+.0%}"

def generate_comparison_section(grid_results_list, labels):
    lines = []
    lines.append("### 参数对比热力表 - 并排对比")
    lines.append("")

    all_params = GRID_CONTINUOUS_PARAMS + GRID_DISCRETE_PARAMS
    d4_present = any(p in grid_results_list[0] for p in GRID_D4_PARAMS)
    if d4_present:
        all_params = list(all_params) + list(GRID_D4_PARAMS)

    for param_name in all_params:
        present = all(param_name in g for g in grid_results_list)
        if not present:
            continue
        lines.append(f"**{param_name}**")
        lines.append("")

        headers = ["扰动"]
        for lbl in labels:
            headers += [f"{lbl}-Sharpe", f"{lbl}-年化", f"{lbl}-DD", f"{lbl}-RelSharpe", f"{lbl}-MC"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + ":----:|" * len(headers) + "|")

        all_levels = sorted(set(gpr.level for g in grid_results_list for gpr in g[param_name]))
        for lvl in all_levels:
            row = [f"{lvl:+.0%}" if param_name != "top_n" else f"n={int(lvl)}"]
            for g in grid_results_list:
                matches = [gpr for gpr in g[param_name] if abs(gpr.level - lvl) < 0.001]
                if matches:
                    gpr = matches[0]
                    s = f'{gpr.sharpe:.3f}' if not np.isnan(gpr.sharpe) else 'N/A'
                    r = f'{gpr.annual_return*100:.2f}%' if not np.isnan(gpr.annual_return) else 'N/A'
                    d = f'{gpr.max_drawdown*100:.2f}%' if not np.isnan(gpr.max_drawdown) else 'N/A'
                    rs = f'{gpr.relative_sharpe:+.3f}' if not np.isnan(gpr.relative_sharpe) else 'N/A'
                    mc = f'{gpr.mc_survival_rate*100:.0f}%'
                    row += [s, r, d, rs, mc]
                else:
                    row += ['N/A', 'N/A', 'N/A', 'N/A', 'N/A']
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return lines

def generate_comparison_cliff_summary(grid_list, labels):
    lines = []
    lines.append("### 对比悬崖效应汇总")
    lines.append("")
    lines.append("| 参数 | 策略 | 悬崖位置 | Sharpe 变化 | DD 变化 | 局部MC 变化 | 严重度 |")
    lines.append("|------|:----:|:--------:|:---------:|:------:|:---------:|:-----:|")

    for grid_results, label in zip(grid_list, labels):
        for param_name, results in grid_results.items():
            if len(results) < 2:
                continue
            max_sharpe_drop = 0.0
            max_dd_jump = 0.0
            cliff_level = results[0].level
            cliff_mc_drop = 0.0
            for i in range(1, len(results)):
                prev = results[i-1]
                curr = results[i]
                if np.isnan(prev.sharpe) or np.isnan(curr.sharpe):
                    continue
                sharpe_drop = prev.sharpe - curr.sharpe
                dd_jump = curr.max_drawdown - prev.max_drawdown
                mc_drop = prev.mc_survival_rate - curr.mc_survival_rate
                if sharpe_drop > 0.15 or dd_jump > 0.03:
                    if abs(sharpe_drop) + abs(dd_jump) > abs(max_sharpe_drop) + abs(max_dd_jump):
                        max_sharpe_drop = sharpe_drop
                        max_dd_jump = dd_jump
                        cliff_level = curr.level
                        cliff_mc_drop = mc_drop
            if max_sharpe_drop > 0.15 or max_dd_jump > 0.03:
                level_str = _format_grid_level(param_name, cliff_level)
                severity = '致命' if max_sharpe_drop > 0.30 else ('高' if max_dd_jump > 0.05 else '中')
                lines.append(
                    f'| {param_name} | {label} | {level_str} | '
                    f'{max_sharpe_drop:+.3f} | {max_dd_jump*100:+.1f}pp | '
                    f'{cliff_mc_drop*100:+.0f}% | {severity} |'
                )

    if len(lines) <= 2:
        lines.append('| - | - | 未发现显著悬崖 | - | - | - | - |')
    lines.append("")
    return lines

def generate_overall_rating(baseline_raw, d4tuned_raw):
    lines = []
    lines.append("### 综合评级 - 与目标对比")
    lines.append("")
    lines.append("| 目标 | 指标 | v2.3 基线 | D4 tuned |")
    lines.append("|------|------|:---------:|:--------:|")

    targets = [
        ("高鲁棒性", "DSR > 0.95", baseline_raw["dsr"], d4tuned_raw["dsr"], lambda v: v >= 0.95),
        ("高鲁棒性", "MC 生存率 > 80%", baseline_raw["mc_survival_rate"], d4tuned_raw["mc_survival_rate"], lambda v: v >= 0.80),
        ("高鲁棒性", "WF 胜率 > 80%", baseline_raw["benchmark_relative_win_rate"], d4tuned_raw["benchmark_relative_win_rate"], lambda v: v >= 0.80),
        ("回撤 < 10%", "最大回撤", baseline_raw["strategy_metrics"]["max_drawdown"], d4tuned_raw["strategy_metrics"]["max_drawdown"], lambda v: v < 0.10),
        ("年化 > 14%", "年化收益", baseline_raw["strategy_metrics"]["annual_return"], d4tuned_raw["strategy_metrics"]["annual_return"], lambda v: v >= 0.14),
    ]

    for cat, metric, bv, dv, check in targets:
        b_ok = check(bv)
        d_ok = check(dv)

        if metric == "DSR > 0.95":
            b_str = f"{bv:.4f} {'O' if b_ok else 'X'}"
            d_str = f"{dv:.4f} {'O' if d_ok else 'X'}"
        elif metric in ("MC 生存率 > 80%", "WF 胜率 > 80%"):
            b_str = f"{bv*100:.1f}% {'O' if b_ok else 'X'}"
            d_str = f"{dv*100:.1f}% {'O' if d_ok else 'X'}"
        else:
            v_str = f"{bv*100:.2f}%"
            b_str = f"{v_str} {'O' if b_ok else 'X'}"
            v_str2 = f"{dv*100:.2f}%"
            d_str = f"{v_str2} {'O' if d_ok else 'X'}"

        lines.append(f"| {cat} | {metric} | {b_str} | {d_str} |")

    lines.append("")
    b_checks = sum(1 for _, _, bv, _, check in targets if check(bv))
    d_checks = sum(1 for _, _, _, dv, check in targets if check(dv))
    total = len(targets)
    lines.append(f"**v2.3 基线**: {b_checks}/{total} 目标达成")
    lines.append(f"**D4 tuned**: {d_checks}/{total} 目标达成")
    lines.append("")
    return lines


# BUILD REPORT
lines = []
lines.append("# Phase 6 全参数 7 级网格鲁棒性对比报告")
lines.append("")
lines.append(f"**评估日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
lines.append(f"**策略对比**: v2.3 基线 ({sum(len(v) for v in baseline_grid.values())} 格点) vs D4 tuned ({sum(len(v) for v in d4tuned_grid.values())} 格点)")
lines.append("")
lines.append("**局部 MC**: 50 次/格点 | **参数数**: 8 连续 + 1 离散 + (D4 开启时 +2)")
lines.append("")

# 1. Baseline metrics
lines.append("---")
lines.append("")
lines.append("## 1. 基准对比")
lines.append("")
labels = ["v2.3 基线", "D4 tuned"]
raws = [baseline_raw, d4tuned_raw]
lines.append("| 指标 | v2.3 基线 | D4 tuned |")
lines.append("|------|:--------:|:--------:|")
for key, name in [("annual_return", "年化收益"), ("max_drawdown", "最大回撤"), ("sharpe_ratio", "夏普比率")]:
    vals = [f'{r["strategy_metrics"][key]*100:.2f}%' if key in ("annual_return", "max_drawdown") else f'{r["strategy_metrics"][key]:.4f}' for r in raws]
    lines.append(f'| {name} | {vals[0]} | {vals[1]} |')
lines.append(f'| DSR | {raws[0]["dsr"]:.4f} | {raws[1]["dsr"]:.4f} |')
lines.append(f'| MC 生存率 | {raws[0]["mc_survival_rate"]*100:.1f}% | {raws[1]["mc_survival_rate"]*100:.1f}% |')
lines.append(f'| WF 胜率 | {raws[0]["benchmark_relative_win_rate"]*100:.1f}% | {raws[1]["benchmark_relative_win_rate"]*100:.1f}% |')
lines.append("")

# 2. Per-parameter comparison tables
lines.append("---")
lines.append("")
lines.append("## 2. 参数热力对比表")
lines.append("")
comp_lines = generate_comparison_section([baseline_grid, d4tuned_grid], labels)
lines.extend(comp_lines)

# 3. Cliff summary comparison
lines.append("---")
lines.append("")
lines.append("## 3. 悬崖效应对比")
lines.append("")
cliff_lines = generate_comparison_cliff_summary([baseline_grid, d4tuned_grid], labels)
lines.extend(cliff_lines)

# 4. Sharpe heatmap for each strategy
lines.append("---")
lines.append("")
lines.append("## 4. Sharpe 热力图概览")
lines.append("")
lines.append("### v2.3 基线")
lines.append("")
sh_b = _generate_sharpe_heatmap_overview(baseline_grid)
lines.extend(sh_b)
lines.append("### D4 tuned")
lines.append("")
sh_d = _generate_sharpe_heatmap_overview(d4tuned_grid)
lines.extend(sh_d)

# 5. Overall rating vs targets
lines.append("---")
lines.append("")
lines.append("## 5. 综合评级")
lines.append("")
rating_lines = generate_overall_rating(baseline_raw, d4tuned_raw)
lines.extend(rating_lines)

# 6. Key findings
lines.append("---")
lines.append("")
lines.append("## 6. 关键发现与建议")
lines.append("")

d4_better_params = []
baseline_better_params = []
for param_name in baseline_grid:
    if param_name in d4tuned_grid:
        b_avg_sharpe = np.mean([gpr.sharpe for gpr in baseline_grid[param_name] if not np.isnan(gpr.sharpe)])
        d_avg_sharpe = np.mean([gpr.sharpe for gpr in d4tuned_grid[param_name] if not np.isnan(gpr.sharpe)])
        diff = d_avg_sharpe - b_avg_sharpe
        if diff > 0.02:
            d4_better_params.append((param_name, diff))
        elif diff < -0.02:
            baseline_better_params.append((param_name, -diff))

if d4_better_params:
    lines.append("**D4 tuned 优势参数**:")
    for p, d in sorted(d4_better_params, key=lambda x: -x[1]):
        lines.append(f"- {p}: 平均 Sharpe 高 {d:.3f}")
if baseline_better_params:
    lines.append("**v2.3 基线优势参数**:")
    for p, d in sorted(baseline_better_params, key=lambda x: -x[1]):
        lines.append(f"- {p}: 平均 Sharpe 高 {d:.3f}")

lines.append("")
b_mc_avg = np.mean([gpr.mc_survival_rate for gl in baseline_grid.values() for gpr in gl])
d_mc_avg = np.mean([gpr.mc_survival_rate for gl in d4tuned_grid.values() for gpr in gl])
lines.append(f"**全局 MC 生存率对比**:")
lines.append(f"- v2.3 基线: {b_mc_avg*100:.1f}%")
lines.append(f"- D4 tuned: {d_mc_avg*100:.1f}%")
lines.append(f"- 差异: {(d_mc_avg - b_mc_avg)*100:+.1f}pp")
lines.append("")

b_cliffs = []
d_cliffs = []
for param_name, results in baseline_grid.items():
    for i in range(1, len(results)):
        prev, curr = results[i-1], results[i]
        if not np.isnan(prev.sharpe) and not np.isnan(curr.sharpe):
            if (prev.sharpe - curr.sharpe) > 0.15 or (curr.max_drawdown - prev.max_drawdown) > 0.03:
                b_cliffs.append(param_name)
                break
for param_name, results in d4tuned_grid.items():
    for i in range(1, len(results)):
        prev, curr = results[i-1], results[i]
        if not np.isnan(prev.sharpe) and not np.isnan(curr.sharpe):
            if (prev.sharpe - curr.sharpe) > 0.15 or (curr.max_drawdown - prev.max_drawdown) > 0.03:
                d_cliffs.append(param_name)
                break

lines.append(f"**悬崖效应**: v2.3 基线 {len(b_cliffs)} 个参数有悬崖, D4 tuned {len(d_cliffs)} 个")
if b_cliffs:
    lines.append(f"- 基线悬崖: {', '.join(b_cliffs)}")
if d_cliffs:
    lines.append(f"- D4 悬崖: {', '.join(d_cliffs)}")
lines.append("")

lines.append("**建议**:")
lines.append("")
# Define targets for overall assessment
_targets_for_goals = [
    ("高鲁棒性", "DSR > 0.95", baseline_raw["dsr"], d4tuned_raw["dsr"], lambda v: v >= 0.95),
    ("高鲁棒性", "MC 生存率 > 80%", baseline_raw["mc_survival_rate"], d4tuned_raw["mc_survival_rate"], lambda v: v >= 0.80),
    ("高鲁棒性", "WF 胜率 > 80%", baseline_raw["benchmark_relative_win_rate"], d4tuned_raw["benchmark_relative_win_rate"], lambda v: v >= 0.80),
    ("回撤 < 10%", "最大回撤", baseline_raw["strategy_metrics"]["max_drawdown"], d4tuned_raw["strategy_metrics"]["max_drawdown"], lambda v: v < 0.10),
    ("年化 > 14%", "年化收益", baseline_raw["strategy_metrics"]["annual_return"], d4tuned_raw["strategy_metrics"]["annual_return"], lambda v: v >= 0.14),
]
b_goals = sum(1 for _, _, bv, _, check in _targets_for_goals if check(bv))
d_goals = sum(1 for _, _, _, dv, check in _targets_for_goals if check(dv))

if d_goals >= b_goals and d_mc_avg > b_mc_avg:
    lines.append("- D4 tuned 整体参数鲁棒性优于基线，建议优先考虑")
elif d_goals >= b_goals:
    lines.append("- D4 tuned 目标达成率更好，但需注意参数平面稳定性")
else:
    lines.append("- v2.3 基线在目前目标下表现更优，建议保留基线为主策略")

if d_cliffs or b_cliffs:
    lines.append("- 存在悬崖效应的参数应设为硬约束，防止小幅波动导致大幅恶化")
    if "momentum_window" in d_cliffs:
        lines.append("  - D4 momentum_window 存在悬崖：考虑固定窗口或加宽缓冲区")
    if "stop_loss" in b_cliffs or "stop_loss" in d_cliffs:
        lines.append("  - Stop loss 悬崖：启用渐进出场代替硬止损")

lines.append("")

# 7. Data files
lines.append("---")
lines.append("")
lines.append("## 7. 数据文件")
lines.append("")
lines.append("- 基线 CSV: output/robustness_phase6/grid_data/v2.3_基线_grid.csv")
lines.append("- D4 CSV: output/robustness_phase6/grid_data/D4_tuned_grid.csv")
lines.append("- 结构化 JSON: output/robustness_phase6/robustness_results.json")
lines.append("")

# Write report
report = "\n".join(lines)
report_path = output_dir / "PHASE6_GRID_COMPARISON_REPORT.md"
with open(report_path, "w") as f:
    f.write(report)

# Write combined JSON
combined = {
    "strategies": [
        {"label": "v2.3 基线", "metrics": baseline_raw["strategy_metrics"], "dsr": baseline_raw["dsr"],
         "mc_survival_rate": baseline_raw["mc_survival_rate"], "wf_win_rate": baseline_raw["benchmark_relative_win_rate"],
         "grid_points": {p: [{"level": g["level"], "sharpe": g["sharpe"], "annual_return": g["annual_return"],
                              "max_drawdown": g["max_drawdown"], "relative_sharpe": g["relative_sharpe"],
                              "mc_survival_rate": g["mc_survival_rate"]} for g in glist]
                         for p, glist in baseline_raw["full_grid"].items()}},
        {"label": "D4 tuned", "metrics": d4tuned_raw["strategy_metrics"], "dsr": d4tuned_raw["dsr"],
         "mc_survival_rate": d4tuned_raw["mc_survival_rate"], "wf_win_rate": d4tuned_raw["benchmark_relative_win_rate"],
         "grid_points": {p: [{"level": g["level"], "sharpe": g["sharpe"], "annual_return": g["annual_return"],
                              "max_drawdown": g["max_drawdown"], "relative_sharpe": g["relative_sharpe"],
                              "mc_survival_rate": g["mc_survival_rate"]} for g in glist]
                         for p, glist in d4tuned_raw["full_grid"].items()}},
    ]
}

with open(output_dir / "robustness_results.json", "w") as f:
    json.dump(combined, f, indent=2, default=str, ensure_ascii=False)

# Write CSV files
_write_grid_csv(baseline_grid, output_dir, "v2.3 基线")
_write_grid_csv(d4tuned_grid, output_dir, "D4 tuned")

print(f"Report: {report_path}")
print(f"JSON:   {output_dir / 'robustness_results.json'}")
print(f"CSV:    {output_dir / 'grid_data/'}")
print("Done!")