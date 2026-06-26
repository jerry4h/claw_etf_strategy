#!/usr/bin/env python3
"""Apply PSS replacement changes to src/robustness.py and scripts/run_robustness.py."""
import sys
sys.path.insert(0, '/home/ubuntu/claw_etf_strategy')

# ── Part 1: src/robustness.py ──
robustness_path = '/home/ubuntu/claw_etf_strategy/src/robustness.py'
with open(robustness_path, 'r') as f:
    content = f.read()

changes_applied = 0

# 3b: Update evaluate_robustness docstring
old_ds = '    """完整鲁棒性评估 (v4: +Phase 6 full_grid)。\n\n    1. 运行基准回测 → 获取 Sharpe, skew, kurtosis\n    2. 计算 DSR\n    3. 运行 MC 生存率测试 (v2 收紧标准)\n    4. 可选：运行 OAT 多级敏感度分析 (v2 新增)\n    5. 运行基准相对胜率 Walk-Forward\n    6. 可选：运行 PBO 概率过拟合 (v3 新增)\n    7. 可选：运行 SPS 起点敏感性 (v3 新增)\n    8. 可选：运行 Phase 6 全参数网格评估 (v4 新增)\n    9. 汇总为 RobustnessResult'
new_ds = '    """完整鲁棒性评估 (v4: PSS 替代 PBO + Phase 6 full_grid)。\n\n    1. 运行基准回测 → 获取 Sharpe, skew, kurtosis\n    2. 计算 DSR\n    3. 运行 MC 生存率测试 (v2 收紧标准)\n    4. 计算 PSS 参数稳定性评分 (v4，从 MC 数据零成本计算)\n    5. 可选：运行 OAT 多级敏感度分析 (v2 新增)\n    6. 运行基准相对胜率 Walk-Forward\n    7. 可选：运行 SPS 起点敏感性 (v3 新增)\n    8. 可选：运行 Phase 6 全参数网格评估 (v4 新增)\n    9. 汇总为 RobustnessResult'
if old_ds in content:
    content = content.replace(old_ds, new_ds)
    changes_applied += 1
    print("✓ Updated evaluate_robustness docstring")
else:
    print("⚠ evaluate_robustness docstring not found (may already be updated)")

# 3c: Remove pbo docstring args
old_args = '        oat: 是否运行 OAT 多级敏感度 (v2 新增)\n        pbo: 是否运行 PBO 计算 (v3 新增)\n        pbo_splits: PBO 子段数 (默认 16)\n        pbo_resamples: PBO 重采样次数 (默认 1000)\n        sps: 是否运行 SPS 起点敏感性 (v3 新增)'
new_args = '        oat: 是否运行 OAT 多级敏感度 (v2 新增)\n        sps: 是否运行 SPS 起点敏感性 (v3 新增)'
if old_args in content:
    content = content.replace(old_args, new_args)
    changes_applied += 1
    print("✓ Removed pbo docstring args")
else:
    print("⚠ pbo docstring args not found (may already be updated)")

# 3d: Remove PBO computation block + add PSS
old_pbo_block = """    # 6. PBO (v3 新增)
    pbo_value = None
    pbo_details = None
    if pbo:
        pbo_value, pbo_details = compute_pbo_simplified(
            config_path, n_splits=pbo_splits, n_resamples=pbo_resamples
        )

    # 7. SPS (v3 新增)"""
new_pss_block = """    # 4. PSS 参数稳定性评分 (v4，从 MC 数据零成本计算)
    pss = compute_pss(mc_details)

    # 5. OAT 多级敏感度 (v2 新增)
    oat_result = None
    if oat:
        oat_result = run_oat_sensitivity(config_path, perturbation=perturbation, n_jobs=n_jobs)

    # 6. WF 基准相对胜率
    wf_rate, wf_details = compute_benchmark_relative_win_rate(config_path, n_windows=n_wf_windows)

    # 7. SPS (v3 新增)"""
if old_pbo_block in content:
    content = content.replace(old_pbo_block, new_pss_block)
    changes_applied += 1
    print("✓ Replaced PBO block with PSS + moved OAT/WF after PSS")
else:
    print("⚠ PBO block not found")

# Now remove old OAT and WF sections that were before the PBO block
# The old structure was:
#   # 4. OAT (v2)
#   # 5. WF
#   # 6. PBO
#   # 7. SPS
# We already replaced #6/#7. Now remove the old #4/#5 (OAT, WF) which are now duplicated
old_oat_wf = """    # 4. OAT 多级敏感度 (v2 新增)
    oat_result = None
    if oat:
        oat_result = run_oat_sensitivity(config_path, perturbation=perturbation, n_jobs=n_jobs)

    # 5. WF 基准相对胜率
    wf_rate, wf_details = compute_benchmark_relative_win_rate(config_path, n_windows=n_wf_windows)

    # 4. PSS"""
if old_oat_wf in content:
    content = content.replace(old_oat_wf, '    # 4. PSS')
    changes_applied += 1
    print("✓ Removed duplicate OAT/WF (now after PSS)")
else:
    print("⚠ Old OAT/WF block not found (structure may differ)")

# 3e: Renumber SPS from #8 to #8, full_grid from #8 to #9
content = content.replace('    # 8. Phase 6 全参数网格 (v4 新增)', '    # 8. Phase 6 全参数网格 (v4 新增)')
content = content.replace('    # 9. 汇总', '    # 9. 汇总')

# 3f: Change RobustnessResult construction: pbo -> pss, remove pbo_details
old_result = """    result_details = {
        'mc_runs': mc_details,
        'wf_windows': wf_details,
        'dsr_debug': {
            'sharpe': sharpe,
            'n_trials': n_trials,
            'n_obs': n_obs,
            'skew': skew,
            'kurtosis': kurtosis,
        },
    }
    if pbo_details is not None:
        result_details['pbo_details'] = pbo_details
    if sps_details is not None and not sps_details.empty:
        result_details['sps_details'] = sps_details.to_dict(orient='records')

    return RobustnessResult(
        dsr=dsr,
        mc_survival_rate=mc_rate,
        benchmark_relative_win_rate=wf_rate,
        strategy_config=str(config_path),
        strategy_metrics=strategy_metrics,
        oat_sensitivity=oat_result,
        pbo=pbo_value,
        starting_point_sensitivity=sps_metrics,
        full_grid=grid_result,
        details=result_details,
    )"""
new_result = """    result_details = {
        'mc_runs': mc_details,
        'wf_windows': wf_details,
        'dsr_debug': {
            'sharpe': sharpe,
            'n_trials': n_trials,
            'n_obs': n_obs,
            'skew': skew,
            'kurtosis': kurtosis,
        },
    }
    if sps_details is not None and not sps_details.empty:
        result_details['sps_details'] = sps_details.to_dict(orient='records')

    return RobustnessResult(
        dsr=dsr,
        mc_survival_rate=mc_rate,
        benchmark_relative_win_rate=wf_rate,
        strategy_config=str(config_path),
        strategy_metrics=strategy_metrics,
        oat_sensitivity=oat_result,
        pss=pss,
        starting_point_sensitivity=sps_metrics,
        full_grid=grid_result,
        details=result_details,
    )"""
if old_result in content:
    content = content.replace(old_result, new_result)
    changes_applied += 1
    print("✓ Updated RobustnessResult construction (pbo→pss, removed pbo_details)")
else:
    print("⚠ RobustnessResult construction not found")

# 3g: _traffic_light() - replace pbo with pss
old_light = """    elif kind == 'pbo':
        return '🟢' if value < 0.10 else ('🟡' if value < 0.30 else '🔴')"""
new_light = """    elif kind == 'pss':
        rp10 = value.get('return_p10', 0) if isinstance(value, dict) else 0
        dp90 = value.get('dd_p90', 1) if isinstance(value, dict) else 1
        if rp10 > 0.10 and dp90 < 0.15:
            return '🟢'
        elif value.get('return_p50', 0) > 0.10 and value.get('dd_p50', 1) < 0.15:
            return '🟡'
        else:
            return '🔴'"""
if old_light in content:
    content = content.replace(old_light, new_light)
    changes_applied += 1
    print("✓ Updated _traffic_light pbo→pss")
else:
    print("⚠ _traffic_light pbo not found")

# 3h: _overall_verdict() - change pbo to pss
old_overall = """    dsr = best.dsr
    mc = best.mc_survival_rate
    wf = best.benchmark_relative_win_rate
    pbo_val = best.pbo
    sps = best.starting_point_sensitivity

    # Count lights
    lights = [
        _traffic_light(dsr, 'dsr'),
        _traffic_light(mc, 'mc'),
        _traffic_light(wf, 'wf'),
    ]
    if pbo_val is not None:
        lights.append(_traffic_light(pbo_val, 'pbo'))
    if sps is not None:
        lights.append(_traffic_light(sps.get('worst_annual_return', 0), 'sps_worst'))"""
new_overall = """    dsr = best.dsr
    mc = best.mc_survival_rate
    wf = best.benchmark_relative_win_rate
    pss_val = best.pss
    sps = best.starting_point_sensitivity

    # Count lights
    lights = [
        _traffic_light(dsr, 'dsr'),
        _traffic_light(mc, 'mc'),
        _traffic_light(wf, 'wf'),
    ]
    if pss_val is not None:
        lights.append(_traffic_light(pss_val, 'pss'))
    if sps is not None:
        lights.append(_traffic_light(sps.get('worst_annual_return', 0), 'sps_worst'))"""
if old_overall in content:
    content = content.replace(old_overall, new_overall)
    changes_applied += 1
    print("✓ Updated _overall_verdict pbo→pss")
else:
    print("⚠ _overall_verdict pbo not found")

# 3i: generate_robustness_report - replace PBO section with PSS section
old_pbo_section = """    # ── PBO (v3 新增) ──
    pbo_available = any(r.pbo is not None for r in results)
    if pbo_available:
        lines.append('---')
        lines.append('')
        lines.append('## ④ PBO（概率过拟合）')
        lines.append('')
        lines.append('| 策略 | PBO | IS/OOS 秩相关 | 评级 | 解读 |')
        lines.append('|------|:---:|:-----------:|:----:|------|')
        for r, label in zip(results, labels):
            pbo_val = r.pbo
            if pbo_val is None:
                continue
            light = _traffic_light(pbo_val, 'pbo')
            rho = r.details.get('pbo_details', {}).get('rank_correlation', 0)
            if pbo_val < 0.10:
                interpret = '没有过拟合，IS/OOS 高度一致'
            elif pbo_val < 0.30:
                interpret = '轻微过拟合嫌疑'
            else:
                interpret = '过度拟合，IS 好 ≠ OOS 好'
            lines.append(f'| {label} | {pbo_val:.4f} | {rho:.4f} | {light} | {interpret} |')
        lines.append('')"""
new_pss_section = """    # ── PSS（参数稳定性评分, v4 替代 PBO）──
    pss_available = any(r.pss is not None for r in results)
    if pss_available:
        lines.append('---')
        lines.append('')
        lines.append('## ④ PSS（参数稳定性评分）')
        lines.append('')
        lines.append('| 策略 | 年化 P10/P50/P90 | DD P10/P50/P90 | Sharpe P10/P50/P90 | CV | 评级 |')
        lines.append('|------|:---:|:---:|:---:|:--:|:--:|')
        for r, label in zip(results, labels):
            pss_val = r.pss
            if pss_val is None:
                continue
            light = _traffic_light(pss_val, 'pss')
            ret_str = f"{pss_val.get('return_p10', 0)*100:.1f}%/{pss_val.get('return_p50', 0)*100:.1f}%/{pss_val.get('return_p90', 0)*100:.1f}%"
            dd_str = f"{pss_val.get('dd_p10', 0)*100:.1f}%/{pss_val.get('dd_p50', 0)*100:.1f}%/{pss_val.get('dd_p90', 0)*100:.1f}%"
            sharpe_str = f"{pss_val.get('sharpe_p10', 0):.2f}/{pss_val.get('sharpe_p50', 0):.2f}/{pss_val.get('sharpe_p90', 0):.2f}"
            cv_str = f"r:{pss_val.get('return_cv', 0):.2f} d:{pss_val.get('dd_cv', 0):.2f} s:{pss_val.get('sharpe_cv', 0):.2f}"
            lines.append(f'| {label} | {ret_str} | {dd_str} | {sharpe_str} | {cv_str} | {light} |')
        lines.append('')"""
if old_pbo_section in content:
    content = content.replace(old_pbo_section, new_pss_section)
    changes_applied += 1
    print("✓ Replaced PBO section with PSS section in report")
else:
    print("⚠ PBO section not found in report")

# 3j: Fix section numbering in SPS - change ⑤ to ⑤ (unchanged but check)
# OAT section numbering
old_oat_num = """        section_num = '⑥' if pbo_available or sps_available else '④'"""
new_oat_num = """        section_num = '⑥' if sps_available else '⑤'"""
if old_oat_num in content:
    content = content.replace(old_oat_num, new_oat_num)
    changes_applied += 1
    print("✓ Fixed OAT section numbering (removed pbo_available reference)")
else:
    print("⚠ OAT section numbering not found")

# 3k: Five-indicator table - replace ④ PBO with ④ PSS
old_five_pbo = """    # ④ PBO
    if pbo_available:
        pbo_vals = []
        for r in results:
            pv = r.pbo
            if pv is None:
                pbo_vals.append('N/A')
            else:
                pbo_vals.append(f'{pv:.4f} {_traffic_light(pv, "pbo")}')
        pbo_best = 'N/A'
        pbo_candidates = [(i, r.pbo) for i, r in enumerate(results) if r.pbo is not None]
        if pbo_candidates:
            pbo_best = labels[min(pbo_candidates, key=lambda x: x[1])[0]]
        lines.append('| **④ PBO** | ' + ' | '.join(pbo_vals) + f' | {pbo_best} |')"""
new_five_pss = """    # ④ PSS
    if pss_available:
        pss_vals = []
        for r in results:
            pv = r.pss
            if pv is None:
                pss_vals.append('N/A')
            else:
                pss_vals.append(f'{pv.get("return_p10", 0)*100:.1f}%/{pv.get("dd_p90", 0)*100:.1f}% {_traffic_light(pv, "pss")}')
        pss_best = 'N/A'
        pss_candidates = [(i, r.pss) for i, r in enumerate(results) if r.pss is not None]
        if pss_candidates:
            pss_best = labels[max(pss_candidates, key=lambda x: x[1].get('return_p10', 0))[0]]
        lines.append('| **④ PSS** | ' + ' | '.join(pss_vals) + f' | {pss_best} |')"""
if old_five_pbo in content:
    content = content.replace(old_five_pbo, new_five_pss)
    changes_applied += 1
    print("✓ Updated five-indicator table ④ PBO→PSS")
else:
    print("⚠ Five-indicator PBO not found")

# 3l: Update 综合判定 references
old_comp_ref = """        if r.pbo is not None:
            all_lights.append(_traffic_light(r.pbo, 'pbo'))"""
new_comp_ref = """        if r.pss is not None:
            all_lights.append(_traffic_light(r.pss, 'pss'))"""
if old_comp_ref in content:
    content = content.replace(old_comp_ref, new_comp_ref)
    content = content.replace(
        """        if r.pbo is not None:
            lights_list.append(_traffic_light(r.pbo, 'pbo'))""",
        """        if r.pss is not None:
            lights_list.append(_traffic_light(r.pss, 'pss'))""")
    changes_applied += 1
    print("✓ Updated 综合判定 pbo→pss references")
else:
    print("⚠ 综合判定 pbo references not found")

# 3m: Remove PBO details section from detailed data
old_pbo_details = """        # PBO 详细数据 (v3)
        pbo_details = r.details.get('pbo_details')
        if pbo_details:
            lines.append(f'**PBO 计算详情**:')
            lines.append(f'- PBO: {r.pbo:.4f}' if r.pbo is not None else '- PBO: N/A')
            lines.append(f'- IS/OOS 秩相关 ρ: {pbo_details.get("rank_correlation", 0):.4f}')
            lines.append(f'- 重采样次数: {pbo_details.get("n_resamples", 0)}')
            lines.append(f'- 子段数: {pbo_details.get("n_splits", 0)}')
            lines.append(f'- IS Sharpe 均值: {pbo_details.get("is_sharpe_mean", 0):.4f}')
            lines.append(f'- OOS Sharpe 均值: {pbo_details.get("oos_sharpe_mean", 0):.4f}')
            lines.append('')

        # SPS 详细数据 (v3)"""
new_pss_details = """        # PSS 详细数据 (v4)
        pss_val = r.pss
        if pss_val:
            lines.append(f'**PSS 参数稳定性详情**:')
            lines.append(f'- 年化收益 P10/P50/P90: {pss_val.get("return_p10", 0)*100:.2f}% / {pss_val.get("return_p50", 0)*100:.2f}% / {pss_val.get("return_p90", 0)*100:.2f}%')
            lines.append(f'- 最大回撤 P10/P50/P90: {pss_val.get("dd_p10", 0)*100:.2f}% / {pss_val.get("dd_p50", 0)*100:.2f}% / {pss_val.get("dd_p90", 0)*100:.2f}%')
            lines.append(f'- Sharpe P10/P50/P90: {pss_val.get("sharpe_p10", 0):.4f} / {pss_val.get("sharpe_p50", 0):.4f} / {pss_val.get("sharpe_p90", 0):.4f}')
            lines.append(f'- CV (收益/回撤/Sharpe): {pss_val.get("return_cv", 0):.4f} / {pss_val.get("dd_cv", 0):.4f} / {pss_val.get("sharpe_cv", 0):.4f}')
            lines.append(f'- 样本数: {pss_val.get("n_total", 0)}')
            lines.append('')

        # SPS 详细数据 (v3)"""
if old_pbo_details in content:
    content = content.replace(old_pbo_details, new_pss_details)
    changes_applied += 1
    print("✓ Replaced PBO details with PSS details in report")
else:
    print("⚠ PBO details section not found")

# 3n: Update JSON output - pbo -> pss
content = content.replace(
    "'pbo': r.pbo,",
    "'pss': r.pss,"
)
content = content.replace(
    "'pbo_details': r.details.get('pbo_details', {}),",
    "'pss_details': r.details.get('pss_details', {}),"
)
changes_applied += 1
print("✓ Updated JSON output pbo→pss")

# Write back
with open(robustness_path, 'w') as f:
    f.write(content)
print(f"\\n✅ robustness.py: {changes_applied} changes applied")

# ── Part 2: scripts/run_robustness.py ──
run_path = '/home/ubuntu/claw_etf_strategy/scripts/run_robustness.py'
with open(run_path, 'r') as f:
    run_content = f.read()

run_changes = 0

# Remove --pbo argument
old_pbo_arg = """    parser.add_argument(
        '--pbo', action='store_true', default=False,
        help='启用 PBO 概率过拟合计算 (v3 新增)'
    )
    parser.add_argument(
        '--pbo-splits', type=int, default=16,
        help='PBO 子段数 (默认: 16)'
    )
    parser.add_argument(
        '--pbo-resamples', type=int, default=1000,
        help='PBO 重采样次数 (默认: 1000)'
    )"""
if old_pbo_arg in run_content:
    run_content = run_content.replace(old_pbo_arg, '')
    run_changes += 1
    print("✓ Removed --pbo/--pbo-splits/--pbo-resamples args")

# Update description
run_content = run_content.replace(
    "description='鲁棒性评估 — 五指标简化版 v3 (DSR / MC 生存率 / 基准相对胜率 / PBO / SPS)'",
    "description='鲁棒性评估 — 五指标简化版 v4 (DSR / PSS / MC 生存率 / 基准相对胜率 / SPS)'"
)
run_changes += 1
print("✓ Updated description")

# Remove PBO display section
old_pbo_display = """            if args.pbo and result.pbo is not None:
                print(f\"  {'⑤' if args.oat else '④'} PBO:                      {result.pbo:.4f}\")"""
if old_pbo_display in run_content:
    run_content = run_content.replace(old_pbo_display, '')
    run_changes += 1
    print("✓ Removed PBO display line")

# Add PSS display  
old_sps_display = """            if args.sps and result.starting_point_sensitivity is not None:
                sps_m = result.starting_point_sensitivity
                print(f\"  {'⑥' if args.oat and args.pbo else '⑤' if args.oat or args.pbo else '④'} SPS 最差起点:           {sps_m['worst_annual_return']*100:.2f}% ({sps_m['worst_start_date']})\")"""
new_pss_sps = """            # PSS always shown (computed from MC data, zero cost)
            if result.pss is not None:
                pss_m = result.pss
                pss_label = 'PSS 参数稳定性'
                print(f\"  {'④' if not args.oat else '⑤'} PSS 年化 P10/P50/P90:      {pss_m['return_p10']*100:.2f}% / {pss_m['return_p50']*100:.2f}% / {pss_m['return_p90']*100:.2f}%\")
                print(f\"     PSS DD P10/P50/P90:       {pss_m['dd_p10']*100:.2f}% / {pss_m['dd_p50']*100:.2f}% / {pss_m['dd_p90']*100:.2f}%\")
                print(f\"     PSS Sharpe P10/P50/P90:   {pss_m['sharpe_p10']:.4f} / {pss_m['sharpe_p50']:.4f} / {pss_m['sharpe_p90']:.4f}\")
                print(f\"     PSS CV (ret/dd/shp):      {pss_m['return_cv']:.2f} / {pss_m['dd_cv']:.2f} / {pss_m['sharpe_cv']:.2f}\")
            if args.sps and result.starting_point_sensitivity is not None:
                sps_m = result.starting_point_sensitivity
                sps_idx = '⑤' if not args.oat else '⑥'
                print(f\"  {sps_idx} SPS 最差起点:           {sps_m['worst_annual_return']*100:.2f}% ({sps_m['worst_start_date']})\")"""
if old_sps_display in run_content:
    run_content = run_content.replace(old_sps_display, new_pss_sps)
    run_changes += 1
    print("✓ Added PSS display + updated SPS numbering")
else:
    print("⚠ SPS display line not found, trying alternative pattern")

# Remove pbo from evaluate_robustness call
run_content = run_content.replace(
    """                pbo=args.pbo,
                pbo_splits=args.pbo_splits,
                pbo_resamples=args.pbo_resamples,
                sps=args.sps,""",
    """                sps=args.sps,"""
)
run_changes += 1
print("✓ Removed pbo params from evaluate_robustness() call")

# Update print header
run_content = run_content.replace(
    "    print(\"鲁棒性评估 — 五指标简化版 v3\")",
    "    print(\"鲁棒性评估 — 五指标简化版 v4\")"
)
run_changes += 1

# Update PBO print line
run_content = run_content.replace(
    "    print(f\"PBO: {'启用' if args.pbo else '关闭'}\")",
    "    print(f\"PSS: 始终运行（MC 数据零成本计算）\")"
)
run_changes += 1
print("✓ Updated header print lines")

with open(run_path, 'w') as f:
    f.write(run_content)
print(f"\\n✅ run_robustness.py: {run_changes} changes applied")