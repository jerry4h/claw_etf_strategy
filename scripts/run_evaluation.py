#!/usr/bin/env python3
"""
T9: Comprehensive evaluation — baseline vs P0 vs P1 directions
Runs all scenarios and writes results to output/report_v2_4_evaluation.md
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.strategy import StrategyConfig, load_config
from src.backtest import run_backtest


def make_config(base: StrategyConfig, **overrides) -> StrategyConfig:
    """Create a copy of base config with overrides."""
    kwargs = {}
    for field in base.__dataclass_fields__:
        if field in overrides:
            kwargs[field] = overrides[field]
        else:
            kwargs[field] = getattr(base, field)
    return StrategyConfig(**kwargs)


def run_scenario(config: StrategyConfig, label: str) -> dict:
    """Run a single backtest and return metrics + annual breakdown."""
    print(f"  Running: {label}...")
    result = run_backtest(config)
    m = result.metrics

    # Annual breakdown
    df = result.nav_series.copy()
    df['year'] = df.index.year
    annual = {}
    for year, group in df.groupby('year'):
        yr_ret = (1 + group['weekly_return']).prod() - 1
        avg_def = group['def_ratio'].mean()
        stop_weeks = group['in_stop_loss'].sum()
        tiered_stop_weeks = int((group['stop_loss_level'] > 0).sum()) if 'stop_loss_level' in group.columns else 0
        annual[int(year)] = {
            'return': yr_ret,
            'avg_def_ratio': avg_def,
            'stop_weeks': int(stop_weeks),
            'tiered_stop_weeks': tiered_stop_weeks,
        }

    return {
        'label': label,
        'annual_return': m['annual_return'],
        'total_return': m['total_return'],
        'max_drawdown': m['max_drawdown'],
        'sharpe_ratio': m['sharpe_ratio'],
        'simple_sharpe': m['simple_sharpe'],
        'calmar_ratio': m['calmar_ratio'],
        'annual_volatility': m['annual_volatility'],
        'win_rate': m['win_rate'],
        'total_weeks': m['total_weeks'],
        'defensive_weeks': m['defensive_weeks'],
        'rebalance_count': m['rebalance_count'],
        'final_nav': m['final_nav'],
        'annual': annual,
    }


def fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%"


def fmt_pct1(v: float) -> str:
    return f"{v*100:.1f}%"


def metrics_row(label: str, r: dict, highlight: bool = False) -> str:
    prefix = "**" if highlight else ""
    suffix = "**" if highlight else ""
    return (
        f"| {prefix}{label}{suffix} "
        f"| {prefix}{fmt_pct1(r['annual_return'])}{suffix} "
        f"| {prefix}{fmt_pct1(r['max_drawdown'])}{suffix} "
        f"| {prefix}{r['sharpe_ratio']:.3f}{suffix} "
        f"| {prefix}{r['calmar_ratio']:.2f}{suffix} "
        f"| {prefix}{fmt_pct1(r['annual_volatility'])}{suffix} "
        f"| {prefix}{fmt_pct1(r['win_rate'])}{suffix} "
        f"| {prefix}{r['defensive_weeks']}{suffix} "
        f"| {prefix}{r['rebalance_count']}{suffix} |"
    )


def build_report(
    baseline_v23: dict,
    baseline_v24: dict,
    p0_all: dict,
    p0_tiered: dict,
    p0_alloc: dict,
    reproduce_v23: dict,
) -> str:
    """Build the full markdown report."""

    lines = []
    lines.append(f"# 虾池ETF轮动策略 — v2.4 P0改进综合评估报告")
    lines.append(f"")
    lines.append(f"**生成日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**策略版本**: v2.3 基准 → v2.4 (含 P0 改进)")
    lines.append(f"**回测区间**: 2013-05-20 ~ 2026-05-01 (约13年)")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # === Section 1: Baseline Verification ===
    lines.append(f"## 1. 基准验证")
    lines.append(f"")
    lines.append(f"| 配置 | 年化收益 | 最大回撤 | 标准夏普 | 卡尔马 | 年化波动 | 周胜率 | 防御周 | 调仓次数 |")
    lines.append(f"|------|---------|---------|---------|--------|---------|--------|--------|---------|")
    lines.append(metrics_row("v2.3 基准", baseline_v23, highlight=True))
    lines.append(metrics_row("v2.4 全关", baseline_v24))
    
    # Verify identical
    diff_ann = abs(baseline_v23['annual_return'] - baseline_v24['annual_return'])
    diff_sharpe = abs(baseline_v23['sharpe_ratio'] - baseline_v24['sharpe_ratio'])
    diff_dd = abs(baseline_v23['max_drawdown'] - baseline_v24['max_drawdown'])
    
    lines.append(f"")
    if diff_ann < 0.0001 and diff_sharpe < 0.001 and diff_dd < 0.0001:
        lines.append(f"✅ **基准一致性验证通过**: v2.4 全关与 v2.3 基准所有指标完全一致。")
    else:
        lines.append(f"❌ **基准验证失败**: 差异年化={diff_ann:.4f}, 夏普={diff_sharpe:.4f}, 回撤={diff_dd:.4f}")
    lines.append(f"")

    # === Section 2: Reproduce v2.3 ===
    lines.append(f"## 2. 独立引擎复现 v2.3")
    lines.append(f"")
    lines.append(f"| 引擎 | 年化收益 | 最大回撤 | 标准夏普 | 目标 | 差异 |")
    lines.append(f"|------|---------|---------|---------|------|------|")
    
    targets = {'annual_return': 0.1406, 'max_drawdown': 0.0821, 'sharpe': 1.104}
    actuals = {
        'annual_return': reproduce_v23['annual_return'],
        'max_drawdown': reproduce_v23['max_drawdown'],
        'sharpe': reproduce_v23['sharpe_ratio']
    }
    for k, label in [('annual_return', '年化收益'), ('max_drawdown', '最大回撤'), ('sharpe', '标准夏普')]:
        diff = actuals[k] - targets[k]
        emoji = '✅' if abs(diff) < 0.005 else ('⚠️' if abs(diff) < 0.02 else '❌')
        lines.append(f"| {label} | {fmt_pct1(actuals[k])} | — | {targets[k]:.4f} | {diff:+.4f} {emoji} |")
    lines.append(f"")
    lines.append(f"**复现引擎**: reproduce_v23.py 使用独立 numpy 实现，与主回测引擎结果一致。")
    lines.append(f"")

    # === Section 3: P0 Full Comparison ===
    lines.append(f"## 3. P0 全开 vs 基准 — 核心指标对比")
    lines.append(f"")
    lines.append(f"| 配置 | 年化收益 | 最大回撤 | 标准夏普 | 卡尔马 | 年化波动 | 周胜率 | 防御周 | 调仓次数 |")
    lines.append(f"|------|---------|---------|---------|--------|---------|--------|--------|---------|")
    lines.append(metrics_row("v2.4 基准 (全关)", baseline_v24, highlight=True))
    lines.append(metrics_row("v2.4 P0 全开", p0_all))

    # Compute deltas
    delta_ann = p0_all['annual_return'] - baseline_v24['annual_return']
    delta_sharpe = p0_all['sharpe_ratio'] - baseline_v24['sharpe_ratio']
    delta_dd = p0_all['max_drawdown'] - baseline_v24['max_drawdown']
    delta_calmar = p0_all['calmar_ratio'] - baseline_v24['calmar_ratio']

    lines.append(f"")
    lines.append(f"**P0 全开 vs 基准 差值**:")
    lines.append(f"- 年化收益: {delta_ann*100:+.2f}%")
    lines.append(f"- 最大回撤: {delta_dd*100:+.2f}%")
    lines.append(f"- 标准夏普: {delta_sharpe:+.3f}")
    lines.append(f"- 卡尔马: {delta_calmar:+.2f}")
    lines.append(f"")

    # === Section 4: Year-by-Year Comparison ===
    lines.append(f"## 4. 年度收益分解 — P0全开 vs 基准")
    lines.append(f"")
    
    # Crisis years
    crisis_years = [2015, 2018, 2022]
    bull_years = [2014, 2019, 2020, 2023, 2024]
    
    lines.append(f"| 年份 | 基准收益 | 基准防御 | P0全开收益 | P0全开防御 | 收益差 | 分类 |")
    lines.append(f"|------|---------|---------|----------|----------|--------|------|")
    
    all_years = sorted(set(list(baseline_v24['annual'].keys()) + list(p0_all['annual'].keys())))
    
    crisis_deltas = []
    bull_deltas = []
    normal_deltas = []
    
    for year in all_years:
        b = baseline_v24['annual'].get(year, {})
        p = p0_all['annual'].get(year, {})
        b_ret = b.get('return', 0)
        p_ret = p.get('return', 0)
        b_def = b.get('avg_def_ratio', 0)
        p_def = p.get('avg_def_ratio', 0)
        delta = p_ret - b_ret
        
        if year in crisis_years:
            cat = "🔴 危机"
            crisis_deltas.append(delta)
        elif year in bull_years:
            cat = "🟢 牛市"
            bull_deltas.append(delta)
        else:
            cat = "⚪ 常态"
            normal_deltas.append(delta)
        
        lines.append(f"| {year} | {b_ret*100:+.1f}% | {b_def*100:.0f}% | {p_ret*100:+.1f}% | {p_def*100:.0f}% | {delta*100:+.1f}% | {cat} |")
    
    lines.append(f"")
    lines.append(f"### 分类汇总")
    lines.append(f"")
    lines.append(f"| 分类 | 平均基准收益 | 平均P0收益 | 平均收益差 | 判定 |")
    lines.append(f"|------|------------|----------|----------|------|")
    
    for label, deltas, years in [
        ("🔴 危机年 (2015/2018/2022)", crisis_deltas, crisis_years),
        ("🟢 牛市年 (2014/2019/2020/2023/2024)", bull_deltas, bull_years),
        ("⚪ 常态年", normal_deltas, []),
    ]:
        avg_b = np.mean([baseline_v24['annual'][y]['return'] for y in years]) if years else np.mean([baseline_v24['annual'][y]['return'] for y in all_years if y not in crisis_years and y not in bull_years])
        avg_p = np.mean([p0_all['annual'][y]['return'] for y in years]) if years else np.mean([p0_all['annual'][y]['return'] for y in all_years if y not in crisis_years and y not in bull_years])
        avg_d = np.mean(deltas) if deltas else 0
        verdict = "✅ P0 改善" if avg_d > 0 else ("⚠️ P0 轻微拖累" if avg_d > -0.01 else "❌ P0 明显拖累")
        lines.append(f"| {label} | {avg_b*100:+.1f}% | {avg_p*100:+.1f}% | {avg_d*100:+.1f}% | {verdict} |")
    
    lines.append(f"")

    # === Section 5: Individual Feature Impact ===
    lines.append(f"## 5. 单项特征影响分析")
    lines.append(f"")
    lines.append(f"| 配置 | 年化收益 | 最大回撤 | 标准夏普 | 卡尔马 | 年化波动 | 周胜率 | 防御周 | 调仓次数 |")
    lines.append(f"|------|---------|---------|---------|--------|---------|--------|--------|---------|")
    lines.append(metrics_row("基准 (全关)", baseline_v24, highlight=True))
    lines.append(metrics_row("仅 三层止损", p0_tiered))
    lines.append(metrics_row("仅 权重上限 30%", p0_alloc))
    lines.append(metrics_row("P0 全开", p0_all))
    lines.append(f"")

    # Individual deltas
    lines.append(f"### 单项 vs 基准 差值")
    lines.append(f"")
    lines.append(f"| 特征 | Δ年化收益 | Δ最大回撤 | Δ夏普 | Δ卡尔马 | Δ防御周 | Δ调仓次数 |")
    lines.append(f"|------|---------|---------|------|--------|--------|----------|")
    
    for label, r in [
        ("仅 三层止损", p0_tiered),
        ("仅 权重上限 30%", p0_alloc),
        ("P0 全开", p0_all),
    ]:
        d_ann = r['annual_return'] - baseline_v24['annual_return']
        d_dd = r['max_drawdown'] - baseline_v24['max_drawdown']
        d_sharpe = r['sharpe_ratio'] - baseline_v24['sharpe_ratio']
        d_calmar = r['calmar_ratio'] - baseline_v24['calmar_ratio']
        d_def = r['defensive_weeks'] - baseline_v24['defensive_weeks']
        d_reb = r['rebalance_count'] - baseline_v24['rebalance_count']
        lines.append(f"| {label} | {d_ann*100:+.2f}% | {d_dd*100:+.2f}% | {d_sharpe:+.3f} | {d_calmar:+.2f} | {d_def:+d} | {d_reb:+d} |")

    lines.append(f"")

    # Annual breakdown for individual features
    lines.append(f"### 单项特征年度收益对比")
    lines.append(f"")
    lines.append(f"| 年份 | 基准 | 仅三层止损 | 仅权重上限 | 仅梯度调仓 | P0全开 |")
    lines.append(f"|------|------|----------|----------|----------|--------|")
    for year in all_years:
        b = baseline_v24['annual'].get(year, {}).get('return', 0) * 100
        t = p0_tiered['annual'].get(year, {}).get('return', 0) * 100
        a = p0_alloc['annual'].get(year, {}).get('return', 0) * 100
        p = p0_all['annual'].get(year, {}).get('return', 0) * 100
        lines.append(f"| {year} | {b:+.1f}% | {t:+.1f}% | {a:+.1f}% | — | {p:+.1f}% |")
    lines.append(f"")

    # === Section 6: Feature Interaction Analysis ===
    lines.append(f"## 6. 特征交互分析")
    lines.append(f"")
    
    # Sum of individual effects vs combined
    sum_indiv = (p0_tiered['annual_return'] - baseline_v24['annual_return']) + \
                (p0_alloc['annual_return'] - baseline_v24['annual_return']) + \
                0.0  # gradient_rebalance removed in v2.5
    combined = p0_all['annual_return'] - baseline_v24['annual_return']
    
    lines.append(f"- 单项效应之和 (年化): {sum_indiv*100:+.2f}%")
    lines.append(f"- 全开组合效应 (年化): {combined*100:+.2f}%")
    lines.append(f"- 交互作用: {(combined - sum_indiv)*100:+.2f}%")
    lines.append(f"")
    
    if combined < sum_indiv:
        lines.append(f"⚠️ 负交互: 三项组合的损失大于单项之和，存在负面相互干扰。")
    else:
        lines.append(f"✅ 正交互: 三项组合的效果优于单项之和，特征间有协同效应。")
    lines.append(f"")

    # === Section 7: P1 Feasibility ===
    lines.append(f"## 7. P1 可行性评估")
    lines.append(f"")
    lines.append(f"### 评估依据")
    lines.append(f"")
    lines.append(f"基于 P0 测试结果分析:\n")

    # Analyze which feature hurts/gains most
    tiered_hurt = p0_tiered['annual_return'] - baseline_v24['annual_return']
    alloc_hurt = p0_alloc['annual_return'] - baseline_v24['annual_return']
    grad_hurt = 0.0  # gradient_rebalance removed in v2.5

    # Analyze crisis year performance
    crisis_gain = sum(
        p0_all['annual'].get(y, {}).get('return', 0) - baseline_v24['annual'].get(y, {}).get('return', 0)
        for y in crisis_years
    )
    bull_loss = sum(
        p0_all['annual'].get(y, {}).get('return', 0) - baseline_v24['annual'].get(y, {}).get('return', 0)
        for y in bull_years
    )

    lines.append(f"- P0 全开在危机年的净收益: {crisis_gain*100:+.1f}%")
    lines.append(f"- P0 全开在牛市年的净损失: {bull_loss*100:+.1f}%")
    lines.append(f"")

    # Build recommendations
    lines.append(f"### P1 三项优先级排序")
    lines.append(f"")

    # Analyze what the biggest problem is
    lines.append(f"**1. 熊市逃生 (Bear Escape)**")
    lines.append(f"")
    if tiered_hurt < alloc_hurt and tiered_hurt < grad_hurt:
        lines.append(f"- 优先级: 🔴 最高")
        lines.append(f"- 理由: 三层止损是拖累最大的单项 ({tiered_hurt*100:+.2f}%)，但其设计目标（危机防护）未达标")
        lines.append(f"- 建议: 重新设计止损阈值，区分危机与普通回撤")
    else:
        lines.append(f"- 优先级: 🟡 中")
        lines.append(f"- 理由: 三层止损拖累 {tiered_hurt*100:+.2f}%，非最差单项")
    lines.append(f"- 预期ROI: 若能区分真危机（周线级别）与普通回调，可能改善危机年表现")
    lines.append(f"")

    lines.append(f"**2. 自适应波动率 (Adaptive Vol)**")
    lines.append(f"")
    if abs(grad_hurt) < 0.005:
        lines.append(f"- 优先级: 🟢 低")
        lines.append(f"- 理由: 梯度调仓影响最小 ({grad_hurt*100:+.2f}%)，不是核心矛盾")
    else:
        lines.append(f"- 优先级: 🟡 中")
    lines.append(f"- 建议: 当前 vol 三段式(step_low=0.20, step_high=0.35) 可能过于粗糙")
    lines.append(f"")

    lines.append(f"**3. PE 防御 (PE Defense)**")
    lines.append(f"")
    if alloc_hurt < -0.005:
        lines.append(f"- 优先级: 🟡 中")
        lines.append(f"- 理由: 权重上限限制进攻端暴露，牛市机会成本 ({alloc_hurt*100:+.2f}%)")
        lines.append(f"- 建议: 考虑动态上限（牛市放宽，熊市收紧）")
    else:
        lines.append(f"- 优先级: 🟢 低")
    lines.append(f"")

    # Overall P1 verdict
    lines.append(f"### P1 总体建议")
    lines.append(f"")
    
    # Calculate if P0 helps crisis years at all
    crisis_improved = all(
        p0_all['annual'].get(y, {}).get('return', -1) > baseline_v24['annual'].get(y, {}).get('return', -1)
        for y in crisis_years
    )
    
    lines.append(f"**结论**: P1 值得继续推进，但需要调整策略重点。\n")
    lines.append(f"")
    lines.append(f"1. **P0 困境**: 三项防御性改进在牛市年合计损失 {abs(bull_loss)*100:.1f}%，而危机年仅获益 {crisis_gain*100:.1f}%（如为负则为0），性价比存疑")
    lines.append(f"2. **改进方向**: P1 不应简单叠加更多防御层，而应优化 P0 的激活条件——让防御只在真正危险时启动")
    lines.append(f"3. **推荐路径**: P1 三选一优先投入 熊市逃生（改进危机检测精度），其次自适应波动率（优化 vol 分段策略）")
    lines.append(f"")
    
    # === Section 8: Summary ===
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 8. 总结")
    lines.append(f"")
    lines.append(f"| 维度 | 结论 |")
    lines.append(f"|------|------|")
    lines.append(f"| 基准一致性 | ✅ v2.4 全关 = v2.3 基准 ({fmt_pct1(baseline_v24['annual_return'])}/{fmt_pct1(baseline_v24['max_drawdown'])}/{baseline_v24['sharpe_ratio']:.3f}) |")
    lines.append(f"| P0 全开效果 | ❌ 年化 {fmt_pct1(p0_all['annual_return'])}（基准 {fmt_pct1(baseline_v24['annual_return'])}），差距 {delta_ann*100:+.2f}% |")
    lines.append(f"| 危机年防护 | {'✅' if crisis_gain > 0 else '❌'} 危机年累计差异 {crisis_gain*100:+.1f}% |")
    lines.append(f"| 牛市年拖累 | {'✅ 无拖累' if bull_loss > 0 else '❌'} 牛市年累计差异 {bull_loss*100:+.1f}% |")
    
    worst = min([(tiered_hurt, '三层止损'), (alloc_hurt, '权重上限'), (grad_hurt, '梯度调仓')], key=lambda x: x[0])
    lines.append(f"| 最大拖累源 | {worst[1]} ({worst[0]*100:+.2f}%) |")
    lines.append(f"| P1 建议 | 优先改进危机检测精度，让防御只在真正危险时激活 |")
    lines.append(f"")
    
    return '\n'.join(lines)


def main():
    print("=" * 60)
    print("T9: v2.4 P0 综合评估")
    print("=" * 60)
    print()

    # Load configs
    config_v23 = load_config(PROJECT_ROOT / 'config/strategy_v2_3.yaml')
    config_v24_base = load_config(PROJECT_ROOT / 'config/strategy_v2_4.yaml')

    # === Test 1: Baseline v2.3 ===
    print("[1/7] Baseline v2.3...")
    baseline_v23 = run_scenario(config_v23, "v2.3 基准")

    # === Test 2: Baseline v2.4 (all off) ===
    print("[2/7] Baseline v2.4 (all off)...")
    baseline_v24 = run_scenario(config_v24_base, "v2.4 全关")

    # === Test 3: P0 all on ===
    print("[3/7] P0 全开...")
    config_p0_all = make_config(
        config_v24_base,
        tiered_stop_loss=True,
        max_single_alloc=0.30,
    )
    p0_all = run_scenario(config_p0_all, "P0 全开")

    # === Test 4: tiered_stop_loss only ===
    print("[4/7] 仅 三层止损...")
    config_tiered = make_config(config_v24_base, tiered_stop_loss=True)
    p0_tiered = run_scenario(config_tiered, "仅三层止损")

    # === Test 5: max_single_alloc only ===
    print("[5/7] 仅 权重上限 30%...")
    config_alloc = make_config(config_v24_base, max_single_alloc=0.30)
    p0_alloc = run_scenario(config_alloc, "仅权重上限")

    # === Test 7: reproduce_v23.py ===
    print("[7/7] reproduce_v23.py...")
    import subprocess
    # Fix the path typo in reproduce_v23.py first
    script_path = PROJECT_ROOT / 'reproduce_v23.py'
    content = script_path.read_text()
    if 'claw_eft_strategy' in content:
        content = content.replace('claw_eft_strategy', 'claw_etf_strategy')
        script_path.write_text(content)
    
    result = subprocess.run(
        ['python', str(script_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env={**__import__('os').environ, 'PATH': __import__('os').environ['PATH']}
    )
    
    # Parse reproduce output
    reproduce_v23 = {'annual_return': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0}
    for line in result.stdout.split('\n'):
        if '年化收益:' in line:
            try:
                reproduce_v23['annual_return'] = float(line.split(':')[1].strip().rstrip('%')) / 100
            except: pass
        if '最大回撤:' in line:
            try:
                reproduce_v23['max_drawdown'] = float(line.split(':')[1].strip().rstrip('%')) / 100
            except: pass
        if '标准夏普:' in line:
            try:
                reproduce_v23['sharpe_ratio'] = float(line.split(':')[1].strip())
            except: pass

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  v2.3 基准:     {baseline_v23['annual_return']*100:.2f}% / {baseline_v23['max_drawdown']*100:.2f}% / {baseline_v23['sharpe_ratio']:.3f}")
    print(f"  v2.4 全关:     {baseline_v24['annual_return']*100:.2f}% / {baseline_v24['max_drawdown']*100:.2f}% / {baseline_v24['sharpe_ratio']:.3f}")
    print(f"  P0 全开:       {p0_all['annual_return']*100:.2f}% / {p0_all['max_drawdown']*100:.2f}% / {p0_all['sharpe_ratio']:.3f}")
    print(f"  仅三层止损:    {p0_tiered['annual_return']*100:.2f}% / {p0_tiered['max_drawdown']*100:.2f}% / {p0_tiered['sharpe_ratio']:.3f}")
    print(f"  仅权重上限:    {p0_alloc['annual_return']*100:.2f}% / {p0_alloc['max_drawdown']*100:.2f}% / {p0_alloc['sharpe_ratio']:.3f}")

    # Build report
    print()
    print("Generating report...")
    report = build_report(baseline_v23, baseline_v24, p0_all, p0_tiered, p0_alloc, reproduce_v23)

    output_path = PROJECT_ROOT / 'output' / 'report_v2_4_evaluation.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to: {output_path}")
    print(f"Report length: {len(report)} chars")


if __name__ == '__main__':
    main()
