#!/usr/bin/env python3
"""消融实验 runner — 任务ID 4

对 v4.3 基线与 A1/A2/A2b/A3/A4/A5/A6 消融变体逐一运行全期回测，
汇总 Sharpe/年化/MaxDD/Calmar/周胜率，统计止损触发次数，
并逐周比对 A5(无止损) 与基线的 NAV 序列是否完全一致。

只读取 src/ 与 config/，不修改任何现有文件。
结果 JSON 落盘 output/experiments/ablation_results.json，
markdown 表格片段打印到 stdout（供 exp_ablation.md 引用）。
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy import load_config
from src.backtest import run_backtest

VARIANTS = [
    ('v4.3 基线', 'config/strategy_v4_3.yaml'),
    ('A1 纯动量', 'config/experiments/ablation_a1_pure_momentum.yaml'),
    ('A2b 评分无粘性', 'config/experiments/ablation_a2b_score_no_margin.yaml'),
    ('A2 Layer1完整', 'config/experiments/ablation_a2_layer1_full.yaml'),
    ('A3 L1+L2无防御', 'config/experiments/ablation_a3_no_defense.yaml'),
    ('A4 减crisis', 'config/experiments/ablation_a4_no_crisis_boost.yaml'),
    ('A5 减止损', 'config/experiments/ablation_a5_no_stoploss.yaml'),
    ('A6 减inv_vol', 'config/experiments/ablation_a6_no_invvol.yaml'),
]


def count_stop_loss(records: list[dict]) -> tuple[int, int]:
    """统计止损触发事件数（False->True 跳变）与处于止损状态的周数。"""
    triggers = 0
    weeks = 0
    prev = False
    for r in records:
        cur = bool(r['in_stop_loss'])
        if cur and not prev:
            triggers += 1
        if cur:
            weeks += 1
        prev = cur
    return triggers, weeks


def main():
    results = {}
    navs = {}
    for label, cfg_rel in VARIANTS:
        cfg = load_config(PROJECT_ROOT / cfg_rel)
        res = run_backtest(cfg)
        m = res.metrics
        sl_triggers, sl_weeks = count_stop_loss(res.weekly_records)
        results[label] = {
            'config': cfg_rel,
            'sharpe': round(m['sharpe_ratio'], 3),
            'annual_return': round(m['annual_return'], 4),
            'max_drawdown': round(m['max_drawdown'], 4),
            'calmar': round(m['calmar_ratio'], 2),
            'win_rate': round(m['win_rate'], 4),
            'annual_vol': round(m['annual_volatility'], 4),
            'total_weeks': m['total_weeks'],
            'defensive_weeks': m['defensive_weeks'],
            'rebalance_count': m['rebalance_count'],
            'final_nav': round(m['final_nav'], 4),
            'stop_loss_triggers': sl_triggers,
            'stop_loss_weeks': sl_weeks,
        }
        navs[label] = res.nav_series['nav'].values
        print(f"[done] {label:<14} Sharpe={m['sharpe_ratio']:.3f} "
              f"Ann={m['annual_return']*100:.2f}% MaxDD={m['max_drawdown']*100:.2f}% "
              f"Calmar={m['calmar_ratio']:.2f} Win={m['win_rate']*100:.1f}% "
              f"止损触发={sl_triggers}次/{sl_weeks}周")

    # A5 与基线逐周 NAV 一致性检验（证实止损从未触发）
    base_nav, a5_nav = navs['v4.3 基线'], navs['A5 减止损']
    a5_identical = bool(len(base_nav) == len(a5_nav)
                        and np.allclose(base_nav, a5_nav, rtol=0, atol=1e-12))
    max_diff = float(np.max(np.abs(base_nav - a5_nav))) if len(base_nav) == len(a5_nav) else None
    results['_a5_vs_baseline'] = {'identical': a5_identical, 'max_abs_nav_diff': max_diff}
    print(f"\nA5 vs 基线 NAV 逐周一致: {a5_identical} (max |diff| = {max_diff})")

    out_path = PROJECT_ROOT / 'output/experiments/ablation_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}\n")

    # markdown 对比表
    print('| 变体 | Sharpe | 年化 | MaxDD | Calmar | 周胜率 | 年化波动 | 止损触发 |')
    print('|------|--------|------|-------|--------|--------|----------|----------|')
    for label, _ in VARIANTS:
        r = results[label]
        print(f"| {label} | {r['sharpe']:.3f} | {r['annual_return']*100:.2f}% "
              f"| {r['max_drawdown']*100:.2f}% | {r['calmar']:.2f} "
              f"| {r['win_rate']*100:.1f}% | {r['annual_vol']*100:.2f}% "
              f"| {r['stop_loss_triggers']} 次 |")


if __name__ == '__main__':
    main()
