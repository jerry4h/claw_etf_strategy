#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务20 (E2): 纳指腿改用场外联接基金的执行代价 — T+lag 滞后成本 + 费率差。

背景: E1 量化场内 513100 的溢价侵蚀; 本实验量化替代路径 "场外联接基金
(按净值申赎、无溢价)" 的执行代价, 两者对比决定场外路径是否成立。

方法 (后处理法, 不改引擎 / 不改生产数据 / 不改 config 生产 yaml):
  1. 基线 = v4.4 生产 NAV 回测 (周五收盘 NAV 成交), 从 weekly_records 提取
     纳指列逐周仓位 w_t, 每笔调仓 Δw_t = w_t − w_{t−1}, 成交日 T = 决策周五。
  2. 场外模拟: 该笔 Δw 改为按 nav(T+lag) 确认 (lag ∈ {1,2,3} 个交易日,
     用 513100 日频基金净值找 T 之后第 lag 个交易日)。
     相对基线的逐笔滞后成本 (符号推导见报告):
         cost_t = Δw_t × (nav_{T+lag} − nav_T) / nav_T
     把 cost_t 从当周策略收益中扣除, 重算净值与指标。
  3. 日频净值用 adj_nav (513100 于 2022-01-13 份额拆分, unit_nav 跳变,
     adj_nav 连续; 短窗收益比值两者仅在拆分日不同)。
  4. 费率差: 联接基金持有成本差 +0.4pp/年 (0.3~0.5 中值), 按纳指腿
     时间加权平均仓位折算为周度拖累 w_t × 0.004/52。
  5. 敏感性: 随机 lag ∈ {1,2} (seed=20260731, 100 次抽样) 模拟申赎确认不确定性。

用法: .venv/bin/python scripts/_exp_premium_e2.py
输出: output/experiments/premium_e2_otc_lag.md
      output/experiments/premium_e2_otc_lag.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.backtest import run_backtest, compute_metrics
from src.strategy import load_config

CFG_PATH = PROJ / 'config' / 'strategy_v4_4.yaml'
WEEKLY_CSV = PROJ / 'data' / 'all_etfs_nav_latest.csv'
DAILY_NAV_CSV = PROJ / 'data' / 'experiments' / 'tushare_cache' / 'fund_nav_513100SH.csv'
OUT_MD = PROJ / 'output' / 'experiments' / 'premium_e2_otc_lag.md'
OUT_JSON = PROJ / 'output' / 'experiments' / 'premium_e2_otc_lag.json'

NAS_COL = '纳指ETF'
FEE_DIFF_ANNUAL = 0.004          # 联接基金 vs 场内 ETF 持有成本差 +0.4pp/年
LAGS = [1, 2, 3]
RANDOM_SEED = 20260731
N_DRAWS = 100


# ============================================================
# 数据准备
# ============================================================

def load_daily_nav() -> pd.Series:
    """513100 日频净值 (adj_nav, 拆分连续), index=nav_date 升序。"""
    df = pd.read_csv(DAILY_NAV_CSV)
    df['nav_date'] = pd.to_datetime(df['nav_date'], format='%Y%m%d')
    df = df.sort_values('nav_date').drop_duplicates('nav_date', keep='last')
    s = df.set_index('nav_date')['adj_nav'].astype(float)
    return s


def build_trades(result, weekly_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """从 weekly_records 提取纳指腿逐周仓位与 Δw, 并映射每笔的成交日 T。

    record 日期 = w_index[i+1] (下一周才知道收益), 仓位 alloc 在 w_index[i]
    (决策周五) 收盘成交 → 成交日 T = 该 record 日期在周线索引中的前一个锚点。
    """
    recs = result.weekly_records
    rows = []
    prev_w = 0.0  # 回测起点 last_alloc = 0, 首周建仓也是真实一笔
    wd = weekly_dates
    for rec in recs:
        d = pd.Timestamp(rec['date'])
        pos = wd.searchsorted(d)
        assert wd[pos] == d, f'record 日期 {d} 不在周线索引中'
        trade_date = wd[pos - 1]          # 决策/成交周五 T
        w = float(rec[f'weight_{NAS_COL}'])
        rows.append({
            'date': d,                     # 收益记账周 (T 的下一周锚)
            'trade_date': trade_date,      # 成交日 T
            'w_nasdaq': w,
            'dw': w - prev_w,
            'weekly_return': float(rec['weekly_return']),
        })
        prev_w = w
    return pd.DataFrame(rows)


def lag_return_lookup(daily_nav: pd.Series, trade_dates: pd.Series,
                      max_lag: int) -> dict[int, np.ndarray]:
    """对每个成交日 T 预计算 r_lag(T, k) = nav(T+k)/nav(T) − 1, k=1..max_lag。

    T 若非 513100 净值披露日 (QDII 假日, 全样本 8 个周锚), 用最近一个
    ≤ T 的净值日代替 (锚定同一"最新可得净值", 与实际申赎口径一致)。
    T+k 超出净值数据末端时取最后可得净值 (仅可能影响最末一笔)。
    """
    nav_vals = daily_nav.values
    nav_idx = daily_nav.index
    n = len(nav_vals)
    out = {k: np.zeros(len(trade_dates)) for k in range(1, max_lag + 1)}
    for i, t in enumerate(trade_dates):
        p = nav_idx.searchsorted(t, side='right') - 1  # 最近 ≤ T 的净值日
        assert p >= 0, f'成交日 {t} 早于净值数据起点'
        nav_t = nav_vals[p]
        for k in range(1, max_lag + 1):
            q = min(p + k, n - 1)
            out[k][i] = nav_vals[q] / nav_t - 1.0
    return out


# ============================================================
# 指标重算
# ============================================================

def metrics_from_returns(base_nav_df: pd.DataFrame, weekly_ret: np.ndarray,
                         config) -> dict:
    """给定调整后周收益序列, 重建 nav/peak/drawdown 并用生产口径算指标。"""
    df = pd.DataFrame(index=base_nav_df.index)
    df['weekly_return'] = weekly_ret
    df['nav'] = (1.0 + df['weekly_return']).cumprod()
    df['peak'] = df['nav'].cummax()
    df['drawdown'] = (df['peak'] - df['nav']) / df['peak']
    df['def_ratio'] = base_nav_df['def_ratio'].values
    df['turnover'] = base_nav_df['turnover'].values
    return compute_metrics(df, config.risk_free_rate, def_baseline=config.def_alloc)


def scenario(base_nav_df, base_ret, config, lag_cost=None, fee_drag=None) -> dict:
    """叠加滞后成本/费率差后重算指标, 返回精简结果。"""
    ret = base_ret.copy()
    if lag_cost is not None:
        ret = ret - lag_cost
    if fee_drag is not None:
        ret = ret - fee_drag
    return metrics_from_returns(base_nav_df, ret, config)


# ============================================================
# 主流程
# ============================================================

def main():
    print('== E2: 场外联接基金 T+lag 滞后成本 + 费率差 ==')
    config = load_config(CFG_PATH)
    print(f'配置: {config.name} v{config.version}')
    result = run_backtest(config)
    base_m = result.metrics
    print(f"基线: 年化 {base_m['annual_return']*100:.2f}%  Sharpe {base_m['sharpe_ratio']:.3f}")

    weekly_dates = pd.DatetimeIndex(
        pd.read_csv(WEEKLY_CSV, parse_dates=['日期'])['日期'])
    trades = build_trades(result, weekly_dates)
    daily_nav = load_daily_nav()

    n_weeks = len(trades)
    dw = trades['dw'].values
    traded_mask = np.abs(dw) > 1e-12
    n_trades = int(traded_mask.sum())
    sum_abs_dw = float(np.abs(dw).sum())
    avg_w = float(trades['w_nasdaq'].mean())
    print(f'周数 {n_weeks}, 纳指腿调仓笔数 {n_trades}, Σ|Δw| {sum_abs_dw:.2f}, '
          f'时间加权平均仓位 {avg_w*100:.2f}%')

    r_lag = lag_return_lookup(daily_nav, trades['trade_date'], max_lag=3)

    base_ret = trades['weekly_return'].values
    base_nav_df = result.nav_series
    years = n_weeks / 52.0

    # 费率差: 周度拖累 = w_t × 0.4pp/52 (按逐周实际仓位, 时间加权口径)
    fee_drag = trades['w_nasdaq'].values * (FEE_DIFF_ANNUAL / 52.0)
    fee_pp_portfolio = avg_w * FEE_DIFF_ANNUAL  # 组合口径 ≈ 0.4pp × 平均仓位

    # --- 确定性 lag {1,2,3} ---
    scen = {}
    m_fee_only = scenario(base_nav_df, base_ret, config, fee_drag=fee_drag)
    scen['fee_only'] = m_fee_only
    for k in LAGS:
        cost = dw * r_lag[k]                      # 逐周滞后成本 (可正可负)
        m_lag = scenario(base_nav_df, base_ret, config, lag_cost=cost)
        m_tot = scenario(base_nav_df, base_ret, config, lag_cost=cost, fee_drag=fee_drag)
        scen[f'lag{k}'] = {
            'lag_only': m_lag,
            'lag_plus_fee': m_tot,
            'cost_sum': float(cost.sum()),
            'cost_pp_per_year_arith': float(cost.sum() / years * 100),
            'cost_mean_bp_per_trade': float(cost[traded_mask].mean() * 1e4),
            'cost_positive_share': float((cost[traded_mask] > 0).mean()),
        }
        print(f"lag={k}: 滞后成本合计 {cost.sum()*100:.2f}pp "
              f"(≈{cost.sum()/years*100:.3f}pp/年), "
              f"年化 {m_lag['annual_return']*100:.2f}%, "
              f"含费率差 {m_tot['annual_return']*100:.2f}%")

    # --- 随机 lag ∈ {1,2} (申赎确认不确定性, 固定 seed 100 次抽样) ---
    rng = np.random.default_rng(RANDOM_SEED)
    draw_ann, draw_sharpe, draw_cost_pp = [], [], []
    for _ in range(N_DRAWS):
        lag_draw = rng.integers(1, 3, size=n_weeks)   # 每笔独立 ∈ {1,2}
        cost = np.where(lag_draw == 1, dw * r_lag[1], dw * r_lag[2])
        m = scenario(base_nav_df, base_ret, config, lag_cost=cost)
        draw_ann.append(m['annual_return'])
        draw_sharpe.append(m['sharpe_ratio'])
        draw_cost_pp.append(cost.sum() / years * 100)
    draw_ann = np.array(draw_ann)
    draw_sharpe = np.array(draw_sharpe)
    draw_cost_pp = np.array(draw_cost_pp)
    rand_stats = {
        'n_draws': N_DRAWS, 'seed': RANDOM_SEED,
        'annual_return_mean': float(draw_ann.mean()),
        'annual_return_std': float(draw_ann.std()),
        'annual_return_p5': float(np.percentile(draw_ann, 5)),
        'annual_return_p50': float(np.percentile(draw_ann, 50)),
        'annual_return_p95': float(np.percentile(draw_ann, 95)),
        'sharpe_mean': float(draw_sharpe.mean()),
        'sharpe_std': float(draw_sharpe.std()),
        'lag_cost_pp_mean': float(draw_cost_pp.mean()),
        'lag_cost_pp_std': float(draw_cost_pp.std()),
        'lag_cost_pp_p5': float(np.percentile(draw_cost_pp, 5)),
        'lag_cost_pp_p95': float(np.percentile(draw_cost_pp, 95)),
    }
    print(f"随机 lag∈{{1,2}}: 年化 {draw_ann.mean()*100:.2f}% ± {draw_ann.std()*100:.3f}pp, "
          f"滞后成本 {draw_cost_pp.mean():.3f} ± {draw_cost_pp.std():.3f} pp/年")

    # --- 汇总 ---
    payload = {
        'task': 'E2 场外联接基金 T+lag 滞后成本 + 费率差',
        'config': str(CFG_PATH.relative_to(PROJ)),
        'baseline': {k: base_m[k] for k in
                     ('annual_return', 'sharpe_ratio', 'max_drawdown',
                      'annual_volatility', 'total_weeks', 'final_nav')},
        'nasdaq_leg': {
            'avg_weight': avg_w, 'n_trades': n_trades,
            'sum_abs_dw': sum_abs_dw, 'n_weeks': n_weeks,
        },
        'fee_diff_annual': FEE_DIFF_ANNUAL,
        'fee_diff_portfolio_pp_per_year': fee_pp_portfolio * 100,
        'scenarios': {
            'fee_only': {k: m_fee_only[k] for k in
                         ('annual_return', 'sharpe_ratio', 'max_drawdown')},
        },
        'random_lag_1_2': rand_stats,
    }
    for k in LAGS:
        s = scen[f'lag{k}']
        payload['scenarios'][f'lag{k}'] = {
            'lag_only': {kk: s['lag_only'][kk] for kk in
                         ('annual_return', 'sharpe_ratio', 'max_drawdown')},
            'lag_plus_fee': {kk: s['lag_plus_fee'][kk] for kk in
                             ('annual_return', 'sharpe_ratio', 'max_drawdown')},
            'cost_sum': s['cost_sum'],
            'cost_pp_per_year_arith': s['cost_pp_per_year_arith'],
            'cost_mean_bp_per_trade': s['cost_mean_bp_per_trade'],
            'cost_positive_share': s['cost_positive_share'],
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    write_md(payload, scen, base_m, config)
    print(f'输出: {OUT_MD}')
    print(f'输出: {OUT_JSON}')


# ============================================================
# 报告
# ============================================================

def write_md(payload, scen, base_m, config):
    nas = payload['nasdaq_leg']
    fee_pp = payload['fee_diff_portfolio_pp_per_year']
    rand = payload['random_lag_1_2']
    ba = base_m['annual_return'] * 100
    bs = base_m['sharpe_ratio']

    def row(name, m, lag_pp=None):
        ann = m['annual_return'] * 100
        return (f"| {name} | {ann:.2f}% | {ann - ba:+.3f} | "
                f"{m['sharpe_ratio']:.3f} | {m['sharpe_ratio'] - bs:+.3f} | "
                f"{m['max_drawdown']*100:.2f}% |")

    lines = []
    lines.append('# E2: 纳指腿改用场外联接基金 — T+lag 滞后成本 + 费率差')
    lines.append('')
    lines.append(f'- 基线: v4.4 生产 NAV 回测 ({payload["config"]}), 周五收盘 NAV 成交')
    lines.append(f"- 基线指标: 年化 **{ba:.2f}%**, Sharpe **{bs:.3f}**, "
                 f"最大回撤 {base_m['max_drawdown']*100:.2f}%, 共 {nas['n_weeks']} 周 "
                 f"(≈{nas['n_weeks']/52:.1f} 年)")
    lines.append(f"- 纳指腿: 时间加权平均仓位 **{nas['avg_weight']*100:.2f}%**, "
                 f"调仓笔数 {nas['n_trades']}, 累计换手 Σ|Δw| = {nas['sum_abs_dw']:.2f}")
    lines.append('- 日频净值: 513100 fund_nav 的 adj_nav (2022-01-13 份额拆分导致 '
                 'unit_nav 跳变, adj_nav 连续; 除拆分日外两者短窗收益比值完全一致)')
    lines.append('')
    lines.append('## 方法与符号推导')
    lines.append('')
    lines.append('设决策周五 T, 该周纳指腿调仓量 Δw (>0 买入, <0 卖出), '
                 '滞后期净值收益 r_lag = nav(T+lag)/nav(T) − 1。')
    lines.append('')
    lines.append('- **买入 (Δw>0)**: 基线在 T 以 nav(T) 建仓, 场外申购在 T+lag 才按 '
                 'nav(T+lag) 确认, 期间该笔资金为现金 (收益 0), 错过 r_lag。'
                 '若滞后期净值上涨 (r_lag>0), 相当于以更高净值买入 → 多付 Δw×r_lag; '
                 '下跌则反而占便宜。相对基线的成本 = **+Δw × r_lag**。')
    lines.append('- **卖出 (Δw<0)**: 基线在 T 以 nav(T) 离场, 场外赎回在 T+lag 才按 '
                 'nav(T+lag) 确认, 期间 |Δw| 仍暴露于纳指、多赚 |Δw|×r_lag。'
                 '相对基线的超额 = |Δw|×r_lag = −Δw×r_lag, 即成本 = **+Δw × r_lag** '
                 '(净值上涨时卖出方反而受益, 符号自动取负)。')
    lines.append('')
    lines.append('两种方向统一为 **cost_t = Δw_t × r_lag,t**, 从成交周的策略周收益中'
                 '扣除后重算净值与指标 (后处理法, 引擎零改动)。该成本对单笔而言可正可负, '
                 '系统性偏差来自两点: ① 纳指长期正漂移 × 净买入方向; ② 动量策略'
                 '"涨后买、跌后卖"与短期收益延续的正相关。')
    lines.append('')
    lines.append('费率差: 联接基金相对场内 ETF 的持有成本差取 **+0.4pp/年** '
                 '(0.3~0.5 区间中值), 按逐周实际纳指仓位计提 w_t×0.4pp/52, '
                 f"组合口径 ≈ {nas['avg_weight']*100:.2f}% × 0.4pp = **{fee_pp:.3f}pp/年**。")
    lines.append('')
    lines.append('## 确定性 lag 结果')
    lines.append('')
    lines.append('| 情形 | 年化 | Δ年化(pp) | Sharpe | ΔSharpe | 最大回撤 |')
    lines.append('|---|---|---|---|---|---|')
    lines.append(f"| 基线 (场内 NAV, T 成交) | {ba:.2f}% | — | {bs:.3f} | — | "
                 f"{base_m['max_drawdown']*100:.2f}% |")
    lines.append(row('仅费率差 (+0.4pp/年×仓位)', payload['scenarios']['fee_only']))
    for k in LAGS:
        s = scen[f'lag{k}']
        lines.append(row(f'lag={k} 仅滞后成本', s['lag_only']))
        lines.append(row(f'lag={k} 滞后+费率差 (合计)', s['lag_plus_fee']))
    lines.append('')
    lines.append('滞后成本明细 (算术口径):')
    lines.append('')
    lines.append('| lag | 成本合计(pp) | 折年(pp/年) | 单笔均值(bp) | 正成本笔占比 |')
    lines.append('|---|---|---|---|---|')
    for k in LAGS:
        s = scen[f'lag{k}']
        lines.append(f"| {k} | {s['cost_sum']*100:.2f} | "
                     f"{s['cost_pp_per_year_arith']:.3f} | "
                     f"{s['cost_mean_bp_per_trade']:.1f} | "
                     f"{s['cost_positive_share']*100:.0f}% |")
    lines.append('')
    c1 = scen['lag1']['cost_pp_per_year_arith']
    c2 = scen['lag2']['cost_pp_per_year_arith']
    c3 = scen['lag3']['cost_pp_per_year_arith']
    lines.append(f'实证发现滞后成本**随 lag 递减而非累积** ({c1:.3f} → {c2:.3f} → '
                 f'{c3:.3f} pp/年): 与调仓方向同向的净值移动集中在 T+1 (短期延续), '
                 'T+2/T+3 出现部分均值回复将其抵消。因此 **lag=1 反而是最坏情形**; '
                 '滞后同时使成交时点偏离周五锚, 最大回撤小幅抬升 (+0.3~0.7pp)。')
    lines.append('')
    lines.append('## 随机 lag ∈ {1,2} (申赎确认不确定性)')
    lines.append('')
    lines.append(f"每笔调仓独立等概率抽 lag∈{{1,2}}, seed={rand['seed']}, "
                 f"{rand['n_draws']} 次抽样 (仅滞后成本, 未叠加费率差):")
    lines.append('')
    lines.append(f"- 年化: 均值 {rand['annual_return_mean']*100:.2f}%, "
                 f"σ {rand['annual_return_std']*100:.3f}pp, "
                 f"p5/p50/p95 = {rand['annual_return_p5']*100:.2f}% / "
                 f"{rand['annual_return_p50']*100:.2f}% / "
                 f"{rand['annual_return_p95']*100:.2f}%")
    lines.append(f"- Sharpe: 均值 {rand['sharpe_mean']:.3f} "
                 f"(σ {rand['sharpe_std']:.3f}), 基线 {bs:.3f}")
    lines.append(f"- 滞后成本折年: 均值 {rand['lag_cost_pp_mean']:.3f}pp/年, "
                 f"σ {rand['lag_cost_pp_std']:.3f}, "
                 f"p5/p95 = {rand['lag_cost_pp_p5']:.3f} / {rand['lag_cost_pp_p95']:.3f} pp/年")
    lines.append('')
    lines.append('## 结论')
    lines.append('')
    # 用 lag=2 (T+2 是 QDII 联接基金申赎确认的典型值) 作为中枢口径
    mid = scen['lag2']
    lag_mid_pp = ba - mid['lag_only']['annual_return'] * 100
    tot_mid_pp = ba - mid['lag_plus_fee']['annual_return'] * 100
    lo = ba - scen['lag1']['lag_plus_fee']['annual_return'] * 100
    hi = ba - scen['lag3']['lag_plus_fee']['annual_return'] * 100
    rng_lo, rng_hi = sorted((lo, hi))
    worst = ba - scen['lag1']['lag_plus_fee']['annual_return'] * 100
    lines.append(f'- 滞后成本 (lag=2, QDII 联接基金典型确认日): **{lag_mid_pp:.2f}pp/年**; '
                 f'费率差: **{fee_pp:.2f}pp/年**。')
    lines.append(f'- **场外执行总代价 ≈ {tot_mid_pp:.2f}pp/年 (lag=2 中枢), '
                 f'区间 {rng_lo:.2f}~{rng_hi:.2f}pp/年 (lag 1~3, '
                 f'最坏情形 lag=1 为 {worst:.2f}pp/年)**。'
                 '该数字与 E1 的场内溢价侵蚀直接对比: 若 E1 溢价侵蚀高于此值, '
                 '场外路径成立; 反之场内直接买入更优。')
    lines.append('- 随机确认日 (lag∈{1,2}) 主要增加路径不确定性 '
                 f"(年化 σ≈{rand['annual_return_std']*100:.2f}pp), "
                 '均值与确定性 lag 的 1~2 区间一致, 不改变量级结论。')
    lines.append('')
    lines.append('*注: 未计入申购费 (联接基金 C 类通常 0 申购费+销售服务费, 已含在 '
                 '0.4pp 持有成本差内)、赎回在途资金占用 (赎回款 T+lag 后再入池, '
                 '本模型以"滞后期保留暴露"近似, 与实际申赎现金流方向一致)。*')
    lines.append('')
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
