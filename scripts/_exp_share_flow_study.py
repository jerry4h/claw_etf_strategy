#!/usr/bin/env python3
"""
份额资金流因子研究 (E0 数据体检 + E1 信号质量评估)
==================================================
研究问题: ETF 份额增长（申赎净流入的代理变量）能否作为策略池 5 只 ETF
          横截面轮动打分的增量因子?

与前序研究 exp_national_team_signal 的区别:
  - 前序: 全市场宽基 ETF 聚合份额信号 → 单标的(510500)时序 IC, 结论 NO-GO
  - 本次: 策略池自身 5 只 ETF 的份额增长 → 横截面 rank IC (对应 L1 打分层)

E0: 覆盖率 / 本地文件新鲜度 / 精确倍数跳变(拆分折算) / 同值停滞率
E1: 横截面 rank_IC / IR / t-stat / 与 mom6+vol14 正交性
    门禁: |IC| >= 0.03 且 |t-stat| >= 1.5

用法:
  .venv/bin/python scripts/_exp_share_flow_study.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.factors import calculate_momentum, calculate_volatility_tapered  # noqa: E402

NAV_FILE = PROJECT / 'data' / 'all_etfs_nav_latest.csv'
# 策略池 5 只已纳入 config/national_team_etfs.yaml 的 strategy_pool 采集清单，
# 由 fetch_national_team_share.py 随 weekly_refresh 一起刷新，因此直接读该目录。
SHARE_DIR = PROJECT / 'data' / 'national_team' / 'fund_share'
OUT_DIR = PROJECT / 'output' / 'experiments'

# 策略池 5 只 ETF → 份额文件名
CODE_MAP = {
    '纳指ETF': '513100_SH',
    '红利低波ETF': '512890_SH',
    '中证500ETF': '510500_SH',
    '黄金ETF': '518880_SH',
    '国债ETF': '511010_SH',
}

IC_GATE = 0.03
T_GATE = 1.5
ORTHO_GATE = 0.30


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def load_weekly_nav() -> pd.DataFrame:
    """周频净值 (index=周五, columns=ETF)。"""
    df = pd.read_csv(NAV_FILE, index_col=0, parse_dates=True)
    df.index.name = 'date'
    return df


def load_share_raw() -> dict[str, pd.Series]:
    """原始日频份额序列 (未做任何修正)。"""
    out = {}
    for name, fname in CODE_MAP.items():
        path = SHARE_DIR / f'{fname}.csv'
        if not path.exists():
            print(f'  ⚠️  {name}: 份额文件缺失 {fname}.csv')
            continue
        d = pd.read_csv(path, dtype={'trade_date': str})
        d['dt'] = pd.to_datetime(d['trade_date'], format='%Y%m%d')
        s = d.set_index('dt')['fd_share'].sort_index()
        s = s[~s.index.duplicated(keep='last')]
        out[name] = s
    return out


# ----------------------------------------------------------------------
# E0: 数据质量体检
# ----------------------------------------------------------------------
def detect_ratio_jumps(s: pd.Series, tol: float = 0.005) -> list[dict]:
    """检测精确整数倍/整数分之一跳变 —— 份额拆分或计量单位切换的指纹。

    真实申赎不会出现 ratio 精确等于 2.000 / 5.000 / 0.010 的情况。
    """
    ratio = s / s.shift(1)
    hits = []
    for dt, r in ratio.dropna().items():
        if r <= 0 or not np.isfinite(r):
            continue
        for cand in (r, 1.0 / r):
            near = round(cand)
            if near >= 2 and abs(cand - near) / near < tol:
                hits.append({
                    'date': dt.strftime('%Y-%m-%d'),
                    'ratio': round(float(r), 6),
                    'factor': int(near),
                    'kind': '拆分/单位放大' if r > 1 else '折算/单位缩小',
                    'pct_change': round(float(r - 1) * 100, 2),
                })
                break
    return hits


def stagnation_runs(s: pd.Series, since: str = '2016-01-01') -> list[dict]:
    """份额连续不变的段落 (前值填充/断更的指纹)。"""
    sub = s[s.index >= since]
    if len(sub) < 2:
        return []
    grp = (sub.diff() != 0).cumsum()
    agg = sub.groupby(grp).agg(start=lambda x: x.index.min(),
                               end=lambda x: x.index.max(), n='size', val='first')
    agg = agg[agg['n'] >= 20].sort_values('n', ascending=False)
    return [{'start': r.start.strftime('%Y-%m-%d'), 'end': r.end.strftime('%Y-%m-%d'),
             'n_days': int(r.n), 'frozen_value': float(r.val)}
            for r in agg.head(5).itertuples()]


def weekend_rows(s: pd.Series) -> list[str]:
    """落在周六/周日的 trade_date —— 季报快照被混入日频序列的指纹。"""
    return [d.strftime('%Y-%m-%d') for d in s.index[s.index.dayofweek >= 5]]


def e0_quality(share_raw: dict[str, pd.Series], nav_last: pd.Timestamp) -> dict:
    """E0 体检: 覆盖率 / 滞后 / 跳变 / 停滞率。"""
    print('\n' + '=' * 70)
    print(' E0  份额数据质量体检')
    print('=' * 70)
    report = {}
    for name, s in share_raw.items():
        gap = s.index.to_series().diff().dt.days
        recent = s[s.index >= '2020-01-01']
        unchanged = float((recent.diff() == 0).mean()) if len(recent) > 1 else float('nan')
        jumps = detect_ratio_jumps(s)
        pct = s.pct_change()
        lag_days = int((nav_last - s.index.max()).days)
        rec = {
            'n_obs': int(len(s)),
            'start': s.index.min().strftime('%Y-%m-%d'),
            'end': s.index.max().strftime('%Y-%m-%d'),
            'lag_vs_nav_days': lag_days,
            'gap_median_days': float(gap.median()),
            'gap_max_days': float(gap.max()),
            'unchanged_rate_2020plus': round(unchanged, 4),
            'n_jump_gt50pct': int((pct.abs() > 0.5).sum()),
            'n_jump_gt20pct': int((pct.abs() > 0.2).sum()),
            'exact_ratio_jumps': jumps,
            'stagnation_runs': stagnation_runs(s),
            'weekend_rows': weekend_rows(s),
        }
        report[name] = rec
        print(f'\n  {name}')
        print(f'    覆盖 {rec["start"]} ~ {rec["end"]}  n={rec["n_obs"]}  '
              f'滞后NAV {lag_days} 天')
        print(f'    gap 中位={rec["gap_median_days"]:.0f}d 最大={rec["gap_max_days"]:.0f}d  '
              f'2020+同值停滞率={unchanged * 100:.1f}%')
        print(f'    |Δ|>50%: {rec["n_jump_gt50pct"]}  |Δ|>20%: {rec["n_jump_gt20pct"]}')
        if jumps:
            for j in jumps:
                print(f'    🔴 精确倍数跳变 {j["date"]}: ×{j["ratio"]:.4f} '
                      f'({j["factor"]}倍, {j["kind"]}) → 判定为非申赎事件')
        else:
            print('    ✅ 无精确倍数跳变')
        for r in rec['stagnation_runs']:
            print(f'    🔴 份额冻结 {r["start"]}~{r["end"]} 共 {r["n_days"]} 个交易日 '
                  f'恒为 {r["frozen_value"]:.2f}')
        if rec['weekend_rows']:
            print(f'    🔴 {len(rec["weekend_rows"])} 行 trade_date 落在周末('
                  f'季末/年末快照): {rec["weekend_rows"][:4]}...')
    return report


def adjust_splits(share_raw: dict[str, pd.Series], e0: dict) -> dict[str, pd.Series]:
    """按检测到的精确倍数跳变做后向复权，使份额序列在申赎意义上连续。

    后向复权不引入前视: 跳变前的整段仅被常数缩放, 段内 pct_change 不变;
    跨越跳变点的窗口在 t > 跳变日 时才被使用, 此时该事件已公开。
    """
    out = {}
    for name, s in share_raw.items():
        jumps = e0[name]['exact_ratio_jumps']
        adj = s.copy().astype(float)
        for j in sorted(jumps, key=lambda x: x['date'], reverse=True):
            dt = pd.Timestamp(j['date'])
            r = j['ratio']
            adj.loc[adj.index < dt] = adj.loc[adj.index < dt] * r
        out[name] = adj
    return out


# ----------------------------------------------------------------------
# 因子构造
# ----------------------------------------------------------------------
def build_share_panel(share: dict[str, pd.Series], weekly_index: pd.DatetimeIndex,
                      lag_steps: int) -> pd.DataFrame:
    """asof 对齐到周五锚点。

    lag_steps = 回退的**交易日**个数（非日历天）:
      0 = 用周五当日份额——实测 tushare fund_share 周五当天即可得,
          周一调仓完全能用上, 这是现实口径
      1 = 退一步用周四份额, 作为保守旁证
    """
    cols = {}
    for name, s in share.items():
        vals = []
        for dt in weekly_index:
            hist = s[s.index <= dt]
            idx = len(hist) - 1 - lag_steps
            vals.append(hist.iloc[idx] if idx >= 0 else np.nan)
        cols[name] = vals
    return pd.DataFrame(cols, index=weekly_index)


def share_growth(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """window 周份额增长率。"""
    return panel / panel.shift(window) - 1.0


# ----------------------------------------------------------------------
# E1: 横截面 rank IC
# ----------------------------------------------------------------------
def cross_sectional_ic(factor: pd.DataFrame, next_ret: pd.DataFrame,
                       min_assets: int = 3) -> dict:
    """逐周横截面 Spearman rank IC, 再对时序做 t 检验。"""
    ics = []
    dates = []
    for dt in factor.index:
        f = factor.loc[dt]
        r = next_ret.loc[dt] if dt in next_ret.index else None
        if r is None:
            continue
        mask = f.notna() & r.notna()
        if mask.sum() < min_assets:
            continue
        if f[mask].nunique() < 2 or r[mask].nunique() < 2:
            continue
        ic, _ = stats.spearmanr(f[mask], r[mask])
        if np.isfinite(ic):
            ics.append(float(ic))
            dates.append(dt)
    if len(ics) < 20:
        return {'n_weeks': len(ics), 'mean_ic': float('nan'), 'ir': float('nan'),
                't_stat': float('nan'), 'p_value': float('nan'),
                'pct_positive': float('nan'), 'period': 'n/a'}
    arr = np.array(ics)
    t_stat, p_val = stats.ttest_1samp(arr, 0.0)
    return {
        'n_weeks': len(arr),
        'mean_ic': round(float(arr.mean()), 4),
        'std_ic': round(float(arr.std(ddof=1)), 4),
        'ir': round(float(arr.mean() / arr.std(ddof=1)), 4) if arr.std(ddof=1) > 0 else float('nan'),
        't_stat': round(float(t_stat), 3),
        'p_value': round(float(p_val), 4),
        'pct_positive': round(float((arr > 0).mean()), 4),
        'period': f'{dates[0].date()}~{dates[-1].date()}',
    }


def orthogonality(factor: pd.DataFrame, mom: pd.DataFrame,
                  vol: pd.DataFrame) -> dict:
    """与现有因子的横截面相关性（逐周算 Spearman 后取均值）。"""
    res = {}
    for label, other in (('momentum6', mom), ('vol_tapered14', vol)):
        cs = []
        for dt in factor.index:
            if dt not in other.index:
                continue
            a, b = factor.loc[dt], other.loc[dt]
            mask = a.notna() & b.notna()
            if mask.sum() < 3 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
                continue
            c, _ = stats.spearmanr(a[mask], b[mask])
            if np.isfinite(c):
                cs.append(float(c))
        res[label] = round(float(np.mean(cs)), 4) if cs else float('nan')
    return res


HEALTHY = ['红利低波ETF', '中证500ETF', '黄金ETF']  # E0 体检后数据可用的子集


def healthy_subset_ic(share_adj: dict[str, pd.Series], nav: pd.DataFrame) -> dict:
    """判别实验: 剔除数据损坏标的，仅用健康 3 只算横截面 IC。

    目的是区分两种归因: “因子本身无效” vs “仅因脏数据拖累”。
    若健康子集 IC 仍远低于门禁, 则修数据也救不回来。
    """
    sub_share = {k: v for k, v in share_adj.items() if k in HEALTHY}
    next_ret = nav[HEALTHY].pct_change().shift(-1)
    out = {}
    print('\n  -- 判别实验: 仅健康 3 只 (剔除 513100 冻结 / 511010 口径混乱), lag=0 --')
    for w in (4, 8, 13):
        panel = build_share_panel(sub_share, nav.index, lag_steps=0)
        fac = share_growth(panel, w)
        ic = cross_sectional_ic(fac, next_ret)
        out[f'healthy3_w{w}_lag0'] = ic
        print(f'  {"healthy3_w%d_lag0" % w:24s} {ic["n_weeks"]:>5d} {ic["mean_ic"]:>+9.4f} '
              f'{ic["ir"]:>+7.3f} {ic["t_stat"]:>+7.2f} {ic["p_value"]:>7.4f} '
              f'{ic["pct_positive"]:>6.1%}')
    return out


CONCLUSIONS = [
    '## 归因与结论',
    '',
    '### 1) 513100 份额冻结是真实经济现象, 不是数据缺陷',
    '',
    '证据链 (2022+ 日频对齐溢价):',
    '',
    '- 份额冻结日均溢价 3.61% / 中位 2.95% / 最大 13.90% (n=533)',
    '- 份额变动日均溢价 1.32% / 中位 1.24% / 最大 8.98% (n=573)',
    '- 年度份额扩张递减: 2023 +28.2% → 2024 +11.9% → 2025 -0.35% → 2026 -0.01%,',
    '  同期年均溢价 0.77% → 2.26% → 3.44% → 6.47%',
    '- 变动步长量子化为 {+500, +300, -200, -3000} 而非连续值',
    '',
    '→ QDII 额度用尽→申购受限→份额停滞→场内溢价被推高。因此 513100 的份额',
    '反映的是**额度供给约束**而非资金意愿, 结构上就不是资金流代理变量——',
    '此项修数据也无法解决。',
    '',
    '### 2) 因子本身弱, 非脏数据也非披露滞后拖累',
    '',
    '判别实验剔除 513100(冻结) 与 511010(口径混乱) 后, 健康 3 只子集最优仍仅',
    '|IC|=0.0170, t=0.47。全样本最优 adj_w8_lag0 也仅 |IC|=0.0205, t=0.97。',
    '全部 25 个变体 |t| < 1.0。',
    '',
    '### ⚠️ 更正: “披露滞后 7 天”是误判',
    '',
    '初版结论曾称份额披露滞后 7 天、构成实盘硬伤。实测推翻:',
    '',
    '- 直接查 tushare fund_share, 5 只全部有 20260807(周五当日) 数据, **零滞后**',
    '- 本地 4/5 只文件停在 20260731, 真因是它们不在 national_team 采集清单内',
    '  (当时 config/national_team_etfs.yaml 仅含 510500), 属于临时抓取的孤儿文件',
    '- 全历史重拉对比: 共有行 0 不一致, 仅缺末尾 5 个交易日 → E1 样本未被损坏',
    '- 已修复: 新增 yaml `strategy_pool` 键 + fetch 脚本默认追加该清单,',
    '  5 只现均随 weekly_refresh 刷新至最新交易日',
    '',
    '因此门禁改以 lag=0(周五当日份额, 周一调仓可用)为现实口径, lag=1 仅作保守旁证。',
    '结论方向未变(仍 NO-GO), 但“实盘硬伤”这条论据作废。',
    '',
    '### 3) 正交性合格但无价值',
    '',
    'corr(mom6) ≈ -0.045, corr(vol14) ≈ +0.074, 远低于 0.30 阀值——份额增长确实是',
    '独立信息维度, 但是**独立的噪声**而非独立的信息。',
    '',
    '### 4) 副产品: SOP §6.2 R2 份额预警在额度管制期结构失效',
    '',
    'R2 以“份额 5 日变动 > +5%”作为溢价回落预警。但 513100 在额度用尽期份额',
    '恒为 0 变动, R2 永远不会触发——而这正是溢价最高的时段。建议将“份额冻结”',
    '本身作为溢价持续高位的伴随指标纳入哨兵观察项, 而非用作 L1 打分因子。',
    '',
    '### 与前序研究的关系',
    '',
    'exp_national_team_signal (2026-08-04) 以“全市场聚合份额 → 510500 时序 IC”得到',
    'IC=0.0151 NO-GO。本研究改用“策略池自身 5 只 → 横截面 rank IC”这条正交路径,',
    '独立得到同量级结论 (|IC| ≤ 0.021)。两条路径互相印证。',
    '',
]


def write_report(payload: dict) -> Path:
    """生成 markdown 调研报告。"""
    e0, ic, gate = payload['e0_quality'], payload['e1_ic'], payload['gate']
    L = [
        '# 份额资金流因子调研 (E0 数据体检 + E1 信号评估)',
        '', f'生成时间: {payload["generated_at"]}  |  NAV 截至: {payload["nav_last"]}',
        '', '---', '', '## 门禁判定', '',
        f'**{gate["verdict"]}**', '',
    ]
    L += [f'- {e}' for e in gate['evidence']]
    L += ['', '---', '', '## E0 数据质量清单', '',
          '| ETF | 覆盖 | n | 滞后NAV | 2020+停滞率 | 精确倍数跳变 | 最长冻结段 | 周末行 |',
          '|---|---|---|---|---|---|---|---|']
    for name, r in e0.items():
        jm = ('; '.join(f'{j["date"]} ×{j["factor"]}' for j in r['exact_ratio_jumps'])
              or '无')
        st = (f'{r["stagnation_runs"][0]["n_days"]}日 '
              f'({r["stagnation_runs"][0]["start"]}~{r["stagnation_runs"][0]["end"]})'
              if r['stagnation_runs'] else '无')
        L.append(f'| {name} | {r["start"]}~{r["end"]} | {r["n_obs"]} | '
                 f'{r["lag_vs_nav_days"]}d | {r["unchanged_rate_2020plus"]:.1%} | '
                 f'{jm} | {st} | {len(r["weekend_rows"])} |')
    L += ['', '---', '', '## E1 横截面 rank IC', '',
          '| 变体 | 周数 | mean_IC | IR | t | p | IC>0 | corr(mom6) | corr(vol14) |',
          '|---|---|---|---|---|---|---|---|---|']
    for k, v in ic.items():
        if not np.isfinite(v['mean_ic']):
            continue
        o = v.get('orthogonality', {})
        L.append(f'| {k} | {v["n_weeks"]} | {v["mean_ic"]:+.4f} | {v["ir"]:+.3f} | '
                 f'{v["t_stat"]:+.2f} | {v["p_value"]:.4f} | {v["pct_positive"]:.1%} | '
                 f'{o.get("momentum6", float("nan")):+.3f} | '
                 f'{o.get("vol_tapered14", float("nan")):+.3f} |')
    L += ['', '---', '',
          f'门禁阀值: |IC| >= {gate["thresholds"]["abs_ic"]}, '
          f'|t| >= {gate["thresholds"]["abs_t"]}, '
          f'正交性 < {gate["thresholds"]["ortho"]}', '', '---', '']
    L += CONCLUSIONS
    path = OUT_DIR / 'exp_share_flow.md'
    path.write_text('\n'.join(L), encoding='utf-8')
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nav = load_weekly_nav()
    share_raw = load_share_raw()
    if not share_raw:
        print('❌ 无份额数据, 终止')
        return 1

    e0 = e0_quality(share_raw, nav.index.max())
    share_adj = adjust_splits(share_raw, e0)

    # 现有因子（口径对齐 v4.3 生产: mom_window=6, tapered_vol 14+7）
    mom = calculate_momentum(nav, window=6)
    vol = calculate_volatility_tapered(nav, window=14, taper=7)
    next_ret = nav.pct_change().shift(-1)

    print('\n' + '=' * 70)
    print(' E1  横截面 rank IC 评估')
    print('=' * 70)
    print('\n  变体命名: {raw|adj}_w{窗口}_lag{回退交易日数}  (lag0=周五当日份额, 现实口径)')
    print(f'  {"变体":24s} {"周数":>5s} {"mean_IC":>9s} {"IR":>7s} {"t":>7s} '
          f'{"p":>7s} {"IC>0":>6s}')
    print('  ' + '-' * 68)

    results = {}
    for tag, panel_src in (('raw', share_raw), ('adj', share_adj)):
        for lag in (0, 1):
            panel = build_share_panel(panel_src, nav.index, lag_steps=lag)
            for w in (2, 4, 8, 13):
                fac = share_growth(panel, w)
                key = f'{tag}_w{w}_lag{lag}'
                ic = cross_sectional_ic(fac, next_ret)
                ic['orthogonality'] = orthogonality(fac, mom, vol)
                results[key] = ic
                if np.isfinite(ic['mean_ic']):
                    print(f'  {key:24s} {ic["n_weeks"]:>5d} {ic["mean_ic"]:>+9.4f} '
                          f'{ic["ir"]:>+7.3f} {ic["t_stat"]:>+7.2f} {ic["p_value"]:>7.4f} '
                          f'{ic["pct_positive"]:>6.1%}')
                else:
                    print(f'  {key:24s} {ic["n_weeks"]:>5d}  样本不足')

    # 2016+ 子样本（E0 显示 2013-2015 覆盖稀疏）
    print('\n  -- 2016+ 子样本 (剔除 2013-2015 稀疏披露区) --')
    sub_idx = nav.index[nav.index >= '2016-01-01']
    sub_results = {}
    for lag in (0, 1):
        panel = build_share_panel(share_adj, sub_idx, lag_steps=lag)
        for w in (4, 8, 13):
            fac = share_growth(panel, w)
            key = f'adj2016_w{w}_lag{lag}'
            ic = cross_sectional_ic(fac, next_ret.loc[sub_idx])
            ic['orthogonality'] = orthogonality(fac, mom.loc[sub_idx], vol.loc[sub_idx])
            sub_results[key] = ic
            print(f'  {key:24s} {ic["n_weeks"]:>5d} {ic["mean_ic"]:>+9.4f} '
                  f'{ic["ir"]:>+7.3f} {ic["t_stat"]:>+7.2f} {ic["p_value"]:>7.4f} '
                  f'{ic["pct_positive"]:>6.1%}')
    results.update(sub_results)

    # ---- 判别实验: 健康子集 ----
    results.update(healthy_subset_ic(share_adj, nav))

    # ---- 门禁判定: 取 |IC| 最大的现实口径变体(lag=0, 即周五当日份额) ----
    honest = {k: v for k, v in results.items()
              if k.endswith('lag0') and np.isfinite(v['mean_ic'])}
    best_key = max(honest, key=lambda k: abs(honest[k]['mean_ic'])) if honest else None
    verdict, evidence = 'NO-GO', []
    if best_key:
        b = honest[best_key]
        ic_ok = abs(b['mean_ic']) >= IC_GATE
        t_ok = abs(b['t_stat']) >= T_GATE
        o = b.get('orthogonality', {})
        o_ok = all(abs(v) < ORTHO_GATE for v in o.values() if np.isfinite(v))
        evidence = [
            f'最佳现实口径变体(lag=0) = {best_key}',
            f'|mean_IC| = {abs(b["mean_ic"]):.4f} {">=" if ic_ok else "<"} {IC_GATE} '
            f'→ {"PASS" if ic_ok else "FAIL"}',
            f'|t-stat| = {abs(b["t_stat"]):.2f} {">=" if t_ok else "<"} {T_GATE} '
            f'→ {"PASS" if t_ok else "FAIL"}',
            ('正交性 ' + ' '.join(f'corr({k})={v:+.3f}' for k, v in o.items())
             + f' → {"PASS" if o_ok else "FAIL"}') if o else '正交性 未计算(健康子集变体)',
        ]
        verdict = 'GO' if (ic_ok and t_ok and o_ok) else 'NO-GO'

    print('\n' + '=' * 70)
    print(f' E1 门禁判定: {verdict}')
    print('=' * 70)
    for e in evidence:
        print(f'  - {e}')

    payload = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'nav_last': nav.index.max().strftime('%Y-%m-%d'),
        'e0_quality': e0,
        'e1_ic': results,
        'gate': {'verdict': verdict, 'best_variant': best_key, 'evidence': evidence,
                 'thresholds': {'abs_ic': IC_GATE, 'abs_t': T_GATE, 'ortho': ORTHO_GATE}},
    }
    (OUT_DIR / 'exp_share_flow.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    md = write_report(payload)
    print(f'\n✅ JSON 已写入 {OUT_DIR / "exp_share_flow.json"}')
    print(f'✅ 报告已写入 {md}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
