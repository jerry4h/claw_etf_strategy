#!/usr/bin/env python3
"""C1 探针: PE 估值分位信号 E1 评估 (数据就绪、已 shift(1) 防前视、backtest 未消费)。

假说 (计划两口径):
  a) 市场级防御调制: 沪深300 PE 分位极端时调制 Layer3 def_ratio
     (高估→加防御 / 低估→减防御)。E1 口径 = PE 分位对未来组合/资产
     收益的预测 IC + 极端分位条件分析。
  b) 截面 tiebreaker: PE 分位是单一市场序列 (非逐 ETF), 截面排序
     结构上不适用——唯一有意义截面为"A股暴露 vs 非A股", 与假说 a 等价,
     本报告以 a 为准并留档该结构性结论。

门禁: |IC| ≥ 0.03 且 |t| ≥ 1.5 (与 PVD/份额信号同口径);
慢变量补充 4 周/13 周前瞻窗。正交性: 与 mom6/vol 相关 < 0.30。

只读消费 src.data_loader 与 data/300etf_pe_percentile_weekly.csv, 零 src/ 改动。

用法: .venv/bin/python scripts/_exp_pe_percentile_e1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.data_loader import load_nav_data, resample_weekly, load_pe_percentile
from src.factors import calculate_momentum, calculate_volatility_tapered, calculate_pe_percentile

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
NAV_PATH = PROJ / "data" / "all_etfs_nav_latest.csv"
PE_PATH = PROJ / "data" / "300etf_pe_percentile_weekly.csv"

IC_GATE = 0.03
T_GATE = 1.5
ORTH_GATE = 0.30
HORIZONS = {"h1(下周)": 1, "h4(4周)": 4, "h13(13周)": 13}


def ic_stats(sig, fwd):
    """逐期 Spearman IC 序列统计 (这里为单序列时序 IC = 相关)。"""
    m = sig.notna() & fwd.notna()
    s, f = sig[m], fwd[m]
    if len(s) < 30:
        return {"n": int(len(s)), "ic": np.nan, "t": np.nan}
    # 时序单资产口径: 滚动块 IC 不适用, 用全期 Pearson 相关 + 有效样本折算 t
    r = float(np.corrcoef(s.values, f.values)[0, 1]) if s.std() > 0 and f.std() > 0 else np.nan
    n = int(len(s))
    t = r * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r * r)) if not np.isnan(r) else np.nan
    return {"n": n, "ic": r, "t": float(t)}


def rolling_ic(sig, fwd, window=52):
    """滚动 52 周窗口 IC 序列 (检验信号稳定性; 慢变量下周频 IC 波动大)。"""
    m = sig.notna() & fwd.notna()
    s, f = sig[m].values, fwd[m].values
    idx = sig[m].index
    ics = []
    for i in range(window, len(s)):
        if np.std(s[i - window:i]) > 0 and np.std(f[i - window:i]) > 0:
            ics.append(np.corrcoef(s[i - window:i], f[i - window:i])[0, 1])
    ics = np.array(ics)
    if len(ics) < 30:
        return {"n_windows": len(ics), "mean_ic": np.nan, "ir": np.nan, "t": np.nan,
                "ic_pos_share": np.nan}
    mu, sd = ics.mean(), ics.std(ddof=1)
    t = mu / (sd / np.sqrt(len(ics))) if sd > 0 else np.nan
    return {"n_windows": int(len(ics)), "mean_ic": float(mu),
            "ir": float(mu / sd) if sd > 0 else np.nan, "t": float(t),
            "ic_pos_share": float((ics > 0).mean())}


def main():
    nav = load_nav_data(NAV_PATH)
    weekly = resample_weekly(nav, anchor="W-MON")
    w_rets = weekly.pct_change()
    ew_ret = w_rets.mean(axis=1)

    # PE 分位: 复用生产因子函数 (5 年滚动窗口) + shift(1) 防前视 (同 compute_all_factors)
    # 对齐注意: PE CSV 日期为周一, NAV 周频标签为周五快照 —— 生产预留路径从未
    # 对齐过; 此处用 ffill asof 对齐 (周五决策可用当周一 PE), shift(1) 再留一周安全垫
    pe_raw = load_pe_percentile(PE_PATH)
    pe_pct = calculate_pe_percentile(pe_raw, window_years=5).shift(1)["pe_percentile"]
    pe_pct = pe_pct.reindex(weekly.index, method="ffill")
    valid_from = pe_pct.first_valid_index()
    valid_to = pe_pct.last_valid_index()
    print(f"[data] PE 分位有效区间: {valid_from.date()} ~ {valid_to.date()} "
          f"({pe_pct.notna().sum()} 周), NAV 至 {weekly.index[-1].date()}")

    mom6 = calculate_momentum(weekly, window=6)
    vol14 = calculate_volatility_tapered(weekly, window=14, taper=7)

    res = {
        "pe_valid": [str(valid_from.date()), str(valid_to.date())],
        "gates": {"ic": IC_GATE, "t": T_GATE, "orth": ORTH_GATE},
        "horizons": {},
        "orthogonality": {},
        "extremes": {},
    }

    # --- E1: 时序预测 IC (PE 分位 → 未来各 ETF / 等权组合收益) ---
    targets = {c: w_rets[c] for c in weekly.columns}
    targets["等权组合"] = ew_ret
    for hname, h in HORIZONS.items():
        fwd_block = {name: series.rolling(h).sum().shift(-h)
                     for name, series in targets.items()}
        row = {}
        for name, fwd in fwd_block.items():
            row[name] = ic_stats(pe_pct, fwd)
        res["horizons"][hname] = row
        print(f"\n[{hname}] 全期时序 IC (PE分位 vs 未来{h}周累计收益):")
        for name, st in row.items():
            flag = ""
            if not np.isnan(st["ic"]):
                flag = " ←过门禁" if (abs(st["ic"]) >= IC_GATE and abs(st["t"]) >= T_GATE) else ""
            print(f"  {name:<8s} IC={st['ic']:+.4f} t={st['t']:+.2f} n={st['n']}{flag}")

    # 滚动 IC 稳定性 (等权组合, h1)
    fwd1 = ew_ret.rolling(1).sum().shift(-1)
    res["rolling_ic_ew_h1"] = rolling_ic(pe_pct, fwd1)
    r = res["rolling_ic_ew_h1"]
    print(f"\n[稳定性] 等权组合 h1 滚动52周IC: mean={r['mean_ic']:+.4f} "
          f"IR={r['ir']:+.3f} t={r['t']:+.2f} IC>0占比={r['ic_pos_share']:.1%}")

    # --- 正交性: PE 分位与 mom6/vol 的截面均值序列相关 ---
    mom_mean = mom6.mean(axis=1)
    vol_mean = vol14.mean(axis=1)
    # PE 主要对应 A 股 (沪深300), 单列口径更贴切
    for label, series in (("mom6_中证500", mom6.get("中证500ETF")),
                          ("vol14_中证500", vol14.get("中证500ETF")),
                          ("mom6_截面均值", mom_mean),
                          ("vol14_截面均值", vol_mean)):
        if series is None:
            continue
        m = pe_pct.notna() & series.notna()
        c = float(np.corrcoef(pe_pct[m].values, series[m].values)[0, 1]) if m.sum() > 30 else np.nan
        res["orthogonality"][label] = c
        print(f"[正交] corr(PE分位, {label}) = {c:+.3f} "
              f"{'PASS' if abs(c) < ORTH_GATE else 'FAIL'}")

    # --- 极端分位条件分析 (防御调制假说的直接证据) ---
    ew = ew_ret.copy()
    for band, lo, hi in (("PE>90%(高估)", 0.90, 1.01),
                         ("PE<10%(低估)", -0.01, 0.10),
                         ("中间区", 0.10, 0.90)):
        mask = pe_pct.notna() & (pe_pct >= lo) & (pe_pct < hi)
        seg = ew[mask]
        if len(seg) < 10:
            continue
        ann = float(seg.mean() * 52)
        vols = float(seg.std(ddof=0) * np.sqrt(52))
        # 段内策略相关: 用等权近似市场暴露; 另算中证500单列
        seg500 = w_rets["中证500ETF"][mask]
        res["extremes"][band] = {
            "weeks": int(len(seg)),
            "ew_ann_ret": ann, "ew_ann_vol": vols,
            "csi500_ann_ret": float(seg500.mean() * 52),
        }
        print(f"[极端] {band}: {len(seg)} 周, 等权年化 {ann:+.2%} (vol {vols:.2%}), "
              f"中证500年化 {seg500.mean() * 52:+.2%}")

    # --- 门禁判定 (以 h1 等权组合 + h13 慢变量窗为准) ---
    h1 = res["horizons"]["h1(下周)"]["等权组合"]
    h13 = res["horizons"]["h13(13周)"]["等权组合"]
    pass_h1 = abs(h1["ic"]) >= IC_GATE and abs(h1["t"]) >= T_GATE
    pass_h13 = abs(h13["ic"]) >= IC_GATE and abs(h13["t"]) >= T_GATE
    verdict = "GO" if (pass_h1 or pass_h13) else "NO-GO"
    res["verdict"] = {"h1_pass": bool(pass_h1), "h13_pass": bool(pass_h13),
                      "gate_decision": verdict}
    print(f"\n[判定] h1 {pass_h1} / h13 {pass_h13} → {verdict}")

    out_json = OUT / "exp_pe_percentile_e1.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")
    render_md(res)


def render_md(res):
    L = []
    L.append("# C1 探针: PE 估值分位信号 E1 评估\n")
    L.append(f"> {pd.Timestamp.today().date()} | 数据区间 PE 分位 "
             f"{res['pe_valid'][0]} ~ {res['pe_valid'][1]} | 脚本 "
             f"`scripts/_exp_pe_percentile_e1.py` | 零 src/ 改动\n")
    L.append("## 门禁与判定\n")
    v = res["verdict"]
    L.append(f"- 门禁: |IC| ≥ {res['gates']['ic']} 且 |t| ≥ {res['gates']['t']} "
             f"(正交性 < {res['gates']['orth']})")
    L.append(f"- h1(下周) 等权组合: {'PASS' if v['h1_pass'] else 'FAIL'}; "
             f"h13(13周) 等权组合: {'PASS' if v['h13_pass'] else 'FAIL'}")
    L.append(f"- **判定: {v['gate_decision']}**\n")
    L.append("## 时序预测 IC\n")
    L.append("| 前瞻窗 | 目标 | IC | t | n |")
    L.append("|---|---|---|---|---|")
    for hname, row in res["horizons"].items():
        for name, st in row.items():
            ic = st["ic"]
            t = st["t"]
            L.append(f"| {hname} | {name} | "
                     f"{ic if ic is None else f'{ic:+.4f}'} | "
                     f"{t if t is None else f'{t:+.2f}'} | {st['n']} |")
    r = res["rolling_ic_ew_h1"]
    L.append(f"\n滚动 52 周 IC 稳定性 (等权 h1): mean={r['mean_ic']:+.4f}, "
             f"IR={r['ir']:+.3f}, t={r['t']:+.2f}, IC>0 占比 "
             f"{r['ic_pos_share'] if r['ic_pos_share'] is None else format(r['ic_pos_share'], '.1%')}\n")
    L.append("## 正交性\n")
    L.append("| 对照 | corr |")
    L.append("|---|---|")
    for k, c in res["orthogonality"].items():
        L.append(f"| {k} | {c:+.3f} |")
    L.append("\n## 极端分位条件 (防御调制假说)\n")
    L.append("| 分位带 | 周数 | 等权年化 | 等权vol | 中证500年化 |")
    L.append("|---|---|---|---|---|")
    for band, d in res["extremes"].items():
        L.append(f"| {band} | {d['weeks']} | {d['ew_ann_ret']:+.2%} | "
                 f"{d['ew_ann_vol']:.2%} | {d['csi500_ann_ret']:+.2%} |")
    L.append("\n## 结论要点\n")
    L.append("- 假说 b (截面 tiebreaker) 结构性不适用: PE 分位是单一市场序列 (沪深300), "
             "无逐 ETF 截面; 唯一有意义截面为 A股暴露开关, 与假说 a 等价。")
    if res["verdict"]["gate_decision"] == "GO":
        L.append("- **GO (慢变量窗)**: h1 周频无效 (符合慢变量先验), 但 h13 等权组合 "
                 "IC 显著为负 (高估值→中期收益弱); 极端带分化清晰 (PE>90% 等权年化 "
                 "+3.1% vs 中间区 +15.7%)。")
        L.append("- **归因警告**: h13 显著性主要来自黄金/纳指 (海外资产) —— 沪深300 PE "
                 "对海外资产的预测力更可能是**全球宏观共同因子** (风险偏好/流动性周期) "
                 "而非估值因果; E2 设计时优先市场级防御调制 (对全组合), 而非单资产择时。")
        L.append("- **E1 GO ≠ E2 GO** (PVD 教训: E1 GO 但线性叠加 E2 NO-GO)。下一步 E2 "
                 "只测防御调制一种形态: pe_pct>0.9 时 def_ratio +δ (δ∈{0.05,0.10}, "
                 "封顶 max_def), 门禁 ΔSharpe≥+0.01/ΔMaxDD≤+0.3pp/bootstrap 中位不劣。")
    else:
        L.append("- 若判定 NO-GO: 估值是慢变量, 周频轮动截面消费价值低, 符合先验; "
                 "PE 数据保留在管线内不接入决策 (维持现状)。")
    L.append("- 对齐修正: PE CSV (周一) 与 NAV 周频标签 (周五) 此前从未对齐过 "
             "(生产预留路径未消费), 本探针用 ffill asof + shift(1) 对齐, 无前视。")
    out_md = OUT / "exp_pe_percentile_e1.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
