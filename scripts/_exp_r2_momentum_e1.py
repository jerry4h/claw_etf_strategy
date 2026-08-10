#!/usr/bin/env python3
"""C1+ 探针: R² 加权回归动量 E1 评估 (聚宽调研 jq_community_survey.md #1 GO 项)。

来源: 聚宽"核心资产轮动"谱系标准打分 = 回归斜率年化 × R² (趋势一致性加权)。
本项目 mom6 为简单 6 周收益, 无趋势质量加权。作为 mom 的**替换变体**评估:
先验——与 mom6 高相关, 价值在噪声抑制 (R² 惩罚曲折路径) 而非新信息。

变体:
  R2M-A: 6 周 OLS 无权重回归, score = 年化斜率 × R²
  R2M-B: 6 周 WLS 线性递增权重 (近期数据加权, 社区原版), score = 年化斜率 × R²
  (对照) mom6: 简单 6 周收益 (生产口径)

E1 门禁 (同 PVD/份额口径): 截面 rank |IC| ≥ 0.03 且 |t| ≥ 1.5;
替换变体补充: 与 mom6 相关/排名一致率 (判断替换 vs 叠加)。

零 src/ 改动, 只读数据。用法: .venv/bin/python scripts/_exp_r2_momentum_e1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.data_loader import load_nav_data, resample_weekly
from src.factors import calculate_momentum

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
NAV_PATH = PROJ / "data" / "all_etfs_nav_latest.csv"
WINDOW = 6
IC_GATE, T_GATE = 0.03, 1.5
ANNUAL = 52.0


def r2_momentum(weekly: pd.DataFrame, window: int, weighted: bool) -> pd.DataFrame:
    """滚动 window 周回归动量: 年化斜率 × R²。

    weighted=True 时用线性递增权重 (最旧周权重 1, 最新周权重 window)。
    无前视: 窗口仅含截至当周的历史价格。
    """
    logp = np.log(weekly.values.astype(float))
    n, k = logp.shape
    out = np.full((n, k), np.nan)
    x = np.arange(window, dtype=float)
    w = (x + 1.0) if weighted else np.ones(window)
    w = w / w.sum()
    xw_mean = float(np.sum(w * x))
    for t in range(window - 1, n):
        for j in range(k):
            y = logp[t - window + 1:t + 1, j]
            if np.isnan(y).any():
                continue
            yw_mean = float(np.sum(w * y))
            cov = float(np.sum(w * (x - xw_mean) * (y - yw_mean)))
            varx = float(np.sum(w * (x - xw_mean) ** 2))
            if varx <= 0:
                continue
            slope = cov / varx
            yhat = slope * (x - xw_mean) + yw_mean
            ss_res = float(np.sum(w * (y - yhat) ** 2))
            ss_tot = float(np.sum(w * (y - yw_mean) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ann_ret = np.exp(slope * ANNUAL) - 1.0
            out[t, j] = ann_ret * max(r2, 0.0)
    return pd.DataFrame(out, index=weekly.index, columns=weekly.columns)


def cross_sectional_ic(sig: pd.DataFrame, fwd_ret: pd.DataFrame) -> dict:
    """逐周截面 Spearman IC 序列统计。"""
    ics = []
    for t in sig.index:
        if t not in fwd_ret.index:
            continue
        s = sig.loc[t].values
        f = fwd_ret.loc[t].values
        m = ~(np.isnan(s) | np.isnan(f))
        if m.sum() >= 3 and np.std(s[m]) > 0 and np.std(f[m]) > 0:
            rs = pd.Series(s[m]).rank().values
            rf = pd.Series(f[m]).rank().values
            ics.append(np.corrcoef(rs, rf)[0, 1])
    ics = np.array(ics)
    if len(ics) < 30:
        return {"n": len(ics), "mean_ic": np.nan, "t": np.nan, "ir": np.nan,
                "ic_pos_share": np.nan}
    mu, sd = ics.mean(), ics.std(ddof=1)
    return {"n": int(len(ics)), "mean_ic": float(mu),
            "t": float(mu / (sd / np.sqrt(len(ics)))) if sd > 0 else np.nan,
            "ir": float(mu / sd) if sd > 0 else np.nan,
            "ic_pos_share": float((ics > 0).mean())}


def main():
    nav = load_nav_data(NAV_PATH)
    weekly = resample_weekly(nav, anchor="W-MON")
    fwd_ret = weekly.pct_change().shift(-1)          # 下周收益 (IC 目标)
    fwd_ret4 = weekly.pct_change(4).shift(-4)        # 4 周前瞻 (稳健性)

    mom6 = calculate_momentum(weekly, window=WINDOW)
    r2m_a = r2_momentum(weekly, WINDOW, weighted=False)
    r2m_b = r2_momentum(weekly, WINDOW, weighted=True)

    res = {"window": WINDOW, "gates": {"ic": IC_GATE, "t": T_GATE}, "e1": {}}
    print(f"[data] {weekly.index[0].date()} ~ {weekly.index[-1].date()}, "
          f"{len(weekly)} 周, {list(weekly.columns)}")
    for name, sig in (("mom6(生产)", mom6), ("R2M-A(OLS)", r2m_a),
                      ("R2M-B(WLS递增权)", r2m_b)):
        ic1 = cross_sectional_ic(sig, fwd_ret)
        ic4 = cross_sectional_ic(sig, fwd_ret4)
        res["e1"][name] = {"h1": ic1, "h4": ic4}
        p1 = "←过门禁" if (abs(ic1["mean_ic"]) >= IC_GATE and abs(ic1["t"]) >= T_GATE) else ""
        print(f"[{name}] h1: IC={ic1['mean_ic']:+.4f} t={ic1['t']:+.2f} "
              f"IR={ic1['ir']:+.3f} IC>0={ic1['ic_pos_share']:.1%} {p1}")
        print(f"           h4: IC={ic4['mean_ic']:+.4f} t={ic4['t']:+.2f}")

    # 替换变体诊断: 与 mom6 的关系
    m = mom6.notna() & r2m_a.notna()
    corr_a = float(np.corrcoef(mom6.values[m.values], r2m_a.values[m.values])[0, 1])
    mb_m = mom6.notna().values & r2m_b.notna().values
    corr_b = float(np.corrcoef(mom6.values[mb_m], r2m_b.values[mb_m])[0, 1])
    # 排名一致率: 逐周进攻池 top-1 是否相同 (进攻 3 列)
    off = ["纳指ETF", "中证500ETF", "黄金ETF"]
    agree = []
    for t in mom6.index:
        a = mom6.loc[t, off]; b = r2m_a.loc[t, off]
        if a.notna().all() and b.notna().all():
            agree.append(a.idxmax() == b.idxmax())
    agree_rate = float(np.mean(agree))
    res["replacement_diag"] = {"corr_mom6_R2MA": corr_a, "corr_mom6_R2MB": corr_b,
                               "top1_agree_rate_offensive": agree_rate}
    print(f"[替换诊断] corr(mom6, R2M-A)={corr_a:+.3f}, corr(mom6, R2M-B)={corr_b:+.3f}, "
          f"进攻 top-1 一致率={agree_rate:.1%}")

    # 门禁判定 (h1 截面 IC)
    verdicts = {}
    for name in ("R2M-A(OLS)", "R2M-B(WLS递增权)"):
        ic = res["e1"][name]["h1"]
        verdicts[name] = bool(abs(ic["mean_ic"]) >= IC_GATE and abs(ic["t"]) >= T_GATE)
    res["verdict"] = verdicts
    print(f"[判定] " + ", ".join(f"{k}: {'GO' if v else 'NO-GO'}"
                                  for k, v in verdicts.items()))

    out_json = OUT / "exp_r2_momentum_e1.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")
    render_md(res)


def render_md(res):
    L = []
    L.append("# 探针: R² 加权回归动量 E1 评估 (聚宽调研 GO 项)\n")
    L.append(f"> {pd.Timestamp.today().date()} | window={res['window']} 周 | "
             f"脚本 `scripts/_exp_r2_momentum_e1.py` | 来源 output/experiments/jq_community_survey.md #1\n")
    L.append("## E1 截面 rank IC (5 ETF 截面, Spearman)\n")
    L.append("| 信号 | h1 mean_IC | h1 t | h1 IR | IC>0 | h4 mean_IC | h4 t |")
    L.append("|---|---|---|---|---|---|---|")
    for name, d in res["e1"].items():
        L.append(f"| {name} | {d['h1']['mean_ic']:+.4f} | {d['h1']['t']:+.2f} | "
                 f"{d['h1']['ir']:+.3f} | {d['h1']['ic_pos_share']:.1%} | "
                 f"{d['h4']['mean_ic']:+.4f} | {d['h4']['t']:+.2f} |")
    rd = res["replacement_diag"]
    L.append(f"\n## 替换变体诊断\n")
    L.append(f"- corr(mom6, R2M-A) = {rd['corr_mom6_R2MA']:+.3f}; "
             f"corr(mom6, R2M-B) = {rd['corr_mom6_R2MB']:+.3f}")
    L.append(f"- 进攻池 top-1 排名一致率 = {rd['top1_agree_rate_offensive']:.1%}")
    L.append("\n## 判定\n")
    for name, v in res["verdict"].items():
        L.append(f"- {name}: **{'GO (进 E2 替换变体评估)' if v else 'NO-GO (枪毙, 有效结论)'}**")
    L.append("\n先验核对: R² 动量与 mom6 高相关属预期 (替换变体); 若 GO, 价值应在噪声抑制 "
             "(排名稳定性/换手下降) 而非新信息——E2 需以净 Sharpe (含换手成本) 为判定口径。")
    out_md = OUT / "exp_r2_momentum_e1.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
