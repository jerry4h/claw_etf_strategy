#!/usr/bin/env python3
"""阶段三-6: A3 量价联合 DGP 公平性复核 (v4.5-pvd vs v4.6)。

复用 scripts/_exp_volume_price_dgp.py 的拟合/模拟/评估基建 (模块属性覆写
双配置路径), grey_corr_combo 情景 30 seeds 联合量价路径回测。

判据 (计划): v4.6 无结构性损害 — ΔSharpe ≥ −0.03 且 MaxDD 不恶化 (+0.5pp 容差)。

用法: .venv/bin/python scripts/_exp_v46_joint_dgp_check.py
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location(
    "dgpx", PROJ / "scripts" / "_exp_volume_price_dgp.py")
dgpx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dgpx)

OUT = PROJ / "output" / "experiments"
N_EVAL = 30


def main():
    t0 = time.time()
    # 覆写双配置: v4.5-pvd (基线) vs v4.6 (候选)
    dgpx.CFG43 = PROJ / "config" / "strategy_v4_5_pvd.yaml"
    dgpx.CFG45 = PROJ / "config" / "strategy_v4_6.yaml"

    print("[setup] 加载真实数据 + 拟合 VAR/GARCH + 量能模型 ...")
    nav, wk, w_rets = dgpx.dm.load_real()
    w_rets = np.asarray(w_rets, float)
    mu, A, Sigma, nu, resid, coords = dgpx.dm.fit_var_t(w_rets)
    gp, R = dgpx.adv.fit_garch(resid)
    real_dates, first_nav, T = wk.index, wk.iloc[0].values, len(w_rets)

    weekly_ret = wk.pct_change()
    weekly_amt = dgpx.load_weekly_volume_from_cache(
        dgpx.CACHE_DIR, wk.index, list(wk.columns))
    models = dgpx.fit_amount_models(weekly_ret, weekly_amt)
    # 创新项校准 (同 A3 主脚本口径)
    dgpx.calibrate_innovations(w_rets, weekly_amt, models, real_dates)

    print(f"[eval] grey_corr_combo 联合 DGP: {N_EVAL} seeds × (v4.5 vs v4.6) ...")
    rows = dgpx.eval_joint(mu, A, R, nu, gp, T, real_dates, first_nav,
                           list(wk.columns), models,
                           tuple(range(3000, 3000 + N_EVAL)), "v46check")
    med = dgpx.med
    res = {
        "n": len(rows),
        "v45_sharpe_med": med([x["v43"]["sharpe"] for x in rows]),
        "v46_sharpe_med": med([x["v45"]["sharpe"] for x in rows]),
        "delta_sharpe_med": med([x["delta_sharpe"] for x in rows]),
        "delta_sharpe_p10": float(np.percentile([x["delta_sharpe"] for x in rows], 10)),
        "v46_win_rate": float(np.mean([x["delta_sharpe"] > 0 for x in rows])),
        "v45_maxdd_med": med([x["v43"]["maxdd"] for x in rows]),
        "v46_maxdd_med": med([x["v45"]["maxdd"] for x in rows]),
    }
    no_harm = (res["delta_sharpe_med"] >= -0.03
               and res["v46_maxdd_med"] <= res["v45_maxdd_med"] + 0.005)
    res["verdict"] = {"no_structural_harm": bool(no_harm)}
    print(f"  v4.5 Sh={res['v45_sharpe_med']:.3f} | v4.6 Sh={res['v46_sharpe_med']:.3f} "
          f"| Δmed={res['delta_sharpe_med']:+.3f} P10={res['delta_sharpe_p10']:+.3f} "
          f"| 胜率={res['v46_win_rate']:.0%}")
    print(f"  MaxDD: v4.5 {res['v45_maxdd_med']:.2%} / v4.6 {res['v46_maxdd_med']:.2%}")
    print(f"[判定] 无结构性损害: {'是' if no_harm else '否'}")

    out_json = OUT / "exp_v46_joint_dgp_check.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")

    L = []
    L.append("# 阶段三-6: A3 联合 DGP 公平性复核 (v4.5-pvd vs v4.6)\n")
    L.append(f"> {pd_today()} | grey_corr_combo 情景 {res['n']} seeds | "
             "复用 _exp_volume_price_dgp.py 基建 (量模型已校准)\n")
    L.append("| 口径 | v4.5 Sh | v4.6 Sh | Δmed | ΔP10 | v4.6 胜率 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| 联合 DGP | {res['v45_sharpe_med']:.3f} | {res['v46_sharpe_med']:.3f} | "
             f"{res['delta_sharpe_med']:+.3f} | {res['delta_sharpe_p10']:+.3f} | "
             f"{res['v46_win_rate']:.0%} |")
    L.append(f"\nMaxDD 中位: v4.5 {res['v45_maxdd_med']:.2%} / v4.6 {res['v46_maxdd_med']:.2%}\n")
    L.append(f"**判定**: v4.6 无结构性损害 = {'是' if no_harm else '否'} "
             "(判据: ΔSharpe ≥ −0.03 且 MaxDD 不恶化 +0.5pp)\n")
    L.append("机制预期: grey 情景持续高波动 → directed boost 以灰区定向分支为主 "
             "(corr 多数 ≤0.60), PE 调制与 PVD 激活受 vol 门限抑制; Δ 应接近 0 或小幅正向。")
    with open(OUT / "exp_v46_joint_dgp_check.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


def pd_today():
    import pandas as pd
    return pd.Timestamp.today().date()


if __name__ == "__main__":
    main()
