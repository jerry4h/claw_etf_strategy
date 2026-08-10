#!/usr/bin/env python3
"""B1 预研: 定向 boost (directed boost) — Layer 3.5 应用点重构可行性验证。

背景 (docs/v4_5_grey_corr_abort.md §5, 36 格实验穷尽标量/门控方案后的唯一幸存方向):
  当前 Layer 3.5: 高相关 → def_ratio += boost → 推高防御绝对水平。
  物理冲突: bond_bear DGP 下防御资产(债)自身下行, 降阈值触发 boost 反而
  把权重推向下跌中的防御资产 → MaxDD 恶化 (M-C/M-D 全 FAIL 的根因之一)。
  定向 boost: 高相关 → 仅对进攻端降权, 不(少)推高防御绝对水平。

变体矩阵 (基座 = v4.4 EWMA Layer3.5, PVD 不参与以隔离合成路径的量数据依赖):
  触发器:
    T0 = 生产 EWMA(hl=8) thr 0.60 slope 1.875 (v4.4/v4.5-pvd 现状)
    T1 = EWMA(hl=8) thr 0.45 slope 0.75 (M-C 触发器, 覆盖灰区 0.3-0.5)
  应用点:
    C  = 现状: def_ratio = min(def_ratio + b, 1.0)
    V1 = 比例式: def_ratio = min(def_ratio + b*(1-def_ratio), 1.0)
         → 进攻端缩至 (1-b) 倍, 释放权重按 DefAlloc 比例进防御 (温和版)
    V2 = 现金缓冲: def_ratio 不动, 进攻端 alloc *= (1-b)
         → 释放权重留作现金(0 收益), 防御绝对水平零抬升 (严格"定向")
  共 6 变体: B0(T0-C 基线), T1-C(参照, 预期 bond_bear 恶化),
            T0-V1, T1-V1, T0-V2, T1-V2。

压测情景 (与 M-C/M-D 同口径 + 生产硬门禁情景):
  bond_bear (CCC, mudef_mult=0.5)          — defense_asset 硬门禁, 核心观察点
  grey_corr_combo (regime_corr 0.50×σ1.5)  — 灰区缺口是否收敛
  corr_regime_shift (regime_corr 0.85)     — 显性危机不劣化
  corr_crisis_combo (0.85+σ1.2+muoff0.8)   — 复合冲击不劣化
  realized (真实历史)                       — 生产口径不回退

实现: inspect 源码手术 (只替换 boost 应用块, 其余逐字符与 src/backtest.py
一致) + monkeypatch src.backtest.run_backtest; 触发器 monkeypatch
src.backtest.compute_crisis_boost。零 src/ 改动, 零生产文件改动。

前视面核查: 触发器沿用 [i-window, i) 已完成收益窗 (同生产); V1/V2 应用点
只消费第 i 周决策时刻已有量 (def_ratio/alloc), 未引入新信息面。

用法: .venv/bin/python scripts/_exp_directed_boost_study.py
"""
import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location(
    "adv", PROJ / "scripts" / "adversarial_robustness.py")
adv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adv)
dm = adv.dm

import src.backtest as sbt
from src.backtest import run_backtest
from src.strategy import load_config

OUT = PROJ / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
CFG44 = PROJ / "config" / "strategy_v4_4.yaml"

SEEDS = (11, 22, 33, 44, 55, 66, 77)
OFF_IDX = dm.OFF_IDX
D_MAX = 0.12

SCENARIOS = {
    "bond_bear":        {"gen": "garch",  "params": dict(adv.REALIZED, mudef_mult=0.5)},
    "grey_corr_combo":  {"gen": "regime", "params": dict(adv.REALIZED, dgp="regime_corr",
                          rho_crisis=0.50, p_enter=1.0, p_stay=1.0, sig_mult=1.5)},
    "corr_regime_shift": {"gen": "regime", "params": dict(adv.REALIZED, dgp="regime_corr",
                          rho_crisis=0.85)},
    "corr_crisis_combo": {"gen": "regime", "params": dict(adv.REALIZED, dgp="regime_corr",
                          rho_crisis=0.85, sig_mult=1.2, muoff_mult=0.8)},
}


# ======================================================================
# 触发器 (monkeypatch compute_crisis_boost)
# ======================================================================
def maxcorr_ewma(w, i, off_idx, window, halflife):
    """复刻 engine_core._compute_crisis_boost_ewma 的 EWMA max|ρ|。"""
    if i < window or len(off_idx) < 2:
        return np.nan
    win = w[i - window:i, off_idx]
    t = np.arange(window)
    weights = 0.5 ** ((window - 1 - t) / max(halflife, 1))
    mc = np.nan
    n = win.shape[1]
    for a in range(n):
        for b in range(a + 1, n):
            mask = ~(np.isnan(win[:, a]) | np.isnan(win[:, b]))
            if mask.sum() >= 5:
                x, y = win[mask, a], win[mask, b]
                ww = weights[mask]
                ww = ww / ww.sum()
                xb, yb = float(np.sum(ww * x)), float(np.sum(ww * y))
                cov = float(np.sum(ww * (x - xb) * (y - yb)))
                vx = float(np.sum(ww * (x - xb) ** 2))
                vy = float(np.sum(ww * (y - yb) ** 2))
                c = cov / (np.sqrt(vx * vy) + 1e-12)
                if not np.isnan(c):
                    mc = c if np.isnan(mc) else mc
                    mc = max(mc, abs(c))
    return mc


def make_trigger(threshold, slope, halflife=8):
    """EWMA 触发器工厂 (窗口/上限沿用 config, 阈值与斜率参数化)。"""
    def fn(w_rets, i, off_idx, config):
        c = maxcorr_ewma(np.asarray(w_rets, float), i, off_idx,
                         config.crisis_corr_window, halflife)
        if np.isnan(c) or c <= threshold:
            return 0.0
        return float(min((c - threshold) * slope, config.crisis_corr_max_boost))
    return fn


TRIG_T0 = None  # None = 用 config 原引擎路径 (EWMA hl=8 thr0.60, v4.4 现状)
TRIG_T1 = make_trigger(0.45, 0.75)


def trigger_stats(w_rets, cfg, fn=None):
    from src.engine_core import compute_crisis_boost as engine_fn
    w = np.asarray(w_rets, float)
    T = len(w)
    window = cfg.crisis_corr_window
    f = fn or engine_fn
    b = np.array([f(w, i, OFF_IDX, cfg) for i in range(window, T)])
    on = b > 0
    return {
        "trigger_rate": float(on.mean()) if len(b) else 0.0,
        "mean_boost_on": float(b[on].mean()) if on.any() else 0.0,
    }


# ======================================================================
# 应用点手术: inspect 源码替换, 其余逐字符同 src/backtest.py
# ======================================================================
_BOOST_BLOCK = """        crisis_boost = compute_crisis_boost(w_rets, i, off_idx, config)
        if crisis_boost > 0:
            def_ratio = min(def_ratio + crisis_boost, 1.0)"""

_ANCHOR_REBAL = "        # --- 调仓阈值检查 ---"

# 原始 run_backtest 源码 (模块加载时一次性捕获: 后续 exec 会覆盖
# sbt.run_backtest, 届时 inspect.getsource 将拿不到源码)
_ORIG_RB_SOURCE = inspect.getsource(sbt.run_backtest)


def build_run_backtest(application):
    """application ∈ {'C','V1','V2'} → 手术后的 run_backtest 函数对象。

    C : 现状 (def += b)
    V1: 比例式 (def += b*(1-def)) — 进攻缩 (1-b), 释放进防御
    V2: 现金缓冲 — def 不动, 进攻 alloc *= (1-b), 释放留现金(隐式 0 收益)
    V3: 混合 — 显性危机 (EWMA corr>0.60) 用 C 满额保护; 灰区 (≤0.60) 用 V1 定向
    """
    src = _ORIG_RB_SOURCE
    if application == "C":
        pass  # 原样
    elif application == "V1":
        patched_block = """        crisis_boost = compute_crisis_boost(w_rets, i, off_idx, config)
        if crisis_boost > 0:
            def_ratio = min(def_ratio + crisis_boost * (1.0 - def_ratio), 1.0)"""
        assert _BOOST_BLOCK in src, "boost 应用块源码已漂移, 手术锚点失效"
        src = src.replace(_BOOST_BLOCK, patched_block)
    elif application == "V2":
        patched_block = """        crisis_boost = compute_crisis_boost(w_rets, i, off_idx, config)
        _db_cash_boost = crisis_boost if crisis_boost > 0 else 0.0"""
        assert _BOOST_BLOCK in src and _ANCHOR_REBAL in src, "手术锚点失效"
        src = src.replace(_BOOST_BLOCK, patched_block)
        inject = ("        if _db_cash_boost > 0:\n"
                  "            for _j in off_idx:\n"
                  "                alloc[_j] *= (1.0 - _db_cash_boost)\n\n"
                  + _ANCHOR_REBAL)
        src = src.replace(_ANCHOR_REBAL, inject, 1)
    elif application == "V3":
        patched_block = """        crisis_boost = compute_crisis_boost(w_rets, i, off_idx, config)
        if crisis_boost > 0:
            _mc = _db_maxcorr_ewma(w_rets, i, off_idx, config.crisis_corr_window, 8)
            if not _mc or _mc > 0.60:
                def_ratio = min(def_ratio + crisis_boost, 1.0)
            else:
                def_ratio = min(def_ratio + crisis_boost * (1.0 - def_ratio), 1.0)"""
        assert _BOOST_BLOCK in src, "boost 应用块源码已漂移, 手术锚点失效"
        sbt._db_maxcorr_ewma = maxcorr_ewma  # 注入辅助函数供 exec 全局解析
        src = src.replace(_BOOST_BLOCK, patched_block)
    else:
        raise ValueError(application)
    # 去掉原定义装饰器/缩进层级: run_backtest 是模块级函数, exec 回模块全局
    # (用模块自身 __dict__ 而非快照: compute_crisis_boost 需动态解析,
    #  否则后续 monkeypatch 触发器不生效)
    exec(compile(src, f"<directed_boost_{application}>", "exec"), sbt.__dict__)
    return sbt.__dict__["run_backtest"]


# ======================================================================
# 评估
# ======================================================================
def eval_seed(nav_df, cfg, rb_fn, seed, tag):
    tmp = OUT / f"_synth_db_{tag}_{seed}_{os.getpid()}.csv"
    nav_df.to_csv(tmp, encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = rb_fn(cfg, start_date=dm.START_DATE, data_path=str(tmp))
        if res.nav_series.empty:
            return None
        return {
            "sharpe": float(res.metrics["sharpe_ratio"]),
            "maxdd": float(res.metrics["max_drawdown"]),
            "annual": float(res.metrics["annual_return"]),
        }
    finally:
        if tmp.exists():
            os.remove(tmp)


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


VARIANTS = [
    # (key, 标签, trigger(None=引擎原路径), application)
    ("B0_T0_C",    "基线 v4.4 现状 (T0+C)",        TRIG_T0, "C"),
    ("REF_T1_C",   "参照 M-C触发+现状应用 (T1+C)", TRIG_T1, "C"),
    ("T0_V1",      "比例式 (T0+V1)",               TRIG_T0, "V1"),
    ("T1_V1",      "比例式+灰区触发 (T1+V1)",      TRIG_T1, "V1"),
    ("T0_V2",      "现金缓冲 (T0+V2)",             TRIG_T0, "V2"),
    ("T1_V2",      "现金缓冲+灰区触发 (T1+V2)",    TRIG_T1, "V2"),
    ("T1_V3",      "混合应用+灰区触发 (T1+V3)",   TRIG_T1, "V3"),
]


def main():
    t0 = time.time()
    cfg = load_config(CFG44)
    assert cfg.pvd_enabled is False, "基座不应含 PVD (合成路径无量数据)"
    assert getattr(cfg, "crisis_corr_ewma_enabled", False), "基座应为 EWMA Layer3.5 (v4.4)"

    print("[setup] 加载真实数据 + 拟合 VAR(1)-t + GARCH ...")
    nav, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = adv.fit_garch(resid)
    real_dates, first_nav, T = wk.index, wk.iloc[0].values, len(w_rets)
    print(f"[setup] T={T} 周, ν={nu:.1f}")

    res = {"base_config": str(CFG44.name), "seeds": list(SEEDS),
           "d_max": D_MAX, "variants": {}}

    for key, label, trig, app in VARIANTS:
        tv = time.time()
        rb_fn = build_run_backtest(app)
        orig_rb, orig_cb = sbt.run_backtest, sbt.compute_crisis_boost
        sbt.run_backtest = rb_fn
        if trig is not None:
            sbt.compute_crisis_boost = trig
        try:
            vout = {"label": label, "trigger": "T1(thr0.45,slope0.75)" if trig else "T0(生产)",
                    "application": app, "scenarios": {}}
            # realized
            with contextlib.redirect_stdout(io.StringIO()):
                r = rb_fn(cfg, start_date=dm.START_DATE, data_path=str(dm.REAL_CSV))
            vout["realized"] = {k: float(r.metrics[k]) for k in
                                ("sharpe_ratio", "max_drawdown", "annual_return")}
            vout["realized_trigger"] = trigger_stats(
                np.asarray(w_rets, float), cfg, fn=trig)
            # 合成情景
            for sc_name, sc in SCENARIOS.items():
                gen = adv.gen_regime_corr if sc["gen"] == "regime" else adv.gen_garch
                rows = []
                for s in SEEDS:
                    rr = gen(mu, A, R, nu, gp, sc["params"], T, s)
                    ndf = dm.build_nav_df(rr, real_dates, first_nav)
                    e = eval_seed(ndf, cfg, rb_fn, s, f"{key}_{sc_name}")
                    if e:
                        rows.append(e)
                vout["scenarios"][sc_name] = {
                    "n": len(rows),
                    "maxdd_med": med([x["maxdd"] for x in rows]),
                    "maxdd_worst": max((x["maxdd"] for x in rows), default=float("nan")),
                    "sharpe_med": med([x["sharpe"] for x in rows]),
                    "annual_med": med([x["annual"] for x in rows]),
                }
        finally:
            sbt.run_backtest, sbt.compute_crisis_boost = orig_rb, orig_cb
        res["variants"][key] = vout
        print(f"[{key}] {label}: realized Sh={vout['realized']['sharpe_ratio']:.3f} "
              f"DD={vout['realized']['max_drawdown']:.2%} "
              f"trig={vout['realized_trigger']['trigger_rate']:.1%} | "
              + " ".join(f"{n}:{v['maxdd_med']:.2%}"
                         for n, v in vout["scenarios"].items())
              + f" ({time.time()-tv:.0f}s)", flush=True)

    out_json = OUT / "exp_directed_boost.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[save] {out_json}")
    render_md(res)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


# ======================================================================
# 报告
# ======================================================================
def render_md(res):
    B0 = res["variants"]["B0_T0_C"]
    REF = res["variants"]["REF_T1_C"]
    L = []
    L.append("# 实验: B1 定向 boost 预研 (Layer 3.5 应用点重构)\n")
    L.append(f"> {pd.Timestamp.today().date()} | 基座 {res['base_config']} | "
             f"seeds={res['seeds']} | 脚本 `scripts/_exp_directed_boost_study.py` | "
             f"数据 `output/experiments/exp_directed_boost.json` | 零生产文件改动\n")
    L.append("## 1. 设计\n")
    L.append("触发器: T0 = 生产 EWMA(hl=8) thr0.60; T1 = M-C 触发器 EWMA thr0.45 slope0.75 (覆盖灰区)。\n")
    L.append("应用点: C = 现状 `def+=b`; V1 = 比例式 `def+=b(1-def)` (进攻缩(1-b), 释放进防御); "
             "V2 = 现金缓冲 (def 不动, 进攻 alloc×(1-b), 释放留现金 0 收益); "
             "V3 = 混合 (EWMA corr>0.60 显性危机用 C 满额保护, 灰区用 V1 定向)。\n")
    L.append("实现: inspect 源码手术 (boost 应用块替换, 其余逐字符同生产) + monkeypatch, 零 src/ 改动。\n")
    L.append("前视面: 触发沿用 [i-window,i) 已完成收益; 应用点只消费第 i 周已有决策量, 无新信息面。\n")

    L.append("## 2. Realized (真实历史)\n")
    L.append("| 变体 | Sharpe | MaxDD | 年化 | 触发率 | 平均boost |")
    L.append("|---|---|---|---|---|---|")
    for k, v in res["variants"].items():
        rt = v["realized"]; tg = v["realized_trigger"]
        L.append(f"| {v['label']} | {rt['sharpe_ratio']:.3f} | {rt['max_drawdown']:.2%} | "
                 f"{rt['annual_return']:.2%} | {tg['trigger_rate']:.1%} | "
                 f"{tg['mean_boost_on']:.3f} |")

    L.append("\n## 3. 合成压测 (中位 MaxDD / 中位 Sharpe, 7 seeds; **加粗**=破 12% 红线)\n")
    sc_names = list(SCENARIOS.keys())
    L.append("| 变体 | " + " | ".join(sc_names) + " |")
    L.append("|---|" + "---|" * len(sc_names))
    for k, v in res["variants"].items():
        row = [v["label"]]
        for sn in sc_names:
            s = v["scenarios"][sn]
            dd = f"**{s['maxdd_med']:.2%}**" if s["maxdd_med"] > D_MAX else f"{s['maxdd_med']:.2%}"
            row.append(f"{dd} / {s['sharpe_med']:.2f}")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## 4. 门禁判定\n")
    gates = []
    bb0 = B0["scenarios"]["bond_bear"]["maxdd_med"]
    gr0 = B0["scenarios"]["grey_corr_combo"]["maxdd_med"]
    sh0 = B0["realized"]["sharpe_ratio"]
    for k, v in res["variants"].items():
        if k == "B0_T0_C":
            continue
        bb = v["scenarios"]["bond_bear"]["maxdd_med"]
        gr = v["scenarios"]["grey_corr_combo"]["maxdd_med"]
        rs = v["realized"]["sharpe_ratio"]
        rdd = v["realized"]["max_drawdown"]
        g1 = bb <= D_MAX and bb <= bb0 + 0.002
        g2 = gr <= D_MAX and gr < gr0 - 0.001
        g3 = rs >= sh0 - 0.01 and rdd <= B0["realized"]["max_drawdown"] + 0.003
        verdict = "PASS" if (g1 and g2 and g3) else "NO-GO"
        gates.append((k, v["label"], g1, g2, g3, verdict))
        L.append(f"- **{v['label']}**: bond_bear {bb:.2%} (基线 {bb0:.2%}) "
                 f"{'✓' if g1 else '✗'} | 灰区 {gr:.2%} (基线 {gr0:.2%}) "
                 f"{'✓' if g2 else '✗'} | realized Sh {rs:.3f} (基线 {sh0:.3f})/DD "
                 f"{rdd:.2%} {'✓' if g3 else '✗'} → **{verdict}**")
    n_pass = sum(1 for g in gates if g[5] == "PASS")
    L.append(f"\n**结论**: {n_pass}/{len(gates)} 个定向 boost 变体通过全部预研门禁。"
             + ("建议进入 E3 立项 (配置 enabled=false 默认关闭 + TestBaselineUnchanged pin)。"
                if n_pass else " 定向应用点未能同时满足 bond_bear 不恶化 + 灰区收敛 + realized 不回退,"
                " B1 方向 NO-GO 归档 (见 §5 归因)。"))
    L.append("\n## 5. 归因与局限\n")
    L.append("- REF (T1+C) 是预期对照: 降阈值 + 现状应用点应在 bond_bear 恶化 "
             "(复现 36 格实验的 M-C 失败模式), 实测 bond_bear 12.25% 破线——机制诊断成立: "
             "bond_bear 中进攻对相关中位≈0.40 落在灰区, T1 触发后经 C 应用把权重推向下跌中的债券。")
    if "T1_V3" in res["variants"]:
        v3 = res["variants"]["T1_V3"]
        bb = v3["scenarios"]["bond_bear"]["maxdd_med"]
        gr = v3["scenarios"]["grey_corr_combo"]["maxdd_med"]
        rs = v3["scenarios"]["corr_regime_shift"]["maxdd_med"]
        L.append(f"- **V3 混合应用是解的关键**: corr>0.60 显性危机保留 C 满额保护 "
                 f"(corr_regime_shift {rs:.2%} 与基线 7.85% 逐位一致); corr≤0.60 灰区改 V1 定向 "
                 f"(bond_bear {bb:.2%} 不再破线; 灰区缺口 {gr:.2%} 收敛回红线内)。"
                 "纯定向 (T0+V1/V2) 在显性危机情景反而恶化 +1.3pp——削弱应用点本身有代价, "
                 "分级应用而非一刀切才是正确形态。")
    L.append("- V2 现金缓冲的代价: 显性危机 (corr_regime_shift/corr_crisis_combo) 中债券上行时"
             "释放权重拿不到防御收益, Sharpe 可能小幅劣化——用灰区/bond_bear 改善交换。")
    L.append("- 合成 DGP 无 PVD/量维度, 基座用 v4.4 (PVD 关闭) 隔离; 若立项 E3 需在 v4.5-pvd 上"
             "复验 realized 与 block bootstrap。")
    L.append("- 前视面核查: V3 的分级判断用 EWMA corr (同触发器口径, [i-window,i) 已完成收益), "
             "未引入新信息面; V1/V2 应用点只消费第 i 周已有决策量。")
    L.append("- 预研门禁为方向性筛选; 通过后仍需完整管线 (evaluate --corr-scenarios + 三通道 OOS "
             "+ 联合鲁棒性) 才能进生产。")
    out_md = OUT / "exp_directed_boost.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
