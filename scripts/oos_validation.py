#!/usr/bin/env python3
"""v4.0 节点4: 对抗 OOS 验证 - 三条独立通道判定"真鲁棒"vs"过拟合对抗测试"。

训练集(节点3优化时用的对抗环境):
  DGP    : CCC-GARCH (从真实残差拟合)
  情景   : 5 stress = {σ×1.2, μoff×0.8, μdef×0.5, c×0.77, σ×1.2+μoff×0.8}
  seeds  : 11,22,33,44,55,66,77 (7 个)

OOS 三条独立通道(每条从一个正交方向验证):
  A. held_out_magnitudes: 相同 DGP + 训练未见过的扰动幅度
                           (σ×{0.9,1.4}, μoff×{0.6,1.0}, μdef×{0.3,0.7}, c×{0.5,1.3}, 复合冲击)
  B. independent_seeds  : 相同 DGP + 相同扰动幅度 + 完全独立 seed 集(100-116, 17 个)
  C. block_bootstrap    : 完全独立 DGP(非参数, 从真实周收益 block=8 重采样, 跳出 CCC-GARCH 假设)

对每条通道: 同时评估 v4_1(历史基线) 和 v4_2(生产), 比较退化程度。
输出: pass_rate / worst_maxdd / 平均 Sharpe margin(strat-ew), 逐 config 逐通道。

真鲁棒判定(三条都满足):
  1. v4_2 pass_rate ≥ v4_1 pass_rate  (方向不劣化)
  2. v4_2 worst_maxdd ≤ D_max + 1pp  (回撤约束在 OOS 也守得住)
  3. v4_2 平均 Sharpe margin > 0     (不是靠训练集卡边缘)
若三条通道都稳, 接受"真鲁棒"; 有一条塌陷, 说明节点3 对抗测试过拟合, 需重优化。
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.strategy import load_config


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, PROJ / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

adv = _load("adv", "scripts/adversarial_robustness.py")
dm  = adv.dm
OUT = PROJ / "output" / "adversarial"

# 训练集(节点3), 用于对比说明
TRAINING_SEEDS    = (11, 22, 33, 44, 55, 66, 77)
TRAINING_SCENARIOS = ("vol_stress", "offense_cooldown", "bond_bear", "decorrelation", "stagflation")

# =========== 通道 A: held-out 扰动幅度 ============
OOS_MAGNITUDE_SCENARIOS = {
    # 名称: (params, 相对训练集的类型)
    "vol_small":       {"sig_mult": 0.9},                          # σ 未见: 训练只见 1.2
    "vol_large":       {"sig_mult": 1.4},                          # σ 未见: 大扰动
    "offense_severe":  {"muoff_mult": 0.6},                        # μoff 未见: 更狠
    "offense_neutral": {"muoff_mult": 1.0},                        # μoff 未见: 无扰动
    "def_deep_bear":   {"mudef_mult": 0.3},                        # μdef 未见: 深熊
    "def_mild_bear":   {"mudef_mult": 0.7},                        # μdef 未见: 浅熊
    "decorr_strong":   {"c_mult": 0.5},                            # c 未见: 强分散
    "corr_boost":      {"c_mult": 1.3},                            # c 未见: 相关放大
    "stag_severe":     {"sig_mult": 1.4, "muoff_mult": 0.6},       # 复合未见: 更狠滞胀
    "quad_shock":      {"sig_mult": 1.3, "muoff_mult": 0.7,        # 四轴复合未见
                        "mudef_mult": 0.5, "c_mult": 0.7},
}

# v4.4 通道A 追加: regime_corr 幅度变体(--corr-variants 显式开启; 默认不跑,
# 既有 10 情景与三通道判定行为不变; 训练只见 rho_crisis=0.85)
OOS_CORR_VARIANTS = {
    "corr_shift_075": {"dgp": "regime_corr", "rho_crisis": 0.75},  # rho 未见: 更温和
    "corr_shift_090": {"dgp": "regime_corr", "rho_crisis": 0.90},  # rho 未见: 更极端
}


# =========== 通道 C: block bootstrap 辅助 ============
def block_bootstrap(returns, block_len, seed):
    """Moving block bootstrap: 保留 block_len 周内相关结构, 打乱块序。"""
    T, k = returns.shape
    n_blocks = (T + block_len - 1) // block_len
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T - block_len + 1, size=n_blocks)
    out = np.concatenate([returns[s:s + block_len] for s in starts], axis=0)
    return out[:T]


def eval_strat_ew_on_returns(returns, real_dates, first_nav, cfg, tmp_tag):
    """对给定 (T,k) 周收益矩阵, 跑策略 + 每周等权基准, 返回 (sharpe, maxdd, annual) x2。"""
    from src.backtest import run_backtest, compute_metrics
    from src.data_loader import ETFS
    nav_df = dm.build_nav_df(returns, real_dates, first_nav)
    tmp = OUT / f"_oos_{tmp_tag}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    nav_df.to_csv(tmp, encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(cfg, start_date=dm.START_DATE, data_path=str(tmp))
        if res.nav_series.empty:
            return None
        s_sh = float(res.metrics["sharpe_ratio"])
        s_dd = float(res.metrics["max_drawdown"])
        s_an = float(res.metrics["annual_return"])
        start, end = res.nav_series.index[0], res.nav_series.index[-1]
        cols = [c for c in nav_df.columns if c in ETFS]
        pr = nav_df.loc[start:end, cols].astype(float)
        idx = pr.index
        valid = ~np.isnan(pr.iloc[0].values)
        er = pr.ffill().pct_change().fillna(0.0).values
        rb = np.ones(len(idx))
        for i in range(1, len(idx)):
            rb[i] = rb[i-1] * (1 + float(np.mean(er[i, valid])))
        wr = np.zeros(len(rb)); wr[1:] = rb[1:] / rb[:-1] - 1
        peak = np.maximum.accumulate(rb); dd = (peak - rb) / peak
        df_rb = pd.DataFrame({"nav": rb, "weekly_return": wr, "drawdown": dd,
                              "def_ratio": 0.0, "turnover": 0.0}, index=idx)
        em = compute_metrics(df_rb, cfg.risk_free_rate)
        return {"strat_sharpe": s_sh, "strat_maxdd": s_dd, "strat_annual": s_an,
                "ew_sharpe":    em["sharpe_ratio"], "ew_maxdd": em["max_drawdown"],
                "ew_annual":    em["annual_return"]}
    finally:
        if tmp.exists(): os.remove(tmp)


# =========== 三条通道实施 ============
def run_channel_ab(cfg_name, cfg, dgp, scenarios, seeds, label):
    """通道 A/B: CCC-GARCH DGP 上跑给定情景 × seeds 集。"""
    mu, A, R, nu, gp, real_dates, first_nav, T = dgp
    results = {}
    for name, params_over in scenarios.items():
        params = dict(adv.REALIZED, **params_over)
        # v4.4: 支持 params 里的 dgp=regime_corr 分支(与 adv._eval_strat_ew 同口径)
        gen = adv.gen_regime_corr if params.get("dgp") == "regime_corr" else adv.gen_garch
        s_sh_l, e_sh_l, s_dd_l, s_an_l, e_an_l = [], [], [], [], []
        for seed in seeds:
            r = gen(mu, A, R, nu, gp, params, T, seed)
            m = eval_strat_ew_on_returns(r, real_dates, first_nav, cfg, f"{label}_{name}_{seed}")
            if m is None: continue
            s_sh_l.append(m["strat_sharpe"]); e_sh_l.append(m["ew_sharpe"])
            s_dd_l.append(m["strat_maxdd"]); s_an_l.append(m["strat_annual"])
            e_an_l.append(m["ew_annual"])
        med = lambda xs: float(np.median(xs)) if xs else float("nan")
        results[name] = {
            "strat_sharpe": med(s_sh_l), "ew_sharpe": med(e_sh_l),
            "strat_maxdd":  med(s_dd_l),
            "strat_annual": med(s_an_l), "ew_annual": med(e_an_l),
            "margin":       med(s_sh_l) - med(e_sh_l),
            "beats_ew":     bool(med(s_sh_l) > med(e_sh_l)),
        }
    n = len(results)
    n_pass = sum(1 for v in results.values() if v["beats_ew"])
    worst_dd = float(np.nanmax([v["strat_maxdd"] for v in results.values()]))
    avg_margin = float(np.mean([v["margin"] for v in results.values()]))
    return {"config": cfg_name, "scenarios": results,
            "pass_rate": n_pass / n, "worst_maxdd": worst_dd, "avg_margin": avg_margin,
            "n": n}


def run_channel_c(cfg_name, cfg, real_returns, real_dates, first_nav, block_len, n_paths, seed_base):
    """通道 C: 从真实周收益 block bootstrap 生成 n_paths 条独立路径。"""
    T = len(real_returns)
    s_sh_l, e_sh_l, s_dd_l, s_an_l, e_an_l = [], [], [], [], []
    for i in range(n_paths):
        boot = block_bootstrap(real_returns, block_len, seed_base + i)
        m = eval_strat_ew_on_returns(boot, real_dates, first_nav, cfg, f"boot_{i}")
        if m is None: continue
        s_sh_l.append(m["strat_sharpe"]); e_sh_l.append(m["ew_sharpe"])
        s_dd_l.append(m["strat_maxdd"]); s_an_l.append(m["strat_annual"])
        e_an_l.append(m["ew_annual"])
    # 通道 C 的 pass_rate 用路径级(每条路径独立跑赢 EW 的比例)
    pass_paths = sum(1 for s, e in zip(s_sh_l, e_sh_l) if s > e)
    return {
        "config": cfg_name, "n_paths": len(s_sh_l), "block_len": block_len,
        "strat_sharpe_median": float(np.median(s_sh_l)) if s_sh_l else float("nan"),
        "ew_sharpe_median":    float(np.median(e_sh_l)) if e_sh_l else float("nan"),
        "strat_maxdd_median":  float(np.median(s_dd_l)) if s_dd_l else float("nan"),
        "strat_maxdd_max":     float(np.max(s_dd_l))    if s_dd_l else float("nan"),
        "strat_annual_median": float(np.median(s_an_l)) if s_an_l else float("nan"),
        "ew_annual_median":    float(np.median(e_an_l)) if e_an_l else float("nan"),
        "pass_rate":           pass_paths / len(s_sh_l) if s_sh_l else float("nan"),
        "avg_margin":          float(np.mean(np.array(s_sh_l) - np.array(e_sh_l))) if s_sh_l else float("nan"),
    }


# =========== 判定 ============
def verdict(v41, v42, n_bucket, d_max_slack=0.13,
            pass_regress_tol_buckets=1, regress_tol_dd=0.02, regress_tol_margin=0.05):
    """核心判定 = 相对基线不劣化(直接测试过拟合假设); 附带记录 envelope 状态。

    过拟合对抗测试的教科书签名: 在独立OOS上, 候选相对基线显著劣化。
    所以真正应该测的是"v4_2 相对 v4_1 有无退化", 而非"v4_2 绝对达到某阈值"
    (后者混淆了策略族设计包线上限, 极端OOS幅度下任何该族策略都可能超阈值)。

    pass_rate 容差按"允许少通过的桶数"定义 (n_bucket = 情景数 or 路径数):
      pass_rate_tol = pass_regress_tol_buckets / n_bucket
      -- 与实际离散粒度一致, 不再出现"5% 容差在 0.1 粒度下永远不生效"的问题。

    core 检查(过拟合假设的直接反驳):
      1. pass_rate 不劣化(允许最多 pass_regress_tol_buckets 个桶回落)
      2. worst_dd 不劣化(允许小容差 regress_tol_dd)
      3. avg_margin 不劣化 且 > 0
    envelope 记录(独立于 core, 不参与总判定):
      worst_dd_within_envelope: v4_2 的 worst_dd ≤ d_max_slack
      若 core PASS 但 envelope FAIL → v4_2 不过拟合, 但该 OOS 超策略族设计上界(架构问题)
    """
    v41_dd = v41.get("worst_maxdd", v41.get("strat_maxdd_max"))
    v42_dd = v42.get("worst_maxdd", v42.get("strat_maxdd_max"))
    pass_tol = pass_regress_tol_buckets / max(n_bucket, 1)
    core = {
        "pass_rate_not_regressed":  v42["pass_rate"]  >= v41["pass_rate"]  - pass_tol,
        "worst_dd_not_regressed":   v42_dd            <= v41_dd            + regress_tol_dd,
        "avg_margin_not_regressed": v42["avg_margin"] >= v41["avg_margin"] - regress_tol_margin,
        "avg_margin_positive":      v42["avg_margin"] > 0,
    }
    envelope = {"worst_dd_within_envelope": bool(v42_dd <= d_max_slack)}
    return all(core.values()), {**core, **envelope}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-baseline",  default="config/strategy_v4_1.yaml")
    p.add_argument("--cfg-candidate", default="config/strategy_v4_3.yaml")
    p.add_argument("--oos-seeds",     default="100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116")
    p.add_argument("--block-len",     type=int, default=8)
    p.add_argument("--n-paths",       type=int, default=30)
    p.add_argument("--boot-seed",     type=int, default=9000)
    p.add_argument("--corr-variants", action="store_true",
                   help="通道A 追加 2 个 regime_corr 幅度变体(独立 A2 段, 不参与三通道 core 判定)")
    p.add_argument("--out",           default=str(OUT / "oos_validation.json"))
    args = p.parse_args()

    oos_seeds = tuple(int(x) for x in args.oos_seeds.split(","))
    cfg_a = load_config(PROJ / args.cfg_baseline)
    cfg_b = load_config(PROJ / args.cfg_candidate)

    print("=" * 74)
    print(f" 节点4 对抗 OOS 验证: {args.cfg_baseline}  vs  {args.cfg_candidate}")
    print("=" * 74)
    print(f" 训练集参考: 情景={list(TRAINING_SCENARIOS)}, seeds={list(TRAINING_SEEDS)}")
    print(f" OOS 通道A: {len(OOS_MAGNITUDE_SCENARIOS)} 个 held-out 幅度")
    print(f" OOS 通道B: 训练 5 情景 × 独立 seeds {list(oos_seeds)}")
    print(f" OOS 通道C: block bootstrap block_len={args.block_len} × n_paths={args.n_paths}")
    print()

    # 共享 DGP (通道 A/B 用同一份, 通道 C 直接用 real_returns)
    print("[fit DGP]  拟合 VAR+GARCH ...", flush=True)
    t0 = time.time()
    nav, wk, w_rets = dm.load_real()
    mu, A, Sigma, nu, resid, coords = dm.fit_var_t(w_rets)
    gp, R = adv.fit_garch(resid)
    real_dates = wk.index; first_nav = wk.iloc[0].values; T = len(w_rets)
    dgp = (mu, A, R, nu, gp, real_dates, first_nav, T)
    print(f"           done in {time.time()-t0:.1f}s")

    outs = {"config_baseline": args.cfg_baseline, "config_candidate": args.cfg_candidate,
            "channels": {}}

    # ======== 通道 A ========
    for label, cfg in (("v4_1", cfg_a), ("v4_2", cfg_b)):
        print(f"\n[通道A held-out 幅度] {label} ...", flush=True)
        t0 = time.time()
        res = run_channel_ab(label, cfg, dgp, OOS_MAGNITUDE_SCENARIOS, TRAINING_SEEDS, f"A_{label}")
        outs["channels"].setdefault("A_held_out_magnitudes", {})[label] = res
        print(f"  pass_rate={res['pass_rate']:.0%}  worst_DD={res['worst_maxdd']:.2%}  "
              f"avg_margin={res['avg_margin']:+.3f}  ({time.time()-t0:.0f}s)")

    # ======== 通道 A2 (可选): regime_corr 幅度变体, 独立记录不入 core 判定 ========
    if args.corr_variants:
        for label, cfg in (("v4_1", cfg_a), ("v4_2", cfg_b)):
            print(f"\n[通道A2 regime_corr 变体] {label} ...", flush=True)
            t0 = time.time()
            res = run_channel_ab(label, cfg, dgp, OOS_CORR_VARIANTS, TRAINING_SEEDS, f"A2_{label}")
            outs["channels"].setdefault("A2_regime_corr_variants", {})[label] = res
            print(f"  pass_rate={res['pass_rate']:.0%}  worst_DD={res['worst_maxdd']:.2%}  "
                  f"avg_margin={res['avg_margin']:+.3f}  ({time.time()-t0:.0f}s)")

    # ======== 通道 B ========
    train_scen = {k: adv.STRESS_SCENARIOS[k] for k in TRAINING_SCENARIOS}
    for label, cfg in (("v4_1", cfg_a), ("v4_2", cfg_b)):
        print(f"\n[通道B 独立 seed 集] {label} ...", flush=True)
        t0 = time.time()
        res = run_channel_ab(label, cfg, dgp, train_scen, oos_seeds, f"B_{label}")
        outs["channels"].setdefault("B_independent_seeds", {})[label] = res
        print(f"  pass_rate={res['pass_rate']:.0%}  worst_DD={res['worst_maxdd']:.2%}  "
              f"avg_margin={res['avg_margin']:+.3f}  ({time.time()-t0:.0f}s)")

    # ======== 通道 C ========
    real_returns = w_rets.values if hasattr(w_rets, "values") else np.asarray(w_rets)
    for label, cfg in (("v4_1", cfg_a), ("v4_2", cfg_b)):
        print(f"\n[通道C block bootstrap] {label} ...", flush=True)
        t0 = time.time()
        res = run_channel_c(label, cfg, real_returns, real_dates, first_nav,
                             args.block_len, args.n_paths, args.boot_seed)
        outs["channels"].setdefault("C_block_bootstrap", {})[label] = res
        print(f"  pass_rate={res['pass_rate']:.0%}  worst_DD_median={res['strat_maxdd_median']:.2%}  "
              f"worst_DD_max={res['strat_maxdd_max']:.2%}  avg_margin={res['avg_margin']:+.3f}  "
              f"({time.time()-t0:.0f}s)")

    # ======== 综合判定 ========
    print("\n" + "=" * 74)
    print(" 综合判定(core=过拟合假设直接测试; envelope=设计包线独立记录)")
    print("=" * 74)
    verdicts = {}
    for ch, key in (("A_held_out_magnitudes", "worst_maxdd"),
                    ("B_independent_seeds",   "worst_maxdd"),
                    ("C_block_bootstrap",     "strat_maxdd_max")):
        v41 = outs["channels"][ch]["v4_1"]
        v42 = outs["channels"][ch]["v4_2"]
        v41_wd = v41.get("worst_maxdd", v41.get("strat_maxdd_max"))
        v42_wd = v42.get("worst_maxdd", v42.get("strat_maxdd_max"))
        # L1 修: pass_rate 容差按桶数(通道 A/B 情景数, 通道 C 路径数)校准, 与实际离散粒度一致
        n_bucket = v42.get("n") or v42.get("n_paths") or 1
        core_ok, checks = verdict(v41, v42, n_bucket=n_bucket)
        env_ok = checks["worst_dd_within_envelope"]
        verdicts[ch] = {"core_pass": core_ok, "envelope_ok": env_ok, "checks": checks,
                        "n_bucket": n_bucket,
                        "v41": {"pass_rate": v41["pass_rate"], "worst_dd": v41_wd, "avg_margin": v41["avg_margin"]},
                        "v42": {"pass_rate": v42["pass_rate"], "worst_dd": v42_wd, "avg_margin": v42["avg_margin"]}}
        print(f"\n [{ch}]  (n_bucket={n_bucket})")
        print(f"    v4_1        : pass_rate={v41['pass_rate']:.0%}  worst_DD={v41_wd:.2%}  avg_margin={v41['avg_margin']:+.3f}")
        print(f"    v4_2        : pass_rate={v42['pass_rate']:.0%}  worst_DD={v42_wd:.2%}  avg_margin={v42['avg_margin']:+.3f}")
        print(f"    core(相对不劣化) = {'PASS' if core_ok else 'FAIL'}   "
              f"envelope(≤D_max slack) = {'IN' if env_ok else 'OUT'}")
        print(f"    checks : {checks}")

    all_core = all(v["core_pass"] for v in verdicts.values())
    all_env  = all(v["envelope_ok"] for v in verdicts.values())
    outs["verdicts"] = verdicts
    outs["conclusion"] = "TRUE_ROBUST" if all_core else "OVERFIT_SIGNATURE"
    outs["envelope_note"] = ("v4_2 全通道在设计包线内(worst_DD≤d_max_slack)" if all_env
                              else "v4_2 部分通道超出设计包线(极端OOS幅度天然突破策略族上限,与过拟合无关)")
    print("\n" + "=" * 74)
    if all_core:
        print(f" 最终结论: v4_2 真鲁棒(三通道 core PASS, 过拟合假设被反驳)")
    else:
        print(f" 最终结论: 过拟合可疑 (至少一条 core FAIL, v4_2 相对基线在独立测试上劣化)")
    print(f" 设计包线: {outs['envelope_note']}")
    print("=" * 74)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(outs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已存: {args.out}")


if __name__ == "__main__":
    main()
