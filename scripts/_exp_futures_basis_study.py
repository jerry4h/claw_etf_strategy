#!/usr/bin/env python3
"""股指期货基差 E0 数据管线 + E1 信号质量评估.

用户诉求: 从股指期货信息(尤其机构对冲痕迹)预测风险.

概念前提(E0 预检已确认): 股指期货是双向合约, 持仓量 OI 是配对总数, 天然不含
call/put 方向份额. 最能反映机构对冲需求的是 **基差贴水** —— 机构用 IC/IM 空头
对冲现货多头, 会把期货价格压到现货之下形成贴水, 贴水深度即对冲需求强度读数.

本脚本只做 E0 + E1, 不做 E2, 不改 src/ 与 config/.

用法:
    python scripts/_exp_futures_basis_study.py --fetch    # 取数并缓存(约 320 次 API)
    python scripts/_exp_futures_basis_study.py            # 读缓存 -> E0 -> E1 -> 报告
    python scripts/_exp_futures_basis_study.py --render-only   # 从 json 重放渲染
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CACHE = PROJECT / "data" / "experiments"
OUT_DIR = PROJECT / "output" / "experiments"
OUT_MD = OUT_DIR / "exp_futures_basis.md"
OUT_JSON = OUT_DIR / "exp_futures_basis.json"
OUT_PNG = OUT_DIR / "futures_basis.png"

# 期货根代码 -> 对应现货指数. IC=中证500(策略持仓标的), IF=沪深300(A股整体代理)
ROOTS = {"IC": "000905.SH", "IF": "000300.SH"}
NAV_PATH = PROJECT / "data" / "all_etfs_nav_latest.csv"

NEAR_EXPIRY_DAYS = 5      # days_left <= 5 时年化基差率分母趋零, 切次月合约
MIN_EXPANDING = 52        # expanding 标准化最小窗口(周)
IC_GATE = 0.03            # |IC| 门禁
T_GATE = 1.5              # |t-stat| 门禁
ROLL_IC_WIN = 52          # 滚动 IC 窗口(周), 仅作稳定性描述, 不用于门禁
FWD_WIN = 4               # 未来风险窗口(周)
ORTHO_GATE = 0.30         # 与现有因子相关上限
N_BOOT = 2000             # block bootstrap 路径数.
#   必须让 p_boot 的分辨率下限 1/N_BOOT 小于 BH 首项阈值 q/m (=0.10/30
#   =0.00333), 否则任何组合在数学上都不可能通过 FDR 第一道阈值,
#   "全灭"会是参数人为造成而非数据结论(200 路径时即如此).
BLOCK = 8                 # block 长度(周), 需 >= FWD_WIN 以保留重叠 target 的自相关
SEED = 20260829


# --------------------------------------------------------------------------
# E0: 取数
# --------------------------------------------------------------------------
def _pro():
    for line in (PROJECT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    import tushare as ts
    return ts.pro_api(os.environ["TUSHARE_TOKEN"])


def _retry(fn, *a, **kw):
    """tushare 限速与偶发网络错误的简单重试。"""
    for attempt in range(4):
        try:
            return fn(*a, **kw)
        except Exception as exc:                      # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
            _ = exc
    return None


def fetch_root(pro, root: str, spot_code: str) -> None:
    """拉取单个期货根代码的全部合约日线 + 主力映射 + 现货指数。"""
    print(f"  [{root}] fut_basic ...", flush=True)
    fb = _retry(pro.fut_basic, exchange="CFFEX", fut_type="1",
                fields="ts_code,symbol,name,list_date,delist_date,fut_code")
    fb = fb[fb.fut_code == root].copy()
    print(f"  [{root}] 合约数 {len(fb)}", flush=True)

    print(f"  [{root}] fut_mapping ...", flush=True)
    mp = _retry(pro.fut_mapping, ts_code=f"{root}.CFX")
    mp = mp[["trade_date", "mapping_ts_code"]].drop_duplicates("trade_date")

    rows = []
    codes = sorted(fb.ts_code.unique())
    for i, code in enumerate(codes, 1):
        d = _retry(pro.fut_daily, ts_code=code)
        if d is not None and len(d):
            rows.append(d)
        if i % 20 == 0:
            print(f"  [{root}] fut_daily {i}/{len(codes)}", flush=True)
        time.sleep(0.12)
    allc = pd.concat(rows, ignore_index=True)

    print(f"  [{root}] index_daily {spot_code} ...", flush=True)
    spot = _retry(pro.index_daily, ts_code=spot_code,
                  start_date="20100101", end_date="20261231")

    lo = root.lower()
    fb.to_csv(CACHE / f"raw_fut_{lo}_basic.csv", index=False)
    mp.to_csv(CACHE / f"raw_fut_{lo}_mapping.csv", index=False)
    allc.to_csv(CACHE / f"raw_fut_{lo}_daily.csv", index=False)
    spot.to_csv(CACHE / f"raw_fut_{lo}_spot.csv", index=False)
    print(f"  [{root}] 缓存完成: 日线 {len(allc)} 行, 主力映射 {len(mp)} 日, "
          f"现货 {len(spot)} 日", flush=True)


def fetch_all() -> None:
    pro = _pro()
    for root, spot in ROOTS.items():
        fetch_root(pro, root, spot)


# --------------------------------------------------------------------------
# E0: 基差构造
# --------------------------------------------------------------------------
def build_basis(root: str) -> tuple[pd.DataFrame, dict]:
    """构造日频主力基差序列.

    近到期(days_left <= NEAR_EXPIRY_DAYS)切换到次月合约, 避免年化分母趋零爆炸.

    Returns:
        (日频基差 DataFrame, E0 诊断 dict)
    """
    lo = root.lower()
    fb = pd.read_csv(CACHE / f"raw_fut_{lo}_basic.csv", dtype={"delist_date": str})
    mp = pd.read_csv(CACHE / f"raw_fut_{lo}_mapping.csv", dtype={"trade_date": str})
    dl = pd.read_csv(CACHE / f"raw_fut_{lo}_daily.csv", dtype={"trade_date": str})
    sp = pd.read_csv(CACHE / f"raw_fut_{lo}_spot.csv", dtype={"trade_date": str})

    delist = dict(zip(fb.ts_code, pd.to_datetime(fb.delist_date, format="%Y%m%d")))
    dl["dt"] = pd.to_datetime(dl.trade_date, format="%Y%m%d")
    dl["delist"] = dl.ts_code.map(delist)
    dl["days_left"] = (dl["delist"] - dl["dt"]).dt.days
    dl = dl[dl.days_left >= 0].copy()

    sp["dt"] = pd.to_datetime(sp.trade_date, format="%Y%m%d")
    spot_close = sp.set_index("dt")["close"].sort_index()

    mp["dt"] = pd.to_datetime(mp.trade_date, format="%Y%m%d")
    main_map = mp.set_index("dt")["mapping_ts_code"].sort_index()

    # 逐交易日: 主力合约 + 次月合约(存续且 delist 严格晚于主力的最近一个)
    by_date = {d: g for d, g in dl.groupby("dt")}
    recs = []
    n_switch = 0
    for d, main_code in main_map.items():
        g = by_date.get(d)
        if g is None or d not in spot_close.index:
            continue
        m = g[g.ts_code == main_code]
        if m.empty:
            continue
        m = m.iloc[0]
        use_code, use_close, use_oi, use_days, switched = (
            m.ts_code, m.close, m.oi, m.days_left, False)
        if m.days_left <= NEAR_EXPIRY_DAYS:
            nxt = g[g.delist > m.delist].sort_values("delist")
            if len(nxt):
                n = nxt.iloc[0]
                use_code, use_close, use_oi, use_days, switched = (
                    n.ts_code, n.close, n.oi, n.days_left, True)
                n_switch += 1
        s = spot_close.loc[d]
        recs.append({
            "dt": d, "code": use_code, "main_code": main_code,
            "fut_close": use_close, "spot_close": s, "oi": use_oi,
            "days_left": use_days, "switched": switched,
            "is_roll": False,
        })

    df = pd.DataFrame(recs).set_index("dt").sort_index()
    df["is_roll"] = df.main_code != df.main_code.shift(1)
    df.loc[df.index[0], "is_roll"] = False

    df["basis"] = df.fut_close - df.spot_close
    df["basis_pct"] = df.fut_close / df.spot_close - 1.0
    df["basis_ann"] = df.basis_pct * 365.0 / df.days_left.clip(lower=1)
    # 即便已切次月, 仍有极少数无次月可切的尾部合约, 一律剔除
    bad = df.days_left <= NEAR_EXPIRY_DAYS
    df.loc[bad, "basis_ann"] = np.nan

    diag = {
        "root": root,
        "n_days": int(len(df)),
        "date_min": str(df.index.min().date()),
        "date_max": str(df.index.max().date()),
        "n_contracts_used": int(df.code.nunique()),
        "n_roll_days": int(df.is_roll.sum()),
        "n_switched_to_next": int(n_switch),
        "pct_switched": round(100.0 * n_switch / max(len(df), 1), 2),
        "n_dropped_near_expiry": int(bad.sum()),
        "pct_dropped": round(100.0 * int(bad.sum()) / max(len(df), 1), 2),
        "basis_ann_nan": int(df.basis_ann.isna().sum()),
        "basis_ann_mean_pct": round(float(df.basis_ann.mean() * 100), 3),
        "basis_ann_median_pct": round(float(df.basis_ann.median() * 100), 3),
        "basis_ann_min_pct": round(float(df.basis_ann.min() * 100), 2),
        "basis_ann_max_pct": round(float(df.basis_ann.max() * 100), 2),
        "pct_discount_days": round(float((df.basis_pct < 0).mean() * 100), 2),
    }
    return df, diag


def e0_roll_jump_check(df: pd.DataFrame) -> dict:
    """校验 basis_ann 在主力换月处不产生系统性跳变.

    基差率分子分母同步换月, 理论上换月不引入跳变 —— 这是基差优于价格信号之处.
    做法: 比较换月日与非换月日的 basis_ann 日度绝对变化分布.
    """
    d = df.basis_ann.diff().abs()
    roll = d[df.is_roll & d.notna()]
    norm = d[(~df.is_roll) & d.notna()]
    if len(roll) < 5 or len(norm) < 5:
        return {"available": False}
    u, p = stats.mannwhitneyu(roll, norm, alternative="greater")
    return {
        "available": True,
        "n_roll": int(len(roll)), "n_normal": int(len(norm)),
        "roll_median_abs_chg_pp": round(float(roll.median() * 100), 4),
        "normal_median_abs_chg_pp": round(float(norm.median() * 100), 4),
        "ratio": round(float(roll.median() / max(norm.median(), 1e-12)), 3),
        "mannwhitney_p": round(float(p), 4),
        "no_jump": bool(p > 0.05),
        "_u": float(u),
    }


def e0_spot_etf_consistency(spot_weekly: pd.Series, nav: pd.DataFrame) -> dict:
    """现货指数 000905.SH 与 510500 ETF 净值的周收益一致性抽查。"""
    etf = nav["中证500ETF"]
    common = spot_weekly.index.intersection(etf.index)
    a = spot_weekly.loc[common].pct_change()
    b = etf.loc[common].pct_change()
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return {"available": False, "n": int(m.sum())}
    c = float(np.corrcoef(a[m], b[m])[0, 1])
    return {
        "available": True, "n": int(m.sum()),
        "corr": round(c, 4), "pass": bool(c > 0.98),
        "spot_ann_ret_pct": round(float((a[m].mean() * 52) * 100), 2),
        "etf_ann_ret_pct": round(float((b[m].mean() * 52) * 100), 2),
    }


# --------------------------------------------------------------------------
# 周频对齐与信号构造
# --------------------------------------------------------------------------
def load_weekly():
    """策略周频价格矩阵(索引为周五) + 配置。"""
    from src.data_loader import load_nav_data, resample_weekly
    from src.strategy import load_config
    cfg = load_config(PROJECT / "config" / "strategy_v4_6.yaml")
    weekly = resample_weekly(load_nav_data(NAV_PATH), anchor=cfg.anchor)
    if cfg.start_date:
        weekly = weekly[weekly.index >= pd.to_datetime(cfg.start_date)]
    if cfg.end_date:
        weekly = weekly[weekly.index <= pd.to_datetime(cfg.end_date)]
    return weekly, cfg


def align_to_weekly(daily: pd.DataFrame, w_index, cols,
                    max_stale_days: int = 7) -> pd.DataFrame:
    """把日频序列对齐到周频索引: 取 <= 该周五的最后一个交易日值.

    无前视: 周五收盘信号, 下周一执行 —— 与策略 rebalance 时序一致.
    陈旧超过 max_stale_days(长假/停牌)一律置 NaN, 不做 ffill 掩盖缺失.
    """
    out = {}
    for c in cols:
        s = daily[c].dropna()
        vals = []
        for d in w_index:
            sub = s.loc[:d]
            if len(sub) == 0:
                vals.append(np.nan)
                continue
            vals.append(sub.iloc[-1] if (d - sub.index[-1]).days <= max_stale_days
                        else np.nan)
        out[c] = vals
    return pd.DataFrame(out, index=w_index)


def expanding_z(s: pd.Series) -> pd.Series:
    """expanding z-score. 只用 [0..t] 已完成数据, 禁止全样本 percentile。"""
    mu = s.expanding(MIN_EXPANDING).mean()
    sd = s.expanding(MIN_EXPANDING).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def expanding_tercile(s: pd.Series) -> pd.Series:
    """expanding 三分位分组. G1=最深贴水, G3=最弱贴水/升水。"""
    grp = pd.Series(index=s.index, dtype=object)
    v = s.values.astype(float)
    for i in range(len(s)):
        if not np.isfinite(v[i]):
            continue
        hist = v[: i + 1]
        hist = hist[np.isfinite(hist)]
        if len(hist) < MIN_EXPANDING:
            continue
        q33, q67 = np.percentile(hist, [100 / 3, 200 / 3])
        grp.iloc[i] = "G1" if v[i] <= q33 else ("G3" if v[i] > q67 else "G2")
    return grp


def build_signals(wk_ic: pd.DataFrame, wk_if: pd.DataFrame) -> pd.DataFrame:
    """六个信号变体. 涉及量的字段一律用 oi, 不用 vol。"""
    sig = pd.DataFrame(index=wk_ic.index)
    sig["basis_ann"] = wk_ic["basis_ann"]
    sig["basis_z_exp"] = expanding_z(wk_ic["basis_ann"])
    sig["basis_chg_1w"] = wk_ic["basis_ann"].diff(1)
    sig["basis_chg_4w"] = wk_ic["basis_ann"].diff(4)
    sig["oi_chg_1w"] = wk_ic["oi"].pct_change(1)
    sig["basis_if_minus_ic"] = wk_if["basis_ann"] - wk_ic["basis_ann"]
    return sig


def fwd_maxdd(px: pd.Series, win: int) -> pd.Series:
    """从 t 起持有 win 周的最大回撤(未来信息, 仅作评估 target)。"""
    out = np.full(len(px), np.nan)
    v = px.values.astype(float)
    for i in range(len(v) - win):
        seg = v[i: i + win + 1]
        if not np.isfinite(seg).all():
            continue
        peak = np.maximum.accumulate(seg)
        out[i] = float((seg / peak - 1.0).min())
    return pd.Series(out, index=px.index)


def build_targets(weekly: pd.DataFrame, strat_nav: pd.Series) -> pd.DataFrame:
    """五个目标变量. 风险类为主判据(用户诉求是预测风险), 收益类为辅。"""
    r500 = weekly["中证500ETF"].pct_change()
    rnas = weekly["纳指ETF"].pct_change()
    fut_r = pd.concat([r500.shift(-k) for k in range(1, FWD_WIN + 1)], axis=1)

    tg = pd.DataFrame(index=weekly.index)
    tg["fwd_vol_4w"] = fut_r.std(axis=1, ddof=1) * np.sqrt(52)
    tg["fwd_maxdd_4w"] = fwd_maxdd(weekly["中证500ETF"], FWD_WIN)
    sn = strat_nav.reindex(weekly.index)
    tg["fwd_strat_dd_4w"] = fwd_maxdd(sn, FWD_WIN)
    tg["fwd_ret_1w"] = r500.shift(-1)
    tg["fwd_nasdaq_ret_1w"] = rnas.shift(-1)
    return tg


# 主判据(风险类) / 辅判据(收益类) / 安慰剂
RISK_TARGETS = ["fwd_vol_4w", "fwd_maxdd_4w", "fwd_strat_dd_4w"]
RET_TARGETS = ["fwd_ret_1w"]
PLACEBO = "fwd_nasdaq_ret_1w"


# --------------------------------------------------------------------------
# E1: 评估
# --------------------------------------------------------------------------
def ts_ic(sig: pd.Series, tgt: pd.Series) -> dict:
    """时序 Spearman IC + block bootstrap 显著性.

    两处对既有惯例的必要偏离, 均有明确理由:

    1. 既有 _exp_share_flow_study.py:223 的 cross_sectional_ic 依赖 5 只资产横截面,
       而基差只对中证500 有对应, 横截面退化为单列无法计算 —— 故改时序口径.

    2. 显著性**不能**用"滚动 IC 序列 + ttest_1samp": 52 周滚动窗口相邻重叠 51/52,
       n 个滚动值的独立样本仅约 n/52 个, 违背 iid 假设会把 |t| 夸大一个数量级
       (实测出现过 |t|=15 这种不可信量级, 且与全样本 IC 符号矛盾).
       改用 block bootstrap: block 长度 >= FWD_WIN 以保留重叠 target 的自相关结构,
       t_boot = |median_IC| / std(bootstrap IC), 并给 95% CI 是否含 0.
       滚动 IC 仅保留 pct_positive 作为稳定性描述, 不参与门禁.
    """
    m = sig.notna() & tgt.notna()
    n = int(m.sum())
    base = {"n": n, "ic": None, "p_full": None, "t_boot": None,
            "ci_lo": None, "ci_hi": None, "ci_excludes_zero": False,
            "boot_median": None, "pct_same_sign": None, "pct_positive": None}
    if n < 60:
        return base
    s, t_ = sig[m].values.astype(float), tgt[m].values.astype(float)
    ic, p_full = stats.spearmanr(s, t_)

    # block bootstrap: 整块重采样保留自相关
    rng = np.random.default_rng(SEED)
    n_blk = int(np.ceil(n / BLOCK))
    boots = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, max(n - BLOCK, 1), size=n_blk)
        idx = np.concatenate([np.arange(st, min(st + BLOCK, n)) for st in starts])[:n]
        c, _p = stats.spearmanr(s[idx], t_[idx])
        if np.isfinite(c):
            boots.append(c)
    b = np.asarray(boots, dtype=float)
    if len(b) < 30:
        return base
    med = float(np.median(b))
    sd = float(b.std(ddof=1))
    lo, hi = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    t_boot = abs(med) / sd if sd > 0 else np.nan
    # bootstrap 双尾 p: 取小侧尾比例的两倍, 下界由路径数决定
    frac_le = float((b <= 0).mean())
    p_boot = 2.0 * min(frac_le, 1.0 - frac_le)
    p_boot = max(p_boot, 1.0 / len(b))
    # 正态近似 p 作交叉验证: 尾部计数受分辨率限制, 近似值不受限但对偏态不敏感
    p_norm = float(2.0 * stats.norm.sf(abs(t_boot))) if np.isfinite(t_boot) else np.nan

    # 滚动 IC 仅作稳定性描述
    roll = []
    for i in range(ROLL_IC_WIN, n + 1):
        c, _p = stats.spearmanr(s[i - ROLL_IC_WIN: i], t_[i - ROLL_IC_WIN: i])
        if np.isfinite(c):
            roll.append(c)
    arr = np.asarray(roll, dtype=float)

    return {
        "n": n, "ic": round(float(ic), 4), "p_full": round(float(p_full), 4),
        "t_boot": None if not np.isfinite(t_boot) else round(float(t_boot), 2),
        "p_boot": round(p_boot, 5),
        "p_boot_resolution": round(1.0 / len(b), 5),
        "p_norm": None if not np.isfinite(p_norm) else float(f"{p_norm:.2e}"),
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "boot_median": round(med, 4),
        "pct_same_sign": round(float((np.sign(b) == np.sign(ic)).mean()), 3),
        "pct_positive": None if len(arr) < 3 else round(float((arr > 0).mean()), 3),
        "n_boot": int(len(b)), "n_roll": int(len(arr)),
    }


def gate_of(r: dict) -> bool:
    """门禁: |IC| >= 0.03 且 block bootstrap 95% CI 不含 0 且 |t_boot| >= 1.5。"""
    ic, t_ = r.get("ic"), r.get("t_boot")
    if ic is None or t_ is None:
        return False
    if not (np.isfinite(ic) and np.isfinite(t_)):
        return False
    return (abs(ic) >= IC_GATE and bool(r.get("ci_excludes_zero"))
            and abs(t_) >= T_GATE)


def bh_fdr(pairs: list[tuple[str, float]], q: float = 0.10) -> dict:
    """Benjamini-Hochberg FDR 校正.

    本次测了 6 信号 x 5 目标 = 30 个组合, 5% 水平下纯随机就期望 1.5 个
    假阳性 —— 不做多重比较校正直接宣布 PASS 是不诚实的.
    """
    valid = [(k, p) for k, p in pairs if p is not None and np.isfinite(p)]
    m = len(valid)
    if m == 0:
        return {"q": q, "n_tested": 0, "survivors": [], "crit_p": None}
    ordered = sorted(valid, key=lambda kv: kv[1])
    survivors, crit_p = [], None
    for i, (k, p) in enumerate(ordered, 1):
        if p <= q * i / m:
            crit_p = p
            survivors = [kk for kk, pp in ordered[:i]]
    return {"q": q, "n_tested": m, "survivors": survivors,
            "crit_p": crit_p,
            "bonferroni_p": round(0.05 / m, 5),
            "ranked": [{"pair": k, "p": p, "threshold": round(q * i / m, 5)}
                       for i, (k, p) in enumerate(ordered, 1)][:8]}


def leading_ok(ll: dict) -> dict:
    """领先性判定: 若 k=0 同步相关的绝对值强于所有 k>=1, 则为同步指标.

    计划要求"确认信号真领先而非同步或滞后" —— 同步指标即使 IC 显著也无
    预测价值, 因为它只是当期行情的镜像.
    """
    k0 = ll.get("0")
    fwd = [ll.get(str(k)) for k in range(1, FWD_WIN + 1)]
    fwd = [x for x in fwd if x is not None]
    if k0 is None or not fwd:
        return {"available": False, "is_leading": False}
    a0 = abs(k0["ic"])
    amax_fwd = max(abs(x["ic"]) for x in fwd)
    return {"available": True,
            "abs_ic_k0": round(a0, 4),
            "max_abs_ic_k_ge_1": round(amax_fwd, 4),
            "is_leading": bool(amax_fwd > a0),
            "note": ("k=0 同步相关强于所有领先项 -> 同步指标, 无预测价值"
                     if amax_fwd <= a0 else "存在领先成分")}


def group_contrast(grp: pd.Series, tg: pd.DataFrame) -> dict:
    """expanding 三分位分组的未来风险对照. G1=最深贴水。"""
    out = {}
    for col in RISK_TARGETS + RET_TARGETS + [PLACEBO]:
        row = {}
        for g in ("G1", "G2", "G3"):
            m = (grp == g) & tg[col].notna()
            row[g] = {"n": int(m.sum()),
                      "mean_pct": round(float(tg[col][m].mean() * 100), 3) if m.sum() else None,
                      "median_pct": round(float(tg[col][m].median() * 100), 3) if m.sum() else None}
        a = tg[col][(grp == "G1") & tg[col].notna()]
        b = tg[col][(grp == "G3") & tg[col].notna()]
        if len(a) >= 10 and len(b) >= 10:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            row["G1_vs_G3"] = {"delta_pp": round(float((a.mean() - b.mean()) * 100), 3),
                               "mannwhitney_p": round(float(p), 4),
                               "significant": bool(p < 0.05), "_u": float(u)}
        else:
            row["G1_vs_G3"] = {"delta_pp": None, "mannwhitney_p": None,
                               "significant": False}
        out[col] = row
    return out


def lead_lag(sig: pd.Series, weekly: pd.DataFrame) -> dict:
    """单信号领先滞后扫描 k=-2..+4: 确认信号真领先而非同步或滞后。"""
    r500 = weekly["中证500ETF"].pct_change()
    out = {}
    for k in range(-2, FWD_WIN + 1):
        tgt = r500.shift(-k)
        m = sig.notna() & tgt.notna()
        if m.sum() < 60:
            out[str(k)] = None
            continue
        c, p = stats.spearmanr(sig[m], tgt[m])
        out[str(k)] = {"ic": round(float(c), 4), "p": round(float(p), 4),
                       "n": int(m.sum())}
    return out


def lead_lag_all(sig: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """逐信号扫描. 领先性必须 per-signal 判定 —— 用单一信号的 k 扫描去否定
    全部信号是不严谨的, 不同变体(水平值 vs 差分)的时序结构本就不同。
    """
    return {c: lead_lag(sig[c], weekly) for c in sig.columns}


def orthogonality(sig: pd.DataFrame, weekly: pd.DataFrame, cfg) -> dict:
    """与现有因子 mom6 / tapered_vol14 / Layer3.5 EWMA corr 的相关, 要求 < 0.30。"""
    from src.factors import compute_all_factors
    from src.data_loader import classify_etfs
    from src.engine_core import compute_crisis_boost_directed

    config_dict = {"factors": {
        "mom_window": cfg.mom_window, "vol_window": cfg.vol_window,
        "vol_ddof": cfg.vol_ddof, "pe_window_years": cfg.pe_window_years,
        "ewma_factors_enabled": cfg.ewma_factors_enabled,
        "ewma_mom_halflife": cfg.ewma_mom_halflife,
        "ewma_vol_halflife": cfg.ewma_vol_halflife,
        "vol_taper_enabled": cfg.vol_taper_enabled,
        "vol_taper_window": cfg.vol_taper_window,
        "vol_taper_len": cfg.vol_taper_len,
        "pvd_enabled": cfg.pvd_enabled, "pvd_window": cfg.pvd_window,
        "pvd_min_periods": cfg.pvd_min_periods,
    }}
    fac = compute_all_factors(weekly, None, config_dict)
    mom = fac["momentum"]["中证500ETF"]
    vol = fac["volatility"]["中证500ETF"]
    nas_vol = fac["volatility"]["纳指ETF"]

    # Layer 3.5 EWMA 危机相关 (backtest.py:185 同口径, 窗口 [i-w, i) 已完成收益)
    etf_names = list(weekly.columns)
    off_idx, _def_idx, _nas = classify_etfs(etf_names)
    w_prices = weekly.values
    w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]
    corr_lv = np.full(len(weekly), np.nan)
    for i in range(cfg.crisis_corr_window + 1, len(weekly)):
        _b, lv = compute_crisis_boost_directed(w_rets, i, off_idx, cfg)
        corr_lv[i] = lv
    ewma_corr = pd.Series(corr_lv, index=weekly.index)

    ref = {"mom6": mom, "tapered_vol14": vol, "nasdaq_vol": nas_vol,
           "ewma_crisis_corr": ewma_corr}
    out = {}
    for sname in sig.columns:
        row = {}
        for rname, rs in ref.items():
            m = sig[sname].notna() & rs.notna()
            if m.sum() < 60:
                row[rname] = None
                continue
            c = float(np.corrcoef(sig[sname][m], rs[m])[0, 1])
            row[rname] = round(c, 4)
        vals = [abs(v) for v in row.values() if v is not None]
        row["max_abs"] = round(max(vals), 4) if vals else None
        row["orthogonal"] = bool(vals and max(vals) < ORTHO_GATE)
        out[sname] = row
    return out



# --------------------------------------------------------------------------
# 裁决
# --------------------------------------------------------------------------
def recompute_verdict(res: dict) -> dict:
    """裁决树. 幂等, 可对旧 json 重放.

    除计划阶段 3 的 C1-C4 外, 补足三道计划已要求计算但未约定如何入裁决的条件:
      (a) 多重比较: 30 个组合下 5% 水平期望 1.5 个假阳性, 靠 BH-FDR 筛
      (b) 正交性: 计划要求 |corr| < 0.30, 不正交即无增量
      (c) 领先性: 计划要求"真领先而非同步", 同步指标无预测价值
    """
    ics = res["ic"]
    ortho = res.get("ortho", {})

    raw_risk, raw_ret = [], []
    pairs = []
    for sname, per_t in ics.items():
        for tname, r in per_t.items():
            pairs.append((f"{sname}->{tname}", r.get("p_boot")))
            if not gate_of(r):
                continue
            if tname in RISK_TARGETS:
                raw_risk.append(f"{sname}->{tname}")
            elif tname in RET_TARGETS:
                raw_ret.append(f"{sname}->{tname}")
    placebo_pass = [f"{s}->{PLACEBO}" for s, per_t in ics.items()
                    if gate_of(per_t.get(PLACEBO, {}))]

    fdr = bh_fdr(pairs, q=0.10)
    res["fdr"] = fdr
    leads = res.get("lead", {})

    surv = set(fdr["survivors"])
    after_fdr = [p for p in raw_risk if p in surv]
    ortho_ok = [p for p in after_fdr
                if ortho.get(p.split("->")[0], {}).get("orthogonal")]
    # 领先性 per-signal 判定
    effective = [p for p in ortho_ok
                 if leads.get(p.split("->")[0], {}).get("is_leading")]
    any_leading = any(v.get("is_leading") for v in leads.values())

    res["funnel"] = {
        "raw_risk_pass": raw_risk,
        "after_fdr": after_fdr,
        "after_orthogonality": ortho_ok,
        "after_leading": effective,
        "any_signal_leading": bool(any_leading),
        "leading_signals": [k for k, v in leads.items() if v.get("is_leading")],
    }

    if effective and not placebo_pass:
        case = "C1 风险类过全部四道筛且安慰剂不显著"
        decision = "GO"
        concl = ("基差信号在 IC 门禁、FDR 校正、正交性、领先性四道筛下均存活, "
                 "且纳指安慰剂不显著, 说明确实捕捉到 A 股特有的对冲痕迹。")
        nxt = ("建议进入 E2。注入点候选: A 股风险状态门控(条件性提升防御), "
               "类比 Layer 3.5 分级应用与 PVD 条件激活路径。本次不改 src/。")
    elif raw_risk and placebo_pass:
        case = "C2 风险类过门禁但安慰剂同样显著"
        decision = "条件 NO-GO"
        concl = ("风险预测力存在, 但纳指(与 A 股股指期货无关的境外资产)同样被显著"
                 "预测, 说明信号很可能是全球风险情绪的代理而非 A 股对冲痕迹。")
        nxt = "需先用纳指收益回归残差化基差信号, 再重测风险类 IC。"
    elif raw_risk and not after_fdr:
        case = "C2b 过原始门禁但 FDR 校正后全灭"
        decision = "NO-GO"
        concl = (f"{len(raw_risk)} 项风险类过单独门禁, 但在 {fdr['n_tested']} 个组合的"
                 f"BH-FDR(q=0.10) 校正下无一存活(Bonferroni 阈值 "
                 f"p={fdr['bonferroni_p']})。这些显著性无法与多重比较假阳性区分。")
        nxt = ("不得改 src/。若要继续, 应先预注册单一假说(一信号一目标)后重新"
               "取样检验, 而非在 30 个组合里挑最显著的。")
    elif after_fdr and not ortho_ok:
        case = "C2c FDR 存活但与现有因子不正交"
        decision = "NO-GO"
        concl = ("存活信号与 tapered_vol14 等现有因子 |corr| >= 0.30, "
                 "其预测力大部分已被 Layer 3 波动映射捕获, 无增量。")
        nxt = "不得改 src/。若要继续, 需先就现有波动因子做残差化再测。"
    elif ortho_ok and not effective:
        bad = [p.split("->")[0] for p in ortho_ok]
        det = "; ".join(
            f"{s}: |IC(k=0)|={leads.get(s, {}).get('abs_ic_k0')} vs "
            f"max|IC(k>=1)|={leads.get(s, {}).get('max_abs_ic_k_ge_1')}"
            for s in dict.fromkeys(bad))
        case = "C2d 过前三道筛但为同步指标"
        decision = "NO-GO"
        concl = (f"逐信号领先滞后扫描显示存活信号的 k=0 同步相关强于所有 "
                 f"k>=1 领先项 ({det}) —— 基差是当期行情的镜像而非前瞻"
                 "指标, 无法用于预测。")
        nxt = "不得改 src/。同步指标可作盘面观察工具, 不入策略。"
    elif raw_ret and not raw_risk:
        case = "C3 仅收益类过门禁, 风险类不过"
        decision = "NO-GO(留档)"
        concl = ("基差对下周方向有边际预测力, 但对未来波动与回撤无预测力 —— "
                 "与用户诉求(预测风险)不符。")
        nxt = "记录留档, 不进 E2。"
    else:
        case = "C4 全部不过门禁"
        decision = "NO-GO"
        concl = ("六个基差/持仓量变体对五个目标均未达门禁, 证伪了"
                 "'股指期货基差贴水可预测本策略未来风险'这一假说。")
        nxt = ("不得改 src/。对照主力份额教训(IC=0.015): 本项目第二个被证伪的"
               "对冲痕迹类信号。基差可保留为观察工具, 不入策略。")

    res["verdict"] = {
        "case": case, "decision": decision, "conclusion": concl, "next_step": nxt,
        "risk_pass_raw": raw_risk, "risk_pass_effective": effective,
        "ret_pass": raw_ret, "placebo_pass": placebo_pass,
        "gate": {"ic": IC_GATE, "t_boot": T_GATE, "fdr_q": 0.10,
                 "ortho": ORTHO_GATE},
    }
    return res


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def render(res: dict) -> None:
    L = []
    A = L.append
    A("# 股指期货基差 E0+E1: 能否从对冲痕迹预测风险")
    A("")
    A(f"- 生成时间: {res['generated_at']}")
    A(f"- 数据截止: {res['data_as_of']}")
    A(f"- 门禁: |IC| >= {IC_GATE} 且 |t-stat| >= {T_GATE}")
    A(f"- 范围: E0 数据管线 + E1 信号质量评估, **不做 E2, src/ config/ 零改动**")
    A("")
    A("## 概念前提")
    A("")
    A("股指期货是双向合约, 持仓量 OI 是配对总数, **不含 call/put 方向份额**。")
    A("最能反映机构对冲需求的是基差贴水: 机构用 IC 空头对冲现货多头, 压低期货价格")
    A("形成贴水, 贴水深度即对冲强度读数。本次评估的正是这条价格线索。")
    A("")

    A("## E0 数据质量")
    A("")
    A("| 项 | IC(中证500) | IF(沪深300) |")
    A("|---|---|---|")
    di, df_ = res["e0"]["IC"], res["e0"]["IF"]
    for key, label in [("n_days", "交易日数"), ("date_min", "起始"),
                       ("date_max", "截止"), ("n_contracts_used", "用到合约数"),
                       ("n_roll_days", "主力换月日数"),
                       ("pct_switched", "近到期切次月占比%"),
                       ("pct_dropped", "剔除样本占比%"),
                       ("basis_ann_nan", "basis_ann NaN"),
                       ("basis_ann_mean_pct", "年化基差均值%"),
                       ("basis_ann_median_pct", "年化基差中位%"),
                       ("pct_discount_days", "贴水日占比%")]:
        A(f"| {label} | {di.get(key)} | {df_.get(key)} |")
    A("")
    rj = res["e0_roll_jump"]
    A("### 换月跳变校验")
    A("")
    for root in ("IC", "IF"):
        r = rj.get(root, {})
        if not r.get("available"):
            A(f"- {root}: 样本不足, 未校验")
            continue
        A(f"- {root}: 换月日 basis_ann 日度绝对变化中位 {r['roll_median_abs_chg_pp']}pp "
          f"vs 非换月日 {r['normal_median_abs_chg_pp']}pp (比值 {r['ratio']}), "
          f"Mann-Whitney p={r['mannwhitney_p']} -> "
          f"{'无系统性跳变' if r['no_jump'] else '**存在跳变, 需警惕**'}")
    A("")
    sc = res["e0_spot_etf"]
    if sc.get("available"):
        A(f"### 标的一致性: 000905.SH 现货 vs 510500 ETF 周收益相关 = {sc['corr']} "
          f"({'PASS' if sc['pass'] else 'FAIL'}, n={sc['n']})")
    A("")

    A("## E1 时序 IC (主判据=风险类)")
    A("")
    A("方法说明 (两处必要偏离既有惯例):")
    A("")
    A("1. 既有 `cross_sectional_ic` 依赖 5 只资产横截面, 而基差只对中证500 有对应,")
    A("   横截面退化为单列。故改用时序 Spearman IC。")
    A("2. 显著性**未**用滚动 IC + `ttest_1samp`: 52 周滚动窗口相邻重叠 51/52,")
    A("   独立样本仅约 n/52 个, 违背 iid 会把 |t| 夸大一个数量级 —— 首版实测出现")
    A(f"   |t|=15 且与全样本 IC 符号矛盾。现改用 block bootstrap ({N_BOOT} 路径,")
    A(f"   block={BLOCK} 周 >= 未来窗口 {FWD_WIN} 周以保留重叠 target 自相关),")
    A("   `t_boot = |median_IC| / std(boot_IC)`, 并要求 95% CI 不含 0。")
    A("")
    A(f"门禁: |IC| >= {IC_GATE} 且 95% CI 不含 0 且 |t_boot| >= {T_GATE}。")
    A("")
    hdr = ["信号"] + RISK_TARGETS + RET_TARGETS + [PLACEBO + "(安慰剂)"]
    A("| " + " | ".join(hdr) + " |")
    A("|" + "---|" * len(hdr))
    for sname, per_t in res["ic"].items():
        cells = [f"`{sname}`"]
        for tname in RISK_TARGETS + RET_TARGETS + [PLACEBO]:
            r = per_t.get(tname, {})
            ic, t_ = r.get("ic"), r.get("t_boot")
            if ic is None or t_ is None:
                cells.append("n/a")
                continue
            ci = "CI含0" if not r.get("ci_excludes_zero") else "CI排0"
            mark = " **PASS**" if gate_of(r) else ""
            cells.append(f"{ic:+.4f} / t={t_:.2f} / {ci}{mark}")
        A("| " + " | ".join(cells) + " |")
    A("")
    A(f"样本量 n={res['ic'][list(res['ic'])[0]][RISK_TARGETS[0]]['n']} 周 (IC 线)。")
    A("")
    A("### 分组对照与领先滞后是更可信的证据")
    A("")
    A("下两节的 Mann-Whitney 与分位对照不依赖观测独立性假设, 在本例中比 IC 表更可信。")
    A("")

    A("## 分组对照 (basis_z_exp expanding 三分位)")
    A("")
    A("G1 = 最深贴水(对冲需求最强, 假说风险最高), G3 = 最弱贴水/升水。")
    A("")
    A("| 目标 | G1 均值% | G2 均值% | G3 均值% | G1-G3 | p | 显著 |")
    A("|---|---|---|---|---|---|---|")
    for col, row in res["groups"].items():
        d = row["G1_vs_G3"]
        dlt = "n/a" if d["delta_pp"] is None else f"{d['delta_pp']:+.3f}pp"
        A(f"| {col} | {row['G1']['mean_pct']} | {row['G2']['mean_pct']} | "
          f"{row['G3']['mean_pct']} | {dlt} | {d['mannwhitney_p']} | "
          f"{'YES' if d['significant'] else 'no'} |")
    A("")

    A("## 领先滞后扫描 (逐信号 vs 中证500 收益)")
    A("")
    A("k<0 为信号滞后于收益(同步/反应), k=0 同步, k>0 为信号领先。")
    A("若 |IC(k=0)| 强于所有 k>=1, 则信号只是当期行情的镜像, 无预测价值。")
    A("")
    ks = list(range(-2, FWD_WIN + 1))
    A("| 信号 | " + " | ".join(f"k={k}" for k in ks) + " | 判定 |")
    A("|---|" + "---|" * (len(ks) + 1))
    leads = res.get("lead", {})
    for sname, per_k in res["lead_lag"].items():
        cells = []
        for k in ks:
            x = per_k.get(str(k))
            cells.append("n/a" if x is None else f"{x['ic']:+.4f}")
        ld = leads.get(sname, {})
        judge = ("领先" if ld.get("is_leading") else "**同步**")
        A(f"| `{sname}` | " + " | ".join(cells) + f" | {judge} |")
    A("")
    n_lead = sum(1 for v in leads.values() if v.get("is_leading"))
    A(f"领先性汇总: {n_lead}/{len(leads)} 个信号存在领先成分。")
    A("")

    A("## 正交性 (要求 |corr| < 0.30)")
    A("")
    A("| 信号 | mom6 | tapered_vol14 | nasdaq_vol | ewma_crisis_corr | max | 正交 |")
    A("|---|---|---|---|---|---|---|")
    for sname, row in res["ortho"].items():
        A(f"| `{sname}` | {row.get('mom6')} | {row.get('tapered_vol14')} | "
          f"{row.get('nasdaq_vol')} | {row.get('ewma_crisis_corr')} | "
          f"{row.get('max_abs')} | {'YES' if row.get('orthogonal') else 'no'} |")
    A("")

    v = res["verdict"]
    fn = res.get("funnel", {})
    fd = res.get("fdr", {})
    A("## 四道筛漏斗")
    A("")
    A("单项 IC 门禁过关不等于有效信号。计划已要求的多重比较/正交性/领先性")
    A("均应作为硬约束, 逐道筛后剩下的才是可用增量。")
    A("")
    A("| 筛 | 存活风险类组合数 | 说明 |")
    A("|---|---|---|")
    A(f"| 1. |IC|>={IC_GATE} 且 CI 排 0 且 |t_boot|>={T_GATE} | "
      f"{len(fn.get('raw_risk_pass', []))} | 单项显著性 |")
    A(f"| 2. BH-FDR q=0.10 ({fd.get('n_tested')} 个组合) | "
      f"{len(fn.get('after_fdr', []))} | 排除多重比较假阳性 |")
    A(f"| 3. 正交性 |corr|<{ORTHO_GATE} | "
      f"{len(fn.get('after_orthogonality', []))} | 排除与现有因子重叠 |")
    A(f"| 4. 领先性 (per-signal) | {len(fn.get('after_leading', []))} | "
      f"排除同步指标 (存在领先成分的信号: "
      f"{fn.get('leading_signals') or '无'}) |")
    A("")
    if fd.get("ranked"):
        A(f"FDR 最显著前几项 (Bonferroni 阈值 p={fd.get('bonferroni_p')}, "
          f"p_boot 分辨率下限 {1.0 / N_BOOT:.5f}):")
        A("")
        A("| 组合 | p_boot | BH 阈值 | 存活 |")
        A("|---|---|---|---|")
        surv = set(fd.get("survivors", []))
        for it in fd["ranked"]:
            A(f"| `{it['pair']}` | {it['p']} | {it['threshold']} | "
              f"{'YES' if it['pair'] in surv else 'no'} |")
        A("")

    A("## 裁决")
    A("")
    A(f"**{v['decision']} — {v['case']}**")
    A("")
    A(v["conclusion"])
    A("")
    A(f"- 过单项门禁的风险类: {v['risk_pass_raw'] or '无'}")
    A(f"- 四道筛后有效的风险类: {v['risk_pass_effective'] or '无'}")
    A(f"- 过门禁的收益类: {v['ret_pass'] or '无'}")
    A(f"- 过门禁的安慰剂: {v['placebo_pass'] or '无'}")
    A("")
    A(f"下一步: {v['next_step']}")
    A("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"报告已写出: {OUT_MD}")


def plot(res: dict, wk_ic: pd.DataFrame | None = None) -> None:
    if wk_ic is None:
        print("跳过绘图(需周频基差序列)")
        return
    spec = importlib.util.spec_from_file_location(
        "hb", PROJECT / "scripts" / "_exp_huijin_position_bounds_study.py")
    hb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hb)
    hb.setup_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    ax.plot(wk_ic.index, wk_ic["basis_ann"] * 100, lw=0.9, color="#c0392b")
    ax.axhline(0, color="#555", lw=0.8, ls="--")
    ax.set_title("IC 主力年化基差率 (负值=贴水=对冲需求)")
    ax.set_ylabel("年化基差率 %")
    ax.grid(alpha=0.3)

    ax = axes[1]
    cols = RISK_TARGETS
    x = np.arange(len(cols))
    w = 0.26
    for j, g in enumerate(("G1", "G2", "G3")):
        vals = [res["groups"][c][g]["mean_pct"] or 0.0 for c in cols]
        ax.bar(x + (j - 1) * w, vals, w, label=f"{g}"
               + (" 最深贴水" if g == "G1" else (" 最弱贴水" if g == "G3" else "")))
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_ylabel("未来风险 均值 %")
    ax.set_title("expanding 三分位分组的未来 4 周风险对照")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    print(f"图已写出: {OUT_PNG}")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="取数并缓存")
    ap.add_argument("--render-only", action="store_true", help="从 json 重放渲染")
    args = ap.parse_args()

    if args.fetch:
        CACHE.mkdir(parents=True, exist_ok=True)
        fetch_all()
        return

    if args.render_only:
        res = recompute_verdict(json.loads(OUT_JSON.read_text(encoding="utf-8")))
        OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                       default=str), encoding="utf-8")
        render(res)
        print(f"\n裁决: {res['verdict']['decision']} — {res['verdict']['case']}")
        return

    from src.backtest import run_backtest

    print("E0: 构造基差 ...")
    daily, e0 = {}, {}
    for root in ROOTS:
        d, diag = build_basis(root)
        daily[root], e0[root] = d, diag
        print(f"  {root}: {diag['n_days']} 日, 年化基差中位 "
              f"{diag['basis_ann_median_pct']}%, 贴水日占比 "
              f"{diag['pct_discount_days']}%")

    weekly, cfg = load_weekly()
    print(f"周频: {len(weekly)} 周 {weekly.index.min().date()} ~ "
          f"{weekly.index.max().date()}")

    wk = {}
    for root in ROOTS:
        wk[root] = align_to_weekly(daily[root], weekly.index,
                                   ["basis_ann", "basis_pct", "oi", "spot_close"])
    e0_roll = {r: e0_roll_jump_check(daily[r]) for r in ROOTS}
    e0_spot = e0_spot_etf_consistency(wk["IC"]["spot_close"], weekly)
    print(f"  换月跳变: IC no_jump={e0_roll['IC'].get('no_jump')} | "
          f"标的一致性 corr={e0_spot.get('corr')}")

    print("E1: 信号与目标 ...")
    sig = build_signals(wk["IC"], wk["IF"])
    res_bt = run_backtest(cfg)
    tg = build_targets(weekly, res_bt.nav_series["nav"])
    print("  信号非空周数: " + ", ".join(
        f"{c}={int(sig[c].notna().sum())}" for c in sig.columns))

    ic_tbl = {}
    for sname in sig.columns:
        ic_tbl[sname] = {t: ts_ic(sig[sname], tg[t]) for t in tg.columns}

    grp = expanding_tercile(sig["basis_z_exp"])
    groups = group_contrast(grp, tg)
    ll = lead_lag_all(sig, weekly)
    ortho = orthogonality(sig, weekly, cfg)

    res = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_as_of": str(weekly.index.max().date()),
        "params": {"near_expiry_days": NEAR_EXPIRY_DAYS,
                   "min_expanding": MIN_EXPANDING, "roll_ic_win": ROLL_IC_WIN,
                   "fwd_win": FWD_WIN, "ic_gate": IC_GATE, "t_gate": T_GATE,
                   "ortho_gate": ORTHO_GATE},
        "e0": e0, "e0_roll_jump": e0_roll, "e0_spot_etf": e0_spot,
        "n_weeks": int(len(weekly)),
        "signal_coverage": {c: int(sig[c].notna().sum()) for c in sig.columns},
        "ic": ic_tbl, "groups": groups, "lead_lag": ll,
        "lead": {c: leading_ok(ll[c]) for c in sig.columns}, "ortho": ortho,
    }
    res = recompute_verdict(res)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                   default=str), encoding="utf-8")
    render(res)
    plot(res, wk["IC"])
    print(f"\n裁决: {res['verdict']['decision']} — {res['verdict']['case']}")


if __name__ == "__main__":
    main()
