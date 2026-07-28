"""
数据流形分析 (Data Manifold Analysis)
=====================================
对 5-ETF 周收益做 VAR(1)+Student-t 生成式建模，构造"可能存在的数据空间"，
把 realized 历史标成多维空间里一个点，扫描参数空间跑真实策略 run_backtest，
用二阶多项式响应面拟合后积分"好区域体积占比"作为过拟合判据。
非参数交叉验证：stationary bootstrap(D1) + 对抗噪声决策翻转(D3)。

零新依赖（numpy/scipy/pandas/matplotlib 均已在 venv）。不动生产基线。
"""
import sys, os, json, io, base64, time, warnings, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})

_here = Path(__file__).resolve().parent
PROJ = _here if (_here / "src").exists() else _here.parent
sys.path.insert(0, str(PROJ))
from src.backtest import run_backtest
from src.strategy import load_config
from src.data_loader import load_nav_data, resample_weekly, ETFS

CFG_PATH = str(PROJ / "config/strategy_v4_3.yaml")
REAL_CSV = PROJ / "data/all_etfs_nav_latest.csv"
OUT = PROJ / "output/manifold"
OUT.mkdir(parents=True, exist_ok=True)

ETF_NAMES = list(ETFS)  # 纳指ETF,红利低波ETF,中证500ETF,黄金ETF,国债ETF
OFF_IDX = [0, 2, 3]
DEF_IDX = [1, 4]
NASD, CSI = 0, 2
START_DATE = "2013-05-17"

# ---- 扫描范围（"可能存在的数据空间"边界）----
BOX = {
    "rho_mult":  (0.0, 3.0),   # 趋势持续性倍数
    "c_mult":    (0.0, 2.0),   # 截面相关倍数
    "sig_mult":  (0.6, 1.6),   # 波动倍数
    "mudef_mult":(0.0, 2.0),   # 防御资产漂移倍数
}
REALIZED = {"rho_mult": 1.0, "c_mult": 1.0, "sig_mult": 1.0, "mudef_mult": 1.0}
AXES = ["rho_mult", "c_mult", "sig_mult", "mudef_mult"]
AX_LABEL = {
    "rho_mult": "trend persistence (x realized)",
    "c_mult": "cross-correlation (x realized)",
    "sig_mult": "volatility (x realized)",
    "mudef_mult": "defensive drift (x realized)",
}

# ======================================================================
# 1. 真实数据 & VAR(1)+t 拟合
# ======================================================================
def load_real():
    nav = load_nav_data(REAL_CSV)
    wk = resample_weekly(nav, anchor="W-MON").dropna()
    w_rets = wk.pct_change().dropna().values
    return nav, wk, w_rets

def fit_var_t(w_rets):
    """VAR(1) OLS + Student-t 自由度估计。返回(mu,A,Sigma,nu,resid,coords)。"""
    R = np.asarray(w_rets, float)
    T, k = R.shape
    Y = R[1:]                      # r_1..r_{T-1}
    X = np.hstack([np.ones((T - 1, 1)), R[:-1]])  # [1, r_{t-1}]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (k+1, k)
    mu = beta[0]                   # (k,)
    A = beta[1:].T                # (k,k)  r_t = mu + A @ r_{t-1} + eps
    resid = Y - X @ beta
    Sigma = np.cov(resid, rowvar=False)
    # Student-t df per margin (median, cap)
    dfm = []
    for j in range(k):
        try:
            _, _, df_j = stats.t.fit(resid[:, j])
            dfm.append(df_j)
        except Exception:
            dfm.append(np.nan)
    nu = float(np.nanmedian(dfm))
    nu = float(np.clip(nu, 4.5, 50.0))
    # realized 坐标
    corr = np.corrcoef(resid, rowvar=False)
    rho = float(np.mean(np.diag(A)))
    c = float(np.mean(np.abs(np.tril(corr, -1))))
    mudef = float(np.mean(mu[DEF_IDX]))
    coords = {"rho": rho, "c": c, "rho_mult_base": rho, "mudef_week": mudef,
              "mudef_ann": mudef * 52, "nu": nu,
              "offensive_ann": [float(mu[i]) * 52 for i in OFF_IDX],
              "eig_A": np.linalg.eigvals(A).tolist()}
    return mu, A, Sigma, nu, resid, coords

def _psd_corr_scale(Sigma, c_mult):
    """按 c_mult 缩放相关结构（对角=realized vol²，off-diag 相关缩放），返回 PSD 协方差。"""
    d = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(d, d)
    lam_min = float(np.linalg.eigvalsh(R).min())
    cap = 1.0 / (1.0 - lam_min) if lam_min < 1 else 2.0
    c_mult = min(c_mult, max(0.0, cap - 1e-3))
    M = (1 - c_mult) * np.eye(Sigma.shape[0]) + c_mult * R
    M = (M + M.T) / 2
    ev = np.linalg.eigvalsh(M)
    if ev.min() < 1e-9:
        M += np.eye(Sigma.shape[0]) * (1e-9 - ev.min())
    return np.outer(d, d) * M, c_mult

def _stationary_A(A, rho_mult):
    """按 rho_mult 缩放 diag(A)，保证谱半径<1。"""
    A2 = A.copy()
    np.fill_diagonal(A2, np.diag(A) * rho_mult)
    sr = max(abs(np.linalg.eigvals(A2)))
    if sr >= 0.99:
        f = 0.98 / sr
        np.fill_diagonal(A2, np.diag(A) * rho_mult * f)
        return A2, float(rho_mult * f), True
    return A2, rho_mult, False

def _mvt_samples(Sig, nu, T, rng):
    """手写多元 Student-t 采样（版本无关）：x = L@z * sqrt(nu/chi2(nu))。"""
    k = Sig.shape[0]
    try:
        L = np.linalg.cholesky(Sig)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sig + np.eye(k) * 1e-9)
    z = rng.standard_normal(size=(T, k))
    g = rng.chisquare(nu, size=T)
    scale = np.sqrt(nu / g)
    return (z @ L.T) * scale[:, None]

def gen_returns(mu, A, Sigma, nu, params, T, seed):
    """生成 T×5 简单周收益。params = {rho_mult,c_mult,sig_mult,mudef_mult}。"""
    rng = np.random.default_rng(seed)
    A2, rho_eff, capped = _stationary_A(A, params["rho_mult"])
    Sig, c_eff = _psd_corr_scale(Sigma, params["c_mult"])
    Sig = Sig * (params["sig_mult"] ** 2)
    mu2 = mu.copy()
    mu2[DEF_IDX] = mu[DEF_IDX] * params["mudef_mult"]
    eps = _mvt_samples(Sig, nu, T, rng)
    r = np.zeros((T, 5))
    r_prev = np.zeros(5)
    for t in range(T):
        r[t] = mu2 + A2 @ r_prev + eps[t]
        r_prev = r[t]
    meta = {"rho_eff": rho_eff, "c_eff": c_eff, "capped": capped}
    return r, meta

def build_nav_df(r, real_dates, real_first_nav):
    """r: (T-1,5) -> nav DataFrame with real_dates index (length T)."""
    nav = np.vstack([real_first_nav, real_first_nav * np.cumprod(1.0 + r, axis=0)])
    return pd.DataFrame(nav, index=real_dates, columns=ETF_NAMES)

# ======================================================================
# 2. 评估器（合成 NAV -> 真实 run_backtest）
# ======================================================================
_CFG_CACHE = {}
def _cfg():
    if "c" not in _CFG_CACHE:
        _CFG_CACHE["c"] = load_config(CFG_PATH)
    return _CFG_CACHE["c"]

def evaluate(nav_df, seed=0, want_records=False):
    """写临时 CSV -> run_backtest -> (sharpe, mdd, ann[, records])。"""
    tmp = OUT / f"synth_{os.getpid()}_{seed}.csv"
    nav_df.to_csv(tmp)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_backtest(_cfg(), start_date=START_DATE, data_path=str(tmp))
        m = res.metrics
        sharpe = float(m.get("sharpe_ratio", np.nan))
        mdd = float(m.get("max_drawdown", np.nan))
        ann = float(m.get("annualized_return") or m.get("annualize_return") or np.nan)
        if want_records:
            recs = res.weekly_records
            return sharpe, mdd, ann, recs
        return sharpe, mdd, ann
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

def _weights_from_records(recs):
    """从 weekly_records 抽逐周权重矩阵 (n_weeks, 5)。"""
    if not recs:
        return None
    rows = []
    for rec in recs:
        w = [float(rec.get(f"weight_{e}", 0.0)) for e in ETF_NAMES]
        s = sum(w) or 1.0
        rows.append([x / s for x in w])
    return np.array(rows)

# ======================================================================
# 3. 并行池
# ======================================================================
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def _n_workers():
    try:
        return max(1, min(mp.cpu_count() - 1, 6))
    except Exception:
        return 2

def _eval_task(args):
    r, dates_ser, first_nav, seed, want_rec = args
    dates = pd.to_datetime(dates_ser)
    nav_df = build_nav_df(r, dates, first_nav)
    if want_rec:
        s, d, a, recs = evaluate(nav_df, seed=seed, want_records=True)
        w = _weights_from_records(recs)
        return {"sharpe": s, "mdd": d, "ann": a, "weights": w}
    s, d, a = evaluate(nav_df, seed=seed)
    return {"sharpe": s, "mdd": d, "ann": a}

def run_pool(tasks):
    """tasks: list of dict(params,seed) -> list of result dict (顺序)。"""
    jobs = []
    for t in tasks:
        r, _ = gen_returns(MU, A, SIG, NU, t["params"], T_GEN, t["seed"])
        jobs.append((r, REAL_DATES_STR, REAL_FIRST_NAV, t["seed"], t.get("want_rec", False)))
    nw = _n_workers()
    out = [None] * len(jobs)
    if nw <= 1 or len(jobs) < 4:
        for i, j in enumerate(jobs):
            out[i] = _eval_task(j)
        return out
    try:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_eval_task, j): i for i, j in enumerate(jobs)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
    except Exception as e:
        print(f"[pool fallback -> serial] {e}")
        for i, j in enumerate(jobs):
            out[i] = _eval_task(j)
    return out

# ======================================================================
# 4. 主流程（各 Part JSON 缓存，可断点续跑）
# ======================================================================
def cache(name):
    return OUT / f"{name}.json"

def save(name, obj):
    with open(cache(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)

def load_cache(name):
    p = cache(name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None

# ---- 全局拟合态（在 main 里设置）----
MU = A = SIG = NU = None
T_GEN = 0
REAL_DATES_STR = None
REAL_FIRST_NAV = None
REAL_WRETS = None
REAL_NAV = None

def init_globals():
    global MU, A, SIG, NU, T_GEN, REAL_DATES_STR, REAL_FIRST_NAV, REAL_WRETS, REAL_NAV
    nav, wk, w_rets = load_real()
    REAL_NAV = nav
    REAL_WRETS = w_rets
    MU, A, SIG, NU, resid, coords = fit_var_t(w_rets)
    T_GEN = len(nav.index) - 1
    REAL_DATES_STR = [str(d.date()) for d in nav.index]
    REAL_FIRST_NAV = nav.iloc[0].values.astype(float)
    return coords, resid

def part_sanity(coords):
    """A. 真实数据复现（验证 data_path 注入正确）。"""
    c = load_cache("sanity")
    if c:
        return c
    nav_df = REAL_NAV  # 原样
    s, d, a = evaluate(nav_df, seed=-1)
    c = {"sharpe": s, "mdd": d, "ann": a, "coords": coords}
    save("sanity", c)
    return c

def part_d1_bootstrap(coords, N=200, block=8):
    """D1: stationary bootstrap on real weekly returns -> Sharpe/DD 分布。"""
    c = load_cache("d1")
    if c:
        return c
    rng = np.random.default_rng(20260726)
    R = REAL_WRETS
    T = len(R)
    tasks = []
    for i in range(N):
        # stationary bootstrap indices
        idx = np.zeros(T, dtype=int)
        idx[0] = rng.integers(T)
        b = 1.0 / block
        for t in range(1, T):
            if rng.random() < b:
                idx[t] = rng.integers(T)
            else:
                idx[t] = (idx[t - 1] + 1) % T
        r_synth = R[idx]
        tasks.append({"params": REALIZED, "seed": 1000 + i, "_r": r_synth})
    # 用直接 r（不走 VAR 生成）
    jobs = [(t["_r"], REAL_DATES_STR, REAL_FIRST_NAV, t["seed"], False) for t in tasks]
    res = []
    nw = _n_workers()
    if nw > 1:
        try:
            with ProcessPoolExecutor(max_workers=nw) as ex:
                futs = [ex.submit(_eval_task, j) for j in jobs]
                res = [f.result() for f in as_completed(futs)]
                res = res[:N] if len(res) >= N else res
        except Exception:
            res = [_eval_task(j) for j in jobs]
    else:
        res = [_eval_task(j) for j in jobs]
    sharpes = sorted([x["sharpe"] for x in res if np.isfinite(x["sharpe"])])
    dds = sorted([abs(x["mdd"]) for x in res if np.isfinite(x["mdd"])])
    def pct(arr, q):
        if not arr:
            return None
        return float(np.percentile(arr, q))
    c = {
        "n": len(sharpes), "block": block,
        "sharpe_p05": pct(sharpes, 5), "sharpe_p50": pct(sharpes, 50),
        "sharpe_p95": pct(sharpes, 95), "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "mdd_p50": pct(dds, 50), "mdd_p95": pct(dds, 95),
        "frac_sharpe_neg": float(np.mean(np.array(sharpes) < 0)),
        "realized_sharpe": coords.get("realized_sharpe"),
    }
    save("d1", c)
    return c

def part_d3_adversarial(coords, N=15):
    """D3: 对真实收益注入递增高斯噪声 -> 决策翻转率 + Sharpe 衰减。"""
    c = load_cache("d3")
    if c:
        return c
    # baseline (no noise) weights
    s0, d0, a0, recs0 = evaluate(REAL_NAV, seed=-2, want_records=True)
    W0 = _weights_from_records(recs0)
    base_top = np.argmax(W0, axis=1) if W0 is not None else None
    sigs = np.round(np.arange(0.0, 1.01, 0.1), 2)
    per_sig = REAL_WRETS.std(0).mean()
    out = {"baseline_sharpe": s0, "baseline_mdd": d0, "sigma_unit_weekly": float(per_sig)}
    rows = []
    for sig in sigs:
        tasks = []
        for i in range(N):
            rng = np.random.default_rng(int(7000 + sig * 100 + i))
            noise = rng.normal(0, sig * per_sig, REAL_WRETS.shape)
            r_noisy = REAL_WRETS + noise
            tasks.append((r_noisy, REAL_DATES_STR, REAL_FIRST_NAV, int(8000 + sig * 100 + i), True))
        res = []
        nw = _n_workers()
        if nw > 1:
            try:
                with ProcessPoolExecutor(max_workers=nw) as ex:
                    futs = [ex.submit(_eval_task, j) for j in tasks]
                    res = [f.result() for f in as_completed(futs)]
            except Exception:
                res = [_eval_task(j) for j in tasks]
        else:
            res = [_eval_task(j) for j in tasks]
        sh = [x["sharpe"] for x in res if np.isfinite(x["sharpe"])]
        flips = []
        for x in res:
            W = x.get("weights")
            if W is not None and base_top is not None and len(W) == len(base_top):
                top = np.argmax(W, axis=1)
                flips.append(float(np.mean(top != base_top)))
        rows.append({
            "sigma": float(sig), "sharpe_mean": float(np.mean(sh)) if sh else None,
            "sharpe_p05": float(np.percentile(sh, 5)) if sh else None,
            "flip_rate_mean": float(np.mean(flips)) if flips else None,
        })
    out["curve"] = rows
    save("d3", out)
    return out

def part_slice2d(N=5):
    """E: realized 处 (rho_mult x c_mult) 稠密 2D 切片。"""
    c = load_cache("slice2d")
    if c:
        return c
    rhos = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    cs = [0.0, 0.5, 1.0, 1.5, 2.0]
    cells = []
    tasks = []
    for ri, rv in enumerate(rhos):
        for ci, cv in enumerate(cs):
            for k in range(N):
                params = {"rho_mult": rv, "c_mult": cv, "sig_mult": 1.0, "mudef_mult": 1.0}
                tasks.append({"params": params, "seed": int(20000 + ri * 100 + ci * 10 + k),
                              "_cell": (ri, ci)})
    res = run_pool(tasks)
    acc = {}
    for t, r in zip(tasks, res):
        ri, ci = t["_cell"]
        if np.isfinite(r["sharpe"]):
            acc.setdefault((ri, ci), []).append((r["sharpe"], abs(r["mdd"]), r["ann"]))
    sh_med = np.full((len(rhos), len(cs)), np.nan)
    dd_med = np.full((len(rhos), len(cs)), np.nan)
    an_med = np.full((len(rhos), len(cs)), np.nan)
    for (ri, ci), lst in acc.items():
        sh_med[ri, ci] = np.median([x[0] for x in lst])
        dd_med[ri, ci] = np.median([x[1] for x in lst])
        an_med[ri, ci] = np.median([x[2] for x in lst])
    c = {"rhos": rhos, "cs": cs,
         "sharpe_median": sh_med.tolist(), "mdd_median": dd_med.tolist(),
         "ann_median": an_med.tolist(),
         "realized_cell": [rhos.index(1.0), cs.index(1.0)]}
    save("slice2d", c)
    return c

def part_lhs_4d(N=150, paths=3):
    """F: Latin-hypercube 4D 采样。"""
    c = load_cache("lhs")
    if c:
        return c
    rng = np.random.default_rng(42)
    pts = np.zeros((N, 4))
    for d in range(4):
        cut = np.linspace(0, 1, N + 1)
        a = rng.permutation(N)
        pts[:, d] = BOX[AXES[d]][0] + (cut[a] + rng.uniform(0, 1, N) / N) * (BOX[AXES[d]][1] - BOX[AXES[d]][0])
        pts[:, d] = np.clip(pts[:, d], BOX[AXES[d]][0], BOX[AXES[d]][1])
    tasks = []
    for i in range(N):
        params = {AXES[j]: float(pts[i, j]) for j in range(4)}
        for k in range(paths):
            tasks.append({"params": params, "seed": int(30000 + i * 10 + k), "_i": i})
    res = run_pool(tasks)
    agg = [[] for _ in range(N)]
    for t, r in zip(tasks, res):
        if np.isfinite(r["sharpe"]):
            agg[t["_i"]].append((r["sharpe"], abs(r["mdd"]), r["ann"]))
    samples = []
    for i, lst in enumerate(agg):
        if lst:
            sh = [x[0] for x in lst]
            dd = [x[1] for x in lst]
            an = [x[2] for x in lst]
            samples.append({**{AXES[j]: float(pts[i, j]) for j in range(4)},
                           "sharpe_med": float(np.median(sh)), "sharpe_mean": float(np.mean(sh)),
                           "mdd_med": float(np.median(dd)), "ann_med": float(np.median(an))})
    c = {"samples": samples, "n": len(samples)}
    save("lhs", c)
    return c

def part_sweep1d(N=5):
    """G: 4 条 1D 扫描（其余固定 realized）。"""
    c = load_cache("sweep1d")
    if c:
        return c
    grids = {
        "rho_mult": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "c_mult":   [0.0, 0.5, 1.0, 1.5, 2.0],
        "sig_mult": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        "mudef_mult":[0.0, 0.5, 1.0, 1.5, 2.0],
    }
    out = {}
    for ax in AXES:
        tasks = []
        for vi, v in enumerate(grids[ax]):
            for k in range(N):
                p = {a: 1.0 for a in AXES}
                p[ax] = v
                tasks.append({"params": p, "seed": int(40000 + AXES.index(ax) * 1000 + vi * 10 + k),
                             "_v": v})
        res = run_pool(tasks)
        agg = {}
        for t, r in zip(tasks, res):
            agg.setdefault(t["_v"], []).append((r["sharpe"], abs(r["mdd"])))
        rows = []
        for v in grids[ax]:
            lst = agg.get(v, [])
            if lst:
                sh = [x[0] for x in lst]
                rows.append({"v": v, "sharpe": float(np.median(sh)),
                             "sharpe_p05": float(np.percentile(sh, 5)),
                             "mdd": float(np.median([x[1] for x in lst]))})
        out[ax] = rows
    c = out
    save("sweep1d", c)
    return c

def part_surface(lhs, samples_mc=200000):
    """H: 二阶多项式响应面拟合 + 好区域体积蒙特卡洛积分。"""
    c = load_cache("surface")
    if c:
        return c
    S = lhs["samples"]
    if len(S) < 30:
        return {"error": "too few LHS samples"}
    X = np.array([[s[a] for a in AXES] for s in S])
    y = np.array([s["sharpe_med"] for s in S])
    # 设计矩阵：1, x_i, x_i^2, x_i x_j
    cols = [np.ones(len(X))]
    names = ["1"]
    for i in range(4):
        cols.append(X[:, i]); names.append(AXES[i])
        cols.append(X[:, i] ** 2); names.append(AXES[i] + "^2")
    for i in range(4):
        for j in range(i + 1, 4):
            cols.append(X[:, i] * X[:, j]); names.append(f"{AXES[i]}*{AXES[j]}")
    D = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    yhat = D @ beta
    r2 = float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))
    # MC 体积积分
    rng = np.random.default_rng(7)
    U = np.zeros((samples_mc, 4))
    for d in range(4):
        U[:, d] = rng.uniform(BOX[AXES[d]][0], BOX[AXES[d]][1], samples_mc)
    Dm = [np.ones(samples_mc)]
    for i in range(4):
        Dm.append(U[:, i]); Dm.append(U[:, i] ** 2)
    for i in range(4):
        for j in range(i + 1, 4):
            Dm.append(U[:, i] * U[:, j])
    Dm = np.column_stack(Dm)
    yp = Dm @ beta
    frac = {
        "sharpe_gt_1.0": float(np.mean(yp > 1.0)),
        "sharpe_gt_0.5": float(np.mean(yp > 0.5)),
        "sharpe_gt_0": float(np.mean(yp > 0)),
        "sharpe_lt_0": float(np.mean(yp < 0)),
        "sharpe_lt_neg0.5": float(np.mean(yp < -0.5)),
    }
    # realized 点预测
    Dr = np.array([1] + [1, 1] * 4 + [1] * 6)  # all realized=1 -> all terms=1
    # 实际上 realized 各轴=1 -> x=1, x^2=1, xi*xj=1
    yp_real = float(np.sum(beta))
    # 失败规则：找出对 Sharpe 最负的轴（线性系数）
    lin = {AXES[i]: float(beta[1 + 2 * i]) for i in range(4)}
    quad = {AXES[i]: float(beta[2 + 2 * i]) for i in range(4)}
    worst_axis = min(lin, key=lin.get)
    c = {"r2": r2, "beta": dict(zip(names, [float(b) for b in beta])),
         "frac": frac, "yp_realized": yp_real,
         "lin_coef": lin, "quad_coef": quad, "worst_axis": worst_axis,
         "n_mc": samples_mc}
    save("surface", c)
    return c

# ======================================================================
# 5. 报告渲染
# ======================================================================
def _b64png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def fig_slice2d(slc):
    rhos, cs = slc["rhos"], slc["cs"]
    Z = np.array(slc["sharpe_median"])
    ri, ci = slc["realized_cell"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[cs[0] - 0.25, cs[-1] + 0.25, rhos[0] - 0.25, rhos[-1] + 0.25],
                   cmap="RdYlGn", vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    for i in range(len(rhos)):
        for j in range(len(cs)):
            v = Z[i, j]
            if np.isfinite(v):
                ax.text(cs[j], rhos[i], f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if v > np.nanmedian(Z) else "white")
    ax.plot(cs[ci], rhos[ri], "o", ms=14, mfc="none", mec="blue", mew=2.5)
    ax.set_xlabel(AX_LABEL["c_mult"]); ax.set_ylabel(AX_LABEL["rho_mult"])
    ax.set_title("Sharpe across (trend-persistence x correlation) at realized vol & defensive-drift")
    ax.set_xticks(cs); ax.set_yticks(rhos)
    fig.colorbar(im, ax=ax, label="Sharpe (median)")
    return _b64png(fig)

def fig_sweep(swp):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(AXES):
        rows = swp[ax]
        xs = [r["v"] for r in rows]
        ys = [r["sharpe"] for r in rows]
        p5 = [r["sharpe_p05"] for r in rows]
        axes[i].plot(xs, ys, "o-", color="tab:blue", label="median")
        axes[i].fill_between(xs, p5, ys, alpha=0.2, color="tab:blue", label="p5..p50")
        axes[i].axvline(1.0, color="red", ls="--", lw=1, label="realized")
        axes[i].axhline(0, color="gray", lw=0.8)
        axes[i].set_title(ax); axes[i].set_xlabel(AX_LABEL[ax]); axes[i].grid(alpha=0.3)
        axes[i].legend(fontsize=7)
    fig.suptitle("1-D sensitivity (other axes at realized)", y=1.02)
    return _b64png(fig)

def fig_d1(d1, realized_sharpe):
    c = load_cache("d1")
    # 重建原始 sharpe 列表需重新读? 用统计量画
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [d1["sharpe_p05"], d1["sharpe_p50"], d1["sharpe_p95"]]
    lo, hi = d1["sharpe_p05"], d1["sharpe_p95"]
    ax.barh(["p5", "p50", "p95"], xs, color=["#d62728", "#2ca02c", "#1f77b4"])
    ax.axvline(realized_sharpe, color="k", ls="--", lw=2, label=f"realized={realized_sharpe:.2f}")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Sharpe"); ax.set_title("D1 stationary bootstrap Sharpe distribution")
    ax.legend()
    return _b64png(fig)

def fig_d3(d3):
    rows = d3["curve"]
    xs = [r["sigma"] for r in rows]
    sh = [r["sharpe_mean"] for r in rows]
    fl = [r["flip_rate_mean"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(xs, sh, "o-", color="tab:blue", label="Sharpe (mean)")
    ax1.set_xlabel("noise sigma (x weekly vol)"); ax1.set_ylabel("Sharpe", color="tab:blue")
    ax1.axhline(0, color="gray", lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(xs, fl, "s--", color="tab:red", label="decision flip rate")
    ax2.set_ylabel("decision flip rate", color="tab:red")
    fig.suptitle("D3 adversarial perturbation: when does the strategy flip?")
    fig.legend(loc="upper right", fontsize=8)
    return _b64png(fig)

def fig_volume(surf):
    f = surf["frac"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Sharpe>1.0\n(strong)", "Sharpe>0.5\n(margin)", "Sharpe>0\n(alive)", "Sharpe<0\n(broken)", "Sharpe<-0.5\n(collapsed)"]
    vals = [f["sharpe_gt_1.0"], f["sharpe_gt_0.5"], f["sharpe_gt_0"], f["sharpe_lt_0"], f["sharpe_lt_neg0.5"]]
    cols = ["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"]
    ax.barh(labels, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v*100:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of plausible data-space volume")
    ax.set_title("Good-region volume fraction = overfitting verdict")
    return _b64png(fig)

def render_html(coords, sanity, d1, d3, slc, swp, surf, oos_sharpe):
    realized_sharpe = sanity["sharpe"]
    # 过拟合判定
    good_frac = surf["frac"]["sharpe_gt_0.5"]
    bad_frac = surf["frac"]["sharpe_lt_0"]
    if good_frac > 0.5:
        verdict, verdict_color = "ROBUST BASIN — edge is real, history not a fluke", "#2ca02c"
    elif good_frac > 0.2:
        verdict, verdict_color = "MODERATE — edge holds in a meaningful region but watch fragility", "#ffbb78"
    else:
        verdict, verdict_color = "FRAGILE SLIVER — realized point is a lucky corner, overfitting risk high", "#d62728"
    imgs = {
        "slice": fig_slice2d(slc),
        "sweep": fig_sweep(swp),
        "d1": fig_d1(d1, realized_sharpe),
        "d3": fig_d3(d3),
        "volume": fig_volume(surf),
    }
    lin = surf["lin_coef"]; worst = surf["worst_axis"]
    # 失败区域规则（由系数符号给出方向）
    rules = []
    for ax in AXES:
        b = lin[ax]
        q = surf["quad_coef"][ax]
        if b < 0 and q < 0:
            rules.append(f"<b>{AX_LABEL[ax]}</b> 越大越伤（线性 {b:+.2f}, 二次 {q:+.2f}）→ 高相关/高趋势下策略失效")
        elif b > 0 and q < 0:
            rules.append(f"<b>{AX_LABEL[ax]}</b> 呈倒U，realized 附近最优但极端值失效（{b:+.2f}/{q:+.2f}）")
        elif b < 0:
            rules.append(f"<b>{AX_LABEL[ax]}</b> 偏低更差（{b:+.2f}）→ 该维不足时策略吃力")
        else:
            rules.append(f"<b>{AX_LABEL[ax]}</b> 偏高更差（{b:+.2f}）→ 该维过强时策略受损")
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>数据流形分析报告 - {pd.Timestamp.today().date()}</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;max-width:1100px;margin:0 auto;padding:24px;background:#f7f8fa;color:#1f2937;line-height:1.6}}
.card{{background:#fff;padding:24px;border-radius:12px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.verdict{{padding:28px;border-radius:14px;color:#fff;text-align:center;margin-bottom:20px}}
.verdict h1{{margin:0 0 8px;font-size:30px}}
h2{{margin-top:0;color:#374151;border-left:4px solid #6366f1;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}
td,th{{border:1px solid #e5e7eb;padding:8px 10px;text-align:left;font-size:14px}}
th{{background:#f3f4f6}}
.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0}}
.kpi div{{background:#f9fafb;padding:12px;border-radius:8px;text-align:center}}
.kpi b{{display:block;font-size:22px;color:#111827}}
.kpi span{{font-size:12px;color:#6b7280}}
img{{max-width:100%;border-radius:8px;margin:8px 0}}
.warn{{background:#fff7ed;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:6px;font-size:13px}}
.note{{color:#6b7280;font-size:13px}}
</style></head><body>
<div class="verdict" style="background:{verdict_color}">
<h1>数据流形过拟合判定</h1>
<div style="font-size:18px">{verdict}</div>
</div>
<div class="card"><h2>1. 决策卡</h2>
<div class="kpi">
<div><b>{good_frac*100:.0f}%</b><span>好区域体积占比(Sharpe&gt;0.5)</span></div>
<div><b>{bad_frac*100:.0f}%</b><span>失效区域占比(Sharpe&lt;0)</span></div>
<div><b>{realized_sharpe:.2f}</b><span>realized Sharpe(全史)</span></div>
<div><b>{surf['yp_realized']:.2f}</b><span>响应面在 realized 点预测</span></div>
</div>
<table>
<tr><th>维度</th><th>realized 值</th><th>含义</th></tr>
<tr><td>趋势持续性 ρ</td><td>{coords['rho']:.3f} (diag(A)均值)</td><td>VAR(1) 自相关，动量燃料</td></tr>
<tr><td>截面相关 c</td><td>{coords['c']:.3f}</td><td>残差相关均值，分散度</td></tr>
<tr><td>Student-t ν</td><td>{coords['nu']:.1f}</td><td>肥尾强度（越大越接近高斯）</td></tr>
<tr><td>防御资产漂移</td><td>年化 {coords['mudef_ann']*100:.2f}%</td><td>红利低波+国债漂移（债券牛市强度）</td></tr>
<tr><td>进攻资产漂移</td><td>{['%.2f%%'%(x*100) for x in coords['offensive_ann']]}</td><td>纳指/中证500/黄金</td></tr>
</table>
<div class="note">A 谱半径={max(abs(np.array(coords['eig_A']))):.3f}（&lt;1 平稳）；VAR(1) 在 realized 局部拟合，响应面 R²={surf['r2']:.3f}。</div>
</div>
<div class="card"><h2>2. 空间热力图 — (趋势持续性 × 截面相关) 在 realized 波动/防御漂移处</h2>
<img src="data:image/png;base64,{imgs['slice']}">
<div class="note">蓝圈=realized 点。绿=好区域，红=失效。该切片显示在"波动与防御漂移锁定为现实值"时，动量燃料与分散度二维平面策略的表现。</div>
</div>
<div class="card"><h2>3. 好区域体积占比 — 过拟合判据</h2>
<img src="data:image/png;base64,{imgs['volume']}">
<div class="note">在合理数据空间 (ρ×[0,3], c×[0,2], σ×[0.6,1.6], μdef×[0,2]) 内蒙特卡洛 {surf['n_mc']:,} 点积分。<b>占比越大=策略在更多可能数据上有效=非偶然；占比小=只靠刚好这组资产这段历史=过拟合。</b></div>
</div>
<div class="card"><h2>4. 1-D 敏感性扫描</h2>
<img src="data:image/png;base64,{imgs['sweep']}">
<div class="note">红虚线=realized。看哪条曲线在 realized 附近陡降=该维脆弱。</div>
</div>
<div class="card"><h2>5. D1 非参数交叉验证 — stationary bootstrap Sharpe 分布</h2>
<img src="data:image/png;base64,{imgs['d1']}">
<div class="note">bootstrap 保留真实波动聚集，与 VAR 参数化流形结论交叉验证。p5>0 则 edge 统计显著；p5 越接近 0 越脆。本次 n={d1['n']}, block={d1['block']}。</div>
</div>
<div class="card"><h2>6. D3 对抗扰动 — 决策翻转曲线</h2>
<img src="data:image/png;base64,{imgs['d3']}">
<div class="note">对真实收益注入递增噪声。蓝线 Sharpe 衰减，红虚线=逐周冠军 ETF 改变的比例。翻转率骤升处即"微小抖动翻转策略"的阈值。</div>
</div>
<div class="card"><h2>7. 失败区域画像（响应面系数解读）</h2>
<ul>{''.join(f'<li>{r}</li>' for r in rules)}</ul>
<div class="note">最负线性轴 = <b>{AX_LABEL[worst]}</b>（系数 {lin[worst]:+.3f}）。即此维偏离 realized 朝不利方向时策略最先垮。</div>
</div>
<div class="warn">
<b>方法论边界（必须读）：</b>VAR(1)+Student-t 是 realized 点附近的<b>局部线性生成模型</b>，极端区域可能失真；故用 D1 bootstrap（保留真实波动聚集）交叉验证。
两结论一致才可信。响应面为二阶多项式近似，R²={surf['r2']:.3f}；R² 低说明非线性强、体积占比仅供参考。
PE 过滤器在当前策略评分路径<b>未消费</b>（仅计算），故合成 NAV 无 PE 失配问题；进攻资产漂移 μ_off 固定为 realized（未做轴），与 μ_def 不对称——若需对称可扩第5轴。
</div>
<div class="note" style="margin-top:24px;border-top:1px solid #e5e7eb;padding-top:12px">
数据来源：项目真实 NAV (data/all_etfs_nav_latest.csv, 2013-05~2026, {T_GEN+1} 周)；生成模型 VAR(1)+Student-t(ν={coords['nu']:.1f})；
回测引擎 src.backtest.run_backtest（端到端，含因子计算）；零生产代码改动。本报告仅供方法论审视，不构成投资建议。
</div>
</body></html>"""
    p = OUT / "data_manifold_report.html"
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p

def main():
    t0 = time.time()
    coords, resid = init_globals()
    # 把 realized sharpe 注入 coords 供 D1 引用
    sanity = part_sanity(coords)
    coords["realized_sharpe"] = sanity["sharpe"]
    save("coords", coords)
    print(f"[A] sanity: Sharpe={sanity['sharpe']:.3f} MDD={sanity['mdd']:.3f} (应≈1.61/0.07)")
    print(f"[fit] rho={coords['rho']:.3f} c={coords['c']:.3f} nu={coords['nu']:.1f} mudef_ann={coords['mudef_ann']*100:.2f}%")

    print("[D1] bootstrap ..."); d1 = part_d1_bootstrap(coords, N=200)
    print(f"   Sharpe p5/p50/p95 = {d1['sharpe_p05']:.3f}/{d1['sharpe_p50']:.3f}/{d1['sharpe_p95']:.3f}  frac<0={d1['frac_sharpe_neg']:.2%}")

    print("[D3] adversarial ..."); d3 = part_d3_adversarial(coords, N=15)
    print(f"   baseline Sharpe={d3['baseline_sharpe']:.3f}; 翻转曲线 {len(d3['curve'])} 点")

    print("[E] 2D slice ..."); slc = part_slice2d(N=5)
    print("[F] LHS 4D ..."); lhs = part_lhs_4d(N=150, paths=3)
    print(f"   {lhs['n']} samples")
    print("[G] 1D sweep ..."); swp = part_sweep1d(N=5)
    print("[H] response surface ..."); surf = part_surface(lhs)
    print(f"   R²={surf['r2']:.3f}  好>0.5={surf['frac']['sharpe_gt_0.5']:.1%}  失效<0={surf['frac']['sharpe_lt_0']:.1%}  worst={surf['worst_axis']}")

    print("[I] render ..."); p = render_html(coords, sanity, d1, d3, slc, swp, surf, sanity["sharpe"])
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {p}")
    print(json.dumps({"verdict_frac_gt05": surf['frac']['sharpe_gt_0.5'],
                      "verdict_frac_lt0": surf['frac']['sharpe_lt_0'],
                      "realized_sharpe": sanity['sharpe'],
                      "yp_realized": surf['yp_realized'],
                      "d1_p05": d1['sharpe_p05']}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
