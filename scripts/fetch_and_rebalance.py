#!/usr/bin/env python3
"""从 tushare 拉取 ETF 周线 → 追加 CSV → 输出周一调仓方案
用法: python scripts/fetch_and_rebalance.py [--amount 500000]"""

import sys, os, math, argparse
from pathlib import Path
import pandas as pd, numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not TOKEN:
    print("[ERROR] TUSHARE_TOKEN 未设置. 执行: export TUSHARE_TOKEN=your_token")
    sys.exit(1)

pro = ts.pro_api(TOKEN)
ETFS = {"513100.SH": "纳指ETF", "512890.SH": "红利低波ETF",
        "510300.SH": "沪深300ETF", "518880.SH": "黄金ETF", "511010.SH": "国债ETF"}
NAMES = list(ETFS.values())
CSV_PATH = PROJECT / "data" / "all_etfs_nav_2013_20260622_scaled.csv"

# ── 参数 ──
p = argparse.ArgumentParser(description="虾池ETF轮动 v3.0 数据拉取+调仓")
p.add_argument("--amount", type=float, default=500000, help="总资金 (元)")
p.add_argument("--csv", type=str, default=str(CSV_PATH), help="CSV路径")
a = p.parse_args()

CSV_PATH = Path(a.csv)
if not CSV_PATH.is_absolute():
    CSV_PATH = PROJECT / CSV_PATH

# ── 策略参数 ──
MOM_W, VOL_W, TOP_N = 1.0, 0.857, 2
INV_VOL_W = 8
DEF_ALLOC, STEP_LOW, STEP_HIGH, MAX_DEF = 0.25, 0.20, 0.35, 0.95
MAX_SINGLE = 0.40
REBAL_THRESH = 0.07
OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]
DEFENSIVE = ["红利低波ETF", "国债ETF"]

# ── 1. 读取现有 CSV ──
df = pd.read_csv(CSV_PATH)
df["日期"] = pd.to_datetime(df["日期"])
last_date = df["日期"].max()
print(f"现有数据: {len(df)} 行, 最新: {last_date.date()}")

# ── 2. 拉取新数据 ──
end_date = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y%m%d")
start_date = max((last_date + pd.Timedelta(days=1)).strftime("%Y%m%d"),
                 (pd.Timestamp.now() - pd.Timedelta(days=14)).strftime("%Y%m%d"))

print(f"拉取: {start_date} ~ {end_date}")

all_new = {}
for code, name in ETFS.items():
    try:
        raw = pro.fund_daily(ts_code=code, start_date=start_date, end_date=end_date)
        if raw is None or raw.empty:
            print(f"  {name}: 无新数据")
            continue
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        # 按周取周五
        weekly = raw.set_index("trade_date").resample("W-FRI").last().dropna(subset=["close"])
        for d, row in weekly.iterrows():
            ds = d.strftime("%Y-%m-%d")
            all_new.setdefault(ds, {})[name] = float(row["close"])
        print(f"  {name}: {len(weekly)} 周新数据 (到 {weekly.index[-1].date()})")
    except Exception as e:
        print(f"  {name}: ERROR — {e}")

if all_new:
    new_dates = set(all_new.keys()) - set(df["日期"].astype(str).values)
    if new_dates:
        for d in sorted(new_dates):
            row = {"日期": d}
            for n in NAMES:
                row[n] = all_new[d].get(n, "")
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"  → 追加 {len(new_dates)} 周, CSV={len(df)}行")
    else:
        print(f"  → 数据已在CSV中, 无需追加")
else:
    print("  → 无新数据拉取")

# ── 3. 调仓计算 ──
nav = df.set_index("日期").sort_index()
for c in NAMES:
    if c in nav.columns:
        nav[c] = pd.to_numeric(nav[c], errors="coerce")
nav = nav[NAMES].ffill()

# 通过引擎标准因子计算 (ddof=0), 与 backtest.py 完全对齐
m4 = calculate_momentum(nav, window=4)
v20 = calculate_volatility(nav, window=20)
# 计算 w_rets 用于 invvol (引擎格式: diff returns)
prices = nav[NAMES].values
wr = pd.DataFrame(np.diff(prices, axis=0) / prices[:-1], index=nav.index[1:], columns=NAMES)

i = len(nav) - 1
date_str = nav.index[i].strftime("%Y-%m-%d")

def get_score(etf, idx):
    mv = m4[etf].iloc[idx]; vv = v20[etf].iloc[idx]
    if pd.isna(mv) or pd.isna(vv): return None
    return MOM_W * mv - VOL_W * vv

def get_alloc(idx):
    sc = {}
    for e in OFFENSIVE:
        s = get_score(e, idx)
        if s is not None: sc[e] = s
    rk = sorted(sc, key=lambda e: sc[e], reverse=True)
    sl = rk[:TOP_N]

    vn = v20["纳指ETF"].iloc[idx]
    if pd.isna(vn): dr = DEF_ALLOC
    elif vn < STEP_LOW: dr = DEF_ALLOC
    elif vn > STEP_HIGH: dr = MAX_DEF
    else: dr = DEF_ALLOC + (vn - STEP_LOW)/(STEP_HIGH - STEP_LOW)*(MAX_DEF - DEF_ALLOC)

    iv = {}
    for e in sl:
        s = wr[e].iloc[max(0, idx-INV_VOL_W+1):idx+1].dropna()
        v = np.std(s.values, ddof=0)*math.sqrt(52) if len(s) >= 3 else 0.20
        iv[e] = 1.0/max(v, 0.05)
    t = sum(iv.values())
    wts = {e: w/t for e, w in iv.items()} if t > 0 else {e: 1.0/len(sl) for e in sl}

    al = {}
    for e in DEFENSIVE: al[e] = dr/len(DEFENSIVE)
    ot = 1.0 - dr
    for e, w in wts.items():
        al[e] = al.get(e, 0) + w*ot
    for e in al: al[e] = min(al[e], MAX_SINGLE)
    tot = sum(al.values())
    if tot < 1.0:
        dt = sum(al.get(e,0) for e in DEFENSIVE)
        if dt > 0:
            for e in DEFENSIVE: al[e] += (1.0-tot)*al[e]/dt
    return al, sc, sl, dr

alloc, sc, sel, dr = get_alloc(i)
vn = v20["纳指ETF"].iloc[i]

# prev
prev_al = {}
if i > 20:
    prev_al, _, _, _ = get_alloc(i-1)

# ── 4. 输出 ──
print(f"\n{'='*70}")
print(f" 虾池ETF轮动 v3.0 — 周一调仓方案")
print(f"{'='*70}")
print(f" 基准日: {date_str} (本周五净值)")
print(f" 调仓日: 下周一")
print(f" 数据: {CSV_PATH.name} | {len(nav)}周 | {nav.index[0].date()} ~ {nav.index[-1].date()}")

print(f"\n Layer 1 买什么:  score = mom4 − {VOL_W}×vol20")
for e in sorted(sc, key=lambda x: sc[x], reverse=True):
    mv = m4[e].iloc[i]; vv = v20[e].iloc[i]
    tag = " ← TOP" if e in sel else ""
    print(f"  {e:<10s}  mom4={mv*100:+.2f}%  vol20={vv*100:5.1f}%  score={sc[e]:+.4f}{tag}")

print(f"\n Layer 3 防多少:  纳指 vol20={vn*100:5.1f}% → 防御比例={dr*100:5.0f}%")
print(f"\n Layer 2 买多少:  inv-vol{INV_VOL_W} 权重")

print(f"\n ── 下周一持仓 (¥{a.amount:,}) ──")
for e in NAMES:
    w = alloc.get(e, 0)
    if w > 0.001:
        print(f"  {e:<10s} {w*100:>5.1f}%  ≈ ¥{w*a.amount:>8,.0f}")
print(f"  {'合计':<10s} {sum(alloc.values())*100:>5.1f}%  ≈ ¥{a.amount:,}")

if prev_al:
    mc = max(abs(alloc.get(e,0) - prev_al.get(e,0)) for e in set(alloc)|set(prev_al))
    if mc < REBAL_THRESH:
        print(f"\n 调仓阈值 7%: 最大变化 {mc*100:.1f}% → 不调仓")
    else:
        print(f"\n 调仓阈值 7%: 最大变化 {mc*100:.1f}% → 🔄 执行调仓")
        print(f"\n ── 操作 ──")
        for e in alloc:
            pw = prev_al.get(e,0)*100; cw = alloc[e]*100; dw = cw-pw
            act = "买入" if dw>0.5 else ("卖出" if dw<-0.5 else "—")
            print(f"  {e:<10s} {pw:>6.1f}%→{cw:>6.1f}% ({dw:>+5.1f}%) {act}")

print(f"\n{'='*70}")
print(" ✅ 完成 — 下周一按此调仓")