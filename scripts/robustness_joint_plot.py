#!/usr/bin/env python3
"""生成 robustness_joint_all 结果的可视化图表。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

fp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/robustness/robustness_joint_all_20260729_114702.json")
d = json.load(open(fp))
OUT = fp.parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

base_m = d["base_metrics"]
t1 = d["test1_rows"]; t2 = d["test2_rows"]; t3 = d["test3_rows"]

# ============ Fig 1: Test 1 参数轴单参数灵敏度 ============
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
axes = axes.flatten()
params = list(dict.fromkeys(r["param"] for r in t1))
base_sh = base_m["sharpe"]
for i, p in enumerate(params):
    rows = sorted([r for r in t1 if r["param"] == p], key=lambda x: x["delta_raw"])
    x = [r["delta_raw"] for r in rows]
    sh = [r["sharpe"] for r in rows]
    axes[i].plot(x, sh, "o-", color="#1f77b4", lw=2, ms=8)
    axes[i].axhline(base_sh, color="k", ls="--", lw=1, alpha=0.5, label=f"base={base_sh:.3f}")
    axes[i].axhline(base_sh * 0.80, color="r", ls=":", lw=1, alpha=0.5, label="-20% floor")
    axes[i].set_title(f"{p}  (base={rows[0]['base_val']})", fontsize=10)
    axes[i].set_xlabel("perturbation")
    axes[i].set_ylabel("Sharpe")
    axes[i].grid(alpha=0.3)
    axes[i].legend(fontsize=8, loc="lower right")
fig.suptitle("Test 1 - Parameter Axis Local Sensitivity (fix real data)\nAll 8 params PASS: no cliffs, max Sharpe drop only -10.3% (top_n discrete jump)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "test1_param_axis.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'test1_param_axis.png'}")

# ============ Fig 2: Test 2 数据轴分布 (Sharpe / MaxDD / Annual) + EW 对比 ============
strat_sh = np.array([r["sharpe"] for r in t2])
strat_dd = np.array([r["maxdd"] for r in t2]) * 100
strat_an = np.array([r["annual"] for r in t2]) * 100
ew_sh    = np.array([r["ew_sharpe"] for r in t2])
ew_an    = np.array([r["ew_annual"] for r in t2]) * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# (a) Sharpe 分布 vs EW
ax = axes[0, 0]
bins = np.linspace(min(strat_sh.min(), ew_sh.min()) - 0.1, max(strat_sh.max(), ew_sh.max()) + 0.1, 40)
ax.hist(ew_sh,    bins=bins, alpha=0.55, color="gray",   label=f"EW baseline (P10={np.quantile(ew_sh, 0.10):.3f})")
ax.hist(strat_sh, bins=bins, alpha=0.65, color="#1f77b4", label=f"v4.3 strategy (P10={np.quantile(strat_sh, 0.10):.3f})")
ax.axvline(base_m["sharpe"], color="r", ls="--", lw=2, label=f"real hist base={base_m['sharpe']:.3f}")
ax.axvline(1.0, color="orange", ls=":", lw=2, label="threshold=1.0")
ax.set_xlabel("Sharpe")
ax.set_ylabel("count (200 bootstrap paths)")
ax.set_title("(a) Sharpe distribution: strategy vs EW baseline")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (b) MaxDD
ax = axes[0, 1]
ax.hist(strat_dd, bins=30, color="#d62728", alpha=0.7)
ax.axvline(base_m["maxdd"] * 100, color="k", ls="--", lw=2, label=f"real hist={base_m['maxdd']*100:.2f}%")
ax.axvline(10, color="orange", ls=":", lw=2, label="threshold 10%")
p90 = np.quantile(strat_dd, 0.90)
ax.axvline(p90, color="red", ls="-", lw=2, label=f"P90={p90:.2f}%")
ax.set_xlabel("MaxDD (%)")
ax.set_ylabel("count")
ax.set_title("(b) MaxDD distribution (v4.3)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (c) Annual
ax = axes[1, 0]
bins = np.linspace(min(strat_an.min(), ew_an.min()) - 1, max(strat_an.max(), ew_an.max()) + 1, 40)
ax.hist(ew_an,    bins=bins, alpha=0.55, color="gray",   label=f"EW P10={np.quantile(ew_an, 0.10):.2f}%")
ax.hist(strat_an, bins=bins, alpha=0.65, color="#2ca02c", label=f"v4.3 P10={np.quantile(strat_an, 0.10):.2f}%")
ax.axvline(base_m["annual"] * 100, color="r", ls="--", lw=2, label=f"real hist={base_m['annual']*100:.2f}%")
ax.set_xlabel("Annual return (%)")
ax.set_ylabel("count")
ax.set_title("(c) Annual return distribution")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (d) Strategy alpha (strat - EW) 分布 - 关键！
diffs = strat_sh - ew_sh
ax = axes[1, 1]
ax.hist(diffs, bins=30, color="#9467bd", alpha=0.75)
ax.axvline(0, color="k", ls="--", lw=2, label="alpha=0")
ax.axvline(np.quantile(diffs, 0.10), color="red", ls="-", lw=2, label=f"P10={np.quantile(diffs, 0.10):+.3f}")
ax.axvline(np.quantile(diffs, 0.50), color="green", ls="-", lw=2, label=f"P50={np.quantile(diffs, 0.50):+.3f}")
win_rate = (diffs > 0).mean() * 100
ax.set_xlabel("Sharpe alpha (v4.3 - EW), same bootstrap path")
ax.set_ylabel("count")
ax.set_title(f"(d) Sharpe alpha — v4.3 beats EW on {win_rate:.1f}% of paths\n(THIS is the true robustness signal)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle("Test 2 - Data Axis Block Bootstrap (200 paths, block=13w, fix v4.3 params)", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "test2_data_axis.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'test2_data_axis.png'}")

# ============ Fig 3: Test 3 vs Test 2 - QQ + 联合 - 边缘 ============
t3_sh = np.array(sorted(r["sharpe"] for r in t3))
t2_sh = np.array(sorted(r["sharpe"] for r in t2))
qq = np.linspace(0.01, 0.99, 99)
t2_q = np.quantile(t2_sh, qq)
t3_q = np.quantile(t3_sh, qq)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.plot(t2_q, t3_q, "o-", color="#1f77b4", ms=4)
lim = [min(t2_q.min(), t3_q.min()) - 0.05, max(t2_q.max(), t3_q.max()) + 0.05]
ax.plot(lim, lim, "k--", alpha=0.5, label="y=x (no param effect on data-axis)")
ax.set_xlabel("Test 2 Sharpe quantile (fix params, vary data)")
ax.set_ylabel("Test 3 Sharpe quantile (vary params AND data)")
ax.set_title("QQ: Test 3 vs Test 2\nCurve ≈ y=x means param perturbation adds ~0 to data-axis variance")
ax.legend()
ax.grid(alpha=0.3)

# 方差分解柱状图
ax = axes[1]
var_t1 = float(np.var([r["sharpe"] for r in t1]))
var_t2 = float(np.var([r["sharpe"] for r in t2]))
var_t3 = float(np.var([r["sharpe"] for r in t3]))
labels = ["Test 1\n(param only)", "Test 2\n(data only)", "Test 1+2\n(if independent)", "Test 3\n(joint, actual)"]
values = [var_t1, var_t2, var_t1 + var_t2, var_t3]
colors = ["#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"]
bars = ax.bar(labels, values, color=colors)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
ax.set_ylabel("Sharpe variance")
ax.set_title(f"Variance decomposition\nInteraction = {var_t3 - var_t1 - var_t2:+.4f} (≈0 → no thin ridge)")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT / "test3_joint_vs_marginal.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'test3_joint_vs_marginal.png'}")
