# 阶段三-6: A3 联合 DGP 公平性复核 (v4.5-pvd vs v4.6)

> 2026-08-11 | grey_corr_combo 情景 30 seeds | 复用 _exp_volume_price_dgp.py 基建 (量模型已校准)

| 口径 | v4.5 Sh | v4.6 Sh | Δmed | ΔP10 | v4.6 胜率 |
|---|---|---|---|---|---|
| 联合 DGP | 0.728 | 0.723 | -0.004 | -0.027 | 40% |

MaxDD 中位: v4.5 15.10% / v4.6 14.14%

**判定**: v4.6 无结构性损害 = 是 (判据: ΔSharpe ≥ −0.03 且 MaxDD 不恶化 +0.5pp)

机制预期: grey 情景持续高波动 → directed boost 以灰区定向分支为主 (corr 多数 ≤0.60), PE 调制与 PVD 激活受 vol 门限抑制; Δ 应接近 0 或小幅正向。