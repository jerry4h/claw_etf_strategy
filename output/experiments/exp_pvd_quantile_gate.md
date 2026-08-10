# 实验: B2 PVD quantile 门控单点验证

> 2026-08-11 | 基座 v4.5-pvd 生产 config | q_gate=0.5 | bootstrap block=13 n=120 | 脚本 `scripts/_exp_pvd_quantile_gate.py` | 零 src/ 改动

## 1. 设计

PVD 截面离散度 disp_i = std(pvd[i, off_idx]); expanding 分位 q_i (无前视)。离散度高 = PVD 能区分进攻资产, tiebreaker 有意义; 离散度≈0 = 无排序信息。

- BASE: 生产现状 (vol∈[p25,p75] AND top2 gap<0.05)
- V1 叠加门: 现状双门 AND q≥0.5 | V2 替换门: vol 门 AND q≥0.5 (去 gap) | V3 权重缩放: 双门, 权重 pvd_w×q

## 2. Realized

| 变体 | Sharpe | MaxDD | 年化 | 换手 |
|---|---|---|---|---|
| BASE | 1.6028 | 5.80% | 15.52% | 0.1032 |
| V1 | 1.5813 | 5.80% | 15.32% | 0.1018 |
| V2 | 1.5746 | 5.80% | 15.32% | 0.1013 |
| V3 | 1.5738 | 6.18% | 15.21% | 0.0989 |

## 3. Block bootstrap (中位 Sharpe)

| 变体 | bootstrap 中位 | P10 |
|---|---|---|
| BASE | 1.2630 | 0.9122 |
| V1 | 1.2541 | 0.9162 |
| V2 | 1.2453 | 0.9211 |
| V3 | 1.2439 | 0.9324 |

## 4. E2 gate 判定 (ΔSharpe≥+0.01 AND ΔMaxDD≤+0.3pp AND bootstrap 中位不劣)

| 变体 | ΔSharpe | ΔMaxDD(pp) | bootstrap | 判定 |
|---|---|---|---|---|
| V1 | -0.0215 | -0.00 | ✗ | **NO-GO** |
| V2 | -0.0282 | +0.00 | ✗ | **NO-GO** |
| V3 | -0.0291 | +0.39 | ✗ | **NO-GO** |

**结论**: 0/3 变体通过 E2 gate。 quantile 门控未能在不牺牲 realized Sharpe 的前提下改善风险——先验成立 (Sharpe 已 1.60, 边际预期 < +0.03), B2 方向 NO-GO 归档, PVD 维持现状。