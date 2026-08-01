# E2: 策略回测 A/B 对比报告

> Mixed Parkinson vol vs CC-vol baseline | 门禁 #2 判定: **NO-GO**

## 门禁 #2 判定

**结论: NO-GO**

| 门禁条件 | 要求 | 实际 | 判定 |
|---|---|---|---|
| Sharpe 改善 (Mixed) | ≥+0.02 | -0.3811 | ✗ |
| Sharpe 改善 (最优层 Mixed-L1) | ≥+0.02 | -0.0674 | ✗ |
| MaxDD 恶化 | ≤+0.3pp | +14.34pp | ✗ |
| 对抗中位 Sharpe | ≥基线 | -0.0598 | ✗ |

## 实验组对比（真实历史路径）

| 实验 | Sharpe | MaxDD | 年化收益 | Calmar | Δ Sharpe |
|---|---|---|---|---|---|
| Baseline | 1.4878 | 5.84% | 14.52% | 2.486 | +0.0000 |
| Mixed | 1.1067 | 20.18% | 13.81% | 0.684 | -0.3811 |
| Full-P | 1.0656 | 25.55% | 13.46% | 0.527 | -0.4222 |
| Mixed-L1 | 1.4204 | 6.18% | 13.73% | 2.222 | -0.0674 |
| Mixed-L2 | 1.4878 | 5.84% | 14.52% | 2.486 | +0.0000 |
| Mixed-L3 | 1.3735 | 6.78% | 13.60% | 2.005 | -0.1143 |
| Mixed-Def | 1.1477 | 22.26% | 14.60% | 0.656 | -0.3401 |

注: Mixed-L2 = Baseline (L2 inv-vol 独立计算, 与 vol factor 无关)

## 对抗鲁棒性 (7-seed baseline DGP, 中位数)

| 实验 | 中位 Sharpe | 中位 MaxDD | 中位年化 | Δ中位 Sharpe |
|---|---|---|---|---|
| Baseline | 1.3227 | 7.92% | 13.48% | +0.0000 |
| Mixed | 1.2630 | 7.84% | 12.96% | -0.0598 |
| Full-P | 1.2597 | 7.91% | 13.81% | -0.0631 |
| Mixed-L1 | 1.2770 | 7.92% | 12.84% | -0.0457 |
| Mixed-L2 | 1.3227 | 7.92% | 13.48% | +0.0000 |
| Mixed-L3 | 1.3227 | 7.92% | 13.48% | +0.0000 |
| Mixed-Def | 1.3064 | 7.84% | 13.54% | -0.0163 |

## 内部一致性检验

- Full-P (全 Parkinson) Sharpe = 1.0656 vs Baseline = 1.4878: ✓ Full-P 最差 (预期)

## 层级消融分析

| 层级 | 替换列 | ΔSharpe | 作用路径 |
|---|---|---|---|
| Mixed-L1 | ['中证500ETF', '黄金ETF'] | -0.0674 | 仅评分层进攻 ETF 用 Parkinson (cols 2,3) |
| Mixed-L2 | 无 | +0.0000 | N/A (L2 inv-vol 用 w_rets, 与 vol factor 无关) |
| Mixed-L3 | ['中证500ETF'] | -0.1143 | 仅 M3 防御路径 (col 2, ashare boost) |
| Mixed-Def | ['红利低波ETF'] | -0.3401 | 仅 DefAlloc 动态红利比 (col 1) |

Mixed 整体 ΔSharpe 应≈各层之和（非严格加性因交互效应）。

## 方法说明

- **Monkeypatch**: 替换 `src.factors.compute_all_factors` 返回值中的 volatility 列
- **真实路径**: 对替换列直接注入 E0 计算的 Parkinson vol (window=14)
- **对抗路径**: 对替换列的 CC-vol 乘以历史中位 P/CC 比值（保持波动水平响应）
- **Mixed-L2 无法测试**: inv_vol_weights 从 w_rets 独立计算 vol, 不经过 vol factor
- **层级泄漏**: Mixed-L1 替换 cols [2,3] 同时影响 M3 (col 2); 已在报告中注明
