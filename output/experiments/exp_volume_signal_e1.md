# E1-Volume: 成交量信号增量价值评估报告

> 5 个量因子 × 5 只 ETF 周频 | 门禁判定: **GO**

## 门禁判定

**结论: GO**

GO 因子: price_volume_divergence

CONDITIONAL 因子: volume_ma_ratio

| 因子 | |IC| | |t-stat| | 正交? | 判定 |
|---|---|---|---|---|
| volume_change | 0.0090 | 0.45 | ✓ | — |
| volume_ma_ratio | 0.0364 | 1.73 | ✓ | COND |
| price_volume_divergence | 0.0528 | 2.53 | ✓ | GO |
| turnover_intensity | 0.0096 | 0.52 | ✗ | — |
| volume_volatility | 0.0011 | 0.05 | ✓ | — |

## 0. 数据覆盖

| ETF | 日频天数 | 起止日期 | 周频有效 |
|---|---|---|---|
| 纳指ETF | 3211 | 2013-05-15 ~ 2026-07-30 | 677 周 |
| 红利低波ETF | 1824 | 2019-01-18 ~ 2026-07-30 | 385 周 |
| 中证500ETF | 3248 | 2013-03-15 ~ 2026-07-30 | 677 周 |
| 黄金ETF | 3162 | 2013-07-29 ~ 2026-07-30 | 666 周 |
| 国债ETF | 3244 | 2013-03-25 ~ 2026-07-30 | 677 周 |

## 1. Rank IC (截面秩相关)

| 因子 | mean IC | std IC | t-stat | IR | IC>0 占比 | N周 |
|---|---|---|---|---|---|---|
| volume_change | 0.0090 | 0.5159 | 0.45 | 0.018 | 46.5% | 664 |
| volume_ma_ratio | 0.0364 | 0.5404 | 1.73 | 0.067 | 49.4% | 656 |
| price_volume_divergence | 0.0528 | 0.5351 | 2.53 | 0.099 | 52.0% | 659 |
| turnover_intensity | 0.0096 | 0.4809 | 0.52 | 0.020 | 45.9% | 665 |
| volume_volatility | 0.0011 | 0.5391 | 0.05 | 0.002 | 47.5% | 657 |

门禁标准: |IC| ≥ 0.05 且 |t| ≥ 2.0 → GO; |IC| ∈ [0.03,0.05] → CONDITIONAL

## 2. 与现有因子正交性

| 因子 | corr(momentum) | corr(volatility) | 正交(<0.30) |
|---|---|---|---|
| volume_change | +0.0129 | -0.0261 | ✓ |
| volume_ma_ratio | +0.0236 | +0.0090 | ✓ |
| price_volume_divergence | +0.2872 | -0.0595 | ✓ |
| turnover_intensity | +0.0243 | +0.3459 | ✗ |
| volume_volatility | -0.0090 | +0.0064 | ✓ |

## 3. 因子自相关 AC(1)

| 因子 | 均值 AC(1) | 解读 |
|---|---|---|
| volume_change | -0.326 | 低/噪声 |
| volume_ma_ratio | 0.510 | 高持续性 |
| price_volume_divergence | 0.857 | 高持续性 |
| turnover_intensity | 0.862 | 高持续性 |
| volume_volatility | 0.397 | 中等 |

## 4. 分组回测 (top-2 vs bottom-2)

| 因子 | Top年化 | Bottom年化 | 多空价差年化 | 价差t | 正向率 |
|---|---|---|---|---|---|
| volume_change | 11.43% | 10.13% | 1.29% | 0.29 | 49.7% |
| volume_ma_ratio | 13.37% | 4.03% | 9.34% | 1.93 | 51.5% |
| price_volume_divergence | 14.11% | 5.64% | 8.46% | 1.83 | 53.1% |
| turnover_intensity | 7.80% | 9.32% | -1.51% | -0.36 | 51.0% |
| volume_volatility | 7.53% | 9.24% | -1.71% | -0.37 | 50.5% |

## 5. QDII (纳指) vs 境内 ETF

| 因子 | 纳指 IC | 境内均值 IC | 差值 | 异常? |
|---|---|---|---|---|
| volume_change | -0.0021 | -0.0444 | +0.0423 | ✓ |
| volume_ma_ratio | -0.0039 | -0.0298 | +0.0259 | ✓ |
| price_volume_divergence | 0.0407 | 0.0430 | -0.0023 | ✓ |
| turnover_intensity | -0.0325 | -0.0415 | +0.0090 | ✓ |
| volume_volatility | -0.0108 | -0.0026 | -0.0083 | ✓ |

## 6. 经济直觉验证

- 放量周 (vol_ma_ratio > 1.5) 后一周均值收益年化: 7.20% (N=471)
- 缩量周 (vol_ma_ratio < 0.7) 后一周均值收益年化: 15.21% (N=742)
- 解读: 缩量买入 表现更优

## 关键洞察

ETF 成交量反映的是流动性/套利行为而非方向性信息。
在仅 5 只 ETF 的窄截面中，量因子的截面区分能力天然受限。
即便单因子 IC 显著，在现有策略（动量+波动率+PE）基础上的边际增量仍需回测验证。
