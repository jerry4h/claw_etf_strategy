# 实验: 细圆整配置 (v4_3_rounded_fine) 的数据鲁棒性双重检测

> 任务6 | 2026-07-30 | 配置 `config/experiments/v4_3_rounded_fine.yaml` (def_alloc 0.3492→0.35, step_low 0.0764→0.075, step_high 0.384→0.38, max_def 0.8299→0.83, 其余与生产 v4.3 相同) | 脚本 `scripts/_exp_rounded_robust_boot.py` / `scripts/_exp_rounded_robust_report.py` | 数据 JSON `output/experiments/exp_rounded_robust_adv_eval.json`, `output/experiments/exp_rounded_robust_boot.json`

## 0. 背景与问题

此前实验已证明 rounded_fine 在 **realized 单轴**上与 v4.3 基线无差 (Sharpe 1.489 vs 1.488, MaxDD 5.85% vs 5.84%)。本实验补数据轴双重检测: (1) CCC-GARCH 对抗压力情景 5情景×7seed + 5 机制门禁; (2) moving block bootstrap (block=13周) 200 路径的 Test 2 相对 alpha 判据, 且 seed 序列与生产基线逐路径可配对。问题: **圆整是否在数据轴上同样无损?**

## 1. 对抗侧 — CCC-GARCH 5 压力情景 (seeds 11,22,33,44,55,66,77)

方法: `scripts/evaluate.py --config config/experiments/v4_3_rounded_fine.yaml` (evaluate.py 原生支持 `--config`, 未带 `--save-baseline`, 零副作用)。基线引用 `output/adversarial/baseline_metrics.json` (同一 evaluate.py、同 seeds 11..77、同口径, 未重跑)。

### 1a. realized 复核 (真实历史)

| 配置 | 年化 | MaxDD | Sharpe |
|---|---|---|---|
| v4.3 基线 | 14.52% | 5.84% | 1.488 |
| rounded_fine | 14.49% | 5.85% | 1.489 |

### 1b. 5 情景对比 (7 seeds 中位数; beats = 策略Sharpe中位 ≥ 等权中位)

| 情景 | 策略Sh (rounded / 基线) | 等权Sh (rounded / 基线) | 策略MaxDD (rounded / 基线) | beats_ew (rounded / 基线) |
|---|---|---|---|---|
| baseline *(参照)* | 1.167 / 1.164 | 1.017 / 1.017 | 9.72% / 9.81% | Y / Y |
| vol_stress | 1.006 / 1.004 | 0.877 / 0.877 | 11.04% / 11.03% | Y / Y |
| offense_cooldown | 1.011 / 1.003 | 0.882 / 0.882 | 10.20% / 11.95% | Y / Y |
| bond_bear | 0.903 / 0.903 | 0.803 / 0.803 | 11.04% / 11.11% | Y / Y |
| decorrelation | 1.255 / 1.255 | 1.086 / 1.086 | 8.23% / 8.24% | Y / Y |
| stagflation | 0.845 / 0.842 | 0.743 / 0.743 | 11.42% / 11.41% | Y / Y |

压力情景通过率 (Sharpe口径): rounded 5/5 vs 基线 5/5; 全情景最差 MaxDD: rounded 11.42% vs 基线 11.95% (红线 12%); 最脆弱情景均为 stagflation (rounded 0.845 vs 基线 0.842)。

### 1c. 5 机制门禁判定

| 机制 | 门禁 | rounded 胜率(Sh) / worstDD / 判定 | 基线 胜率(Sh) / worstDD / 判定 |
|---|---|---|---|
| vol_defense | 硬 | 100% / 11.04% / PASS | 100% / 11.03% / PASS |
| defense_asset | 硬 | 100% / 11.04% / PASS | 100% / 11.11% / PASS |
| dispersion | 硬 | 100% / 8.23% / PASS | 100% / 8.24% / PASS |
| composite | 硬 | 100% / 11.42% / PASS | 100% / 11.41% / PASS |
| selection | 软 | 100% / 10.20% / PASS | 100% / 11.95% / PASS |

**evaluate.py 总判定: rounded_fine = PASS** (基线 = PASS); 未过约束: 无。

## 2. bootstrap 侧 — Test 2 (moving block bootstrap, block=13周, 200 路径)

方法: 复用 `robustness_joint.py` 的 `eval_on_bootstrap`/`judge_test2` 依赖链 (`oos.block_bootstrap` seed 确定性), seed 序列与生产基线一致 (seed = 8000 + i, i∈[0,200)); 基线取 `output/robustness/robustness_joint_all_20260729_114702.json` 的 test2_rows (未重跑; verdict 用同一 judge_test2 从其逐路径行重算以补齐 alpha 字段)。

### 2a. seed 复现校验

用 v4.3 生产配置重跑基线 3 个 seed (头/中/尾), 与基线 JSON 对比:

| seed | 本次 Sharpe | 基线 Sharpe | \|Δ\| |
|---|---|---|---|
| 8000 | 1.3482468677 | 1.3482468677 | 0.00e+00 |
| 8047 | 1.7030025859 | 1.7030025859 | 0.00e+00 |
| 8199 | 0.8755240746 | 0.8755240746 | 0.00e+00 |

复现**成功** (逐位一致) → 200 条路径与基线逐路径可配对, 基线直接引用不重跑。

### 2b. 分位数与 alpha 对比 (200 路径, 失败 0)

| 指标 | rounded_fine | v4.3 基线 |
|---|---|---|
| Sharpe P10 / P50 / P90 | 0.886 / 1.260 / 1.714 | 0.886 / 1.261 / 1.713 |
| MaxDD P10 / P50 / P90 | 6.63% / 8.52% / 11.33% | 6.64% / 8.58% / 11.43% |
| 年化 P10 / P50 / P90 | 9.38% / 12.61% / 16.74% | 9.40% / 12.65% / 16.80% |
| 胜率 (策略Sh > 等权Sh) | 95.5% | 96.0% |
| alpha P10 / P50 / P90 | +0.077 / +0.346 / +0.607 | +0.078 / +0.350 / +0.606 |
| 相对 alpha 判据 (胜率≥90% & alpha P10>0) | **PASS** | **PASS** |

### 2c. 逐路径配对差 (rounded − baseline, 同 seed 同路径, n=200)

| 统计量 | ΔSharpe | ΔMaxDD (pp) |
|---|---|---|
| 均值 | +0.0008 | -0.058 |
| P10 / P50 / P90 | -0.0021 / +0.0002 / +0.0036 | -0.073 / — / -0.002 |
| min / max | -0.0532 / +0.0476 | — |

配对解读: ΔSharpe 均值 +0.0008 (std 0.0076), 分布紧贴 0 且轻微偏正 (rounded 优的路径占 54.5%); |ΔSharpe|>0.01 的路径仅 13/200, 极端差 |Δ|max≈0.053 来自个别路径上离散调仓阈值触发时点的微小位移, 无系统性方向。ΔMaxDD 均值 -0.058pp (rounded 略低)。这是圆整无损最强的配对证据: 同一数据路径下两配置几乎逐路径重合。

## 3. 最终结论

**圆整配置在数据轴上通过与基线同级的全部门禁**:

1. 对抗侧: evaluate.py 总判定 **PASS** — 5/5 压力情景 Sharpe 中位胜等权, 4 硬门禁 + 1 软门禁全 PASS, 全情景最差 MaxDD 11.42% < 12% 红线; 逐情景指标与基线差异 ≤0.01 量级 (同 seed 同 DGP 下几乎同轨)。
2. bootstrap 侧: 胜率 95.5% ≥ 90%, alpha P10 +0.077 > 0 → 相对 alpha 判据 **PASS** (基线 96.0% / +0.078); Sharpe/MaxDD/alpha 三组分位数与基线在小数第三位内重合。
3. 配对证据: 200 条同 seed 路径逐路径差 ΔSharpe 均值 +0.0008, P10~P90 = [-0.0021, +0.0036], 以 0 为中心的窄带 → 圆整引入的扰动远小于数据轴自身方差, 统计上不可区分。

结合此前 realized 单轴结果, rounded_fine 圆整在 realized、对抗 (CCC-GARCH)、bootstrap (非参数重采样) 三个数据维度上均无损, 可作为 v4.3 的等价替代配置。

---
*方法论说明: 对抗侧未重跑基线 (baseline_metrics.json 与本次 rounded 运行为同一 evaluate.py、同 seeds 11..77、同 DGP 拟合流程, 口径完全可比, 节省 ~50% 成本); bootstrap 侧通过 3-seed 逐位复现校验后直接引用基线 200 路径。合成/重采样数据临时 CSV 均由被复用函数 try/finally 清理。本实验零生产代码/配置/基线改动。*
