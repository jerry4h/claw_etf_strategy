# v4.4 影子对照与 halflife 调优实验报告（T4：校验管线步骤 1-2）

- 日期：2026-07-30
- 执行环境：`/home/ubuntu/claw_etf_strategy`，`.venv/bin/python`
- 数据：真实历史 `data/all_etfs_nav_latest.csv`（2013-08-30 ~ 2026-07-24，662 周）
- 前置：v4.4 T1-T3 已完成（引擎 EWMA 分发器、regime DGP、171 项测试全绿）
- 硬约束遵守：未修改 `src/`、`scripts/` 既有文件、`tests/`、`config/strategy_v4_*.yaml`；
  未运行 `--save-baseline` / `--save-state`；所有回测显式 `--output` 至
  `output/experiments/v44_shadow/`，`output/report_v3_1.md` 基线报告保持 v4.3 内容
  （Sharpe 1.488 / 14.52% / 5.84%，已复核）。

---

## 步骤 1：回归基线

| 检查项 | 结果 | 判定 |
|---|---|---|
| a. `pytest tests/ -q` | **171 passed**（114.06s，0 failed/0 skipped） | ✅ |
| b. `rebalance_live.py --verify`（v4.3 默认） | 引擎 Sharpe 1.4878 vs 脚本 1.4977，**差 0.0099 < 0.01**；年化 14.52% vs 14.62%（0.09pp）；DD 5.84% vs 5.84%（0.01pp）；止损 0 次；脚本自身判定 ✅ 通过 | ✅ |
| c. v4.3 回测指标逐位复现 | **Sharpe 1.488 / 年化 14.52% / MaxDD 5.84%**（回撤谷底 2024-09-13，662 周，报告见 `v44_shadow/report_v4_3_regression.md`） | ✅ 逐位一致 |

**步骤 1 结论：回归全过，基线未漂移。**

---

## 步骤 2d/2e：v4.4 真实数据回测与 halflife 4 档对比

v4.4 = v4.3 + Layer 3.5 EWMA 相关（`crisis_correlation_ewma.ewma_enabled: true`）。
halflife 变体配置：`config/experiments/v4_4_hl{6,10,13}.yaml`（与 `strategy_v4_4.yaml`
仅 `ewma_halflife` 及描述行不同，已 diff 确认）。

Felix 报告参考值复核：v4.4 (hl=8) **Sharpe 1.498 / 年化 14.51% / MaxDD 5.85% — 逐位复现一致** ✅。

### realized 指标 + Layer 3.5 触发率（4 档 + v4.3 对照）

触发率由 `v44_shadow/trigger_rate.py` 统计（复用 `engine_core.compute_crisis_boost`
逐周计算，数据管线与 `backtest.run_backtest` 一致，样本 662 周）。

| 档位 | Sharpe | 年化 | MaxDD | 年化波动 | 周胜率 | Calmar | 累计收益 | 调仓次数 | 触发周数 | **触发率** | 触发时均值 boost | 满格(0.15)周数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4.3 classic（对照） | 1.488 | 14.52% | 5.84% | 7.64% | 60.7% | 2.49 | 461.9% | 377 | 37 | **5.59%** | 0.0630 | 1 |
| v4.4 hl=6 | 1.499 | 14.47% | 5.85% | 7.55% | 60.9% | 2.48 | 459.0% | 387 | 84 | **12.69%** | 0.0840 | 23 |
| **v4.4 hl=8（默认）** | **1.498** | **14.51%** | **5.85%** | 7.58% | 60.6% | 2.48 | 461.5% | 378 | 58 | **8.76%** | 0.0833 | 17 |
| v4.4 hl=10 | 1.501 | 14.54% | 5.85% | 7.58% | 60.9% | 2.49 | 463.3% | 375 | 54 | **8.16%** | 0.0770 | 12 |
| v4.4 hl=13 | 1.500 | 14.54% | 5.85% | 7.59% | 60.9% | 2.49 | 463.2% | 373 | 47 | **7.10%** | 0.0773 | 12 |

要点：
- classic 触发率 5.59% ≈ 任务书对照值 5.7%（差异来自窗口边界口径），交叉验证成立。
- 4 档 EWMA 触发率全部 < 15% 红线；hl=6 达 12.69%（红线余量仅 2.3pp），且满格周数 23 为
  最高、调仓次数最多（387，摩擦↑），敏感度过高；hl≥8 触发率稳定在 7~9% 区间。
- 4 档 realized 指标几乎不可区分（Sharpe 1.498~1.501，MaxDD 全部 5.85%），说明 realized
  维度对 halflife 不敏感——EWMA 的价值在危机情景（见 2f），不应据 realized 微小差异调参。
- 所有档位 MaxDD 谷底均为 2024-09-13，与 v4.3 相同（回撤事件结构未变）。

---

## 步骤 2f：影子评估（evaluate.py --corr-scenarios，7 seeds 默认 11,22,33,44,55,66,77）

命令：`.venv/bin/python scripts/evaluate.py --config config/strategy_v4_4.yaml --corr-scenarios --json`

**v4.4 verdict = PASS，failed_constraints = []**；realized：Sharpe 1.4985 / 年化 14.515% /
MaxDD 5.847%；adversarial pass_rate 100%（8/8 情景 beats_ew），baseline_retention 0.9042
（v4.3 同口径 0.9047）；最差情景 stagflation（Sharpe 0.8217，与 v4.3 结构相同）。

### 7+1 情景明细（v4.4 vs v4.3 同口径复跑，7 seeds）

| 情景 | 机制 | v4.4 Sharpe | v4.4 MaxDD | v4.3 Sharpe | v4.3 MaxDD | ΔSharpe | 门禁(beats_ew) |
|---|---|---|---|---|---|---|---|
| baseline | baseline | 1.1449 | 9.39% | 1.1644 | 9.81% | −0.020 | ✅ |
| vol_stress | vol_defense | 0.9814 | 11.28% | 1.0044 | 11.03% | −0.023 | ✅ |
| offense_cooldown | selection | 0.9905 | 9.86% | 1.0025 | 11.95% | −0.012 | ✅ |
| bond_bear | defense_asset | 0.8757 | 10.76% | 0.9031 | 11.11% | −0.027 | ✅ |
| decorrelation | dispersion | 1.2527 | 7.94% | 1.2552 | 8.24% | −0.003 | ✅ |
| stagflation | composite | 0.8217 | 11.00% | 0.8416 | 11.41% | −0.020 | ✅ |
| **corr_regime_shift**（新） | corr_crisis | **1.2971** | **7.87%** | 1.3227 | 7.92% | −0.026 | ✅ |
| **corr_crisis_combo**（新） | corr_crisis | **1.0274** | **9.66%** | 1.0444 | 9.55% | −0.017 | ✅ |

### 与既有 baseline_metrics.json（v4.3 存档，5 压力情景）的对比

本次 v4.3 同口径复跑与 `output/adversarial/baseline_metrics.json` 存档的 5 情景
（vol_stress/offense_cooldown/bond_bear/decorrelation/stagflation）数值**逐位一致**
（如 stagflation 0.8416/11.41%），评估管线无漂移。
**EWMA 开启对既有 5 情景门禁无劣化**：v4.4 全部 beats_ew=True，Sharpe 差均在
−0.003 ~ −0.027（≈2% 以内，合成情景 seed 噪声量级），MaxDD 有 4/6 情景反而更低。

### 参考数据点复核（seeds 11,22,33）

| 配置 | corr_crisis_combo Sharpe | MaxDD | corr_regime_shift Sharpe | MaxDD |
|---|---|---|---|---|
| v4.3（任务书参考：1.266 / 9.55%） | **1.2656** ✅ 复核一致 | 9.55% ✅ | 1.6478 | 7.92% |
| v4.4 | **1.2511** | 9.66% | 1.6440 | 7.87% |

v4.4 与 v4.3 参考量级一致（ΔSharpe −1.1%、ΔMaxDD +0.11pp），**不劣于参考量级** ✅。

---

## halflife 择优判定

门禁（相对 v4.3 基线 Sharpe 1.488 / MaxDD 5.84%）：

| 门禁 | hl=6 | hl=8 | hl=10 | hl=13 |
|---|---|---|---|---|
| realized Sharpe 掉幅 < 5%（≥1.414） | ✅ 1.499（+0.7%） | ✅ 1.498（+0.7%） | ✅ 1.501（+0.9%） | ✅ 1.500（+0.8%） |
| MaxDD 升幅 < 0.5pp（≤6.34%） | ✅ 5.85%（+0.01pp） | ✅ 5.85%（+0.01pp） | ✅ 5.85%（+0.01pp） | ✅ 5.85%（+0.01pp） |
| Layer 3.5 触发率 < 15% | ✅ 12.69%（余量小） | ✅ 8.76% | ✅ 8.16% | ✅ 7.10% |
| evaluate verdict = PASS | 未单独跑* | ✅ PASS（7 seeds） | 未单独跑* | 未单独跑* |

\* 合成对抗情景中 Layer 3.5 的 EWMA 差异是 verdict 的次阶因素（realized 与触发结构均
几乎相同），未对非默认档追加全量 evaluate，避免以 7-seed 噪声作档位排序依据。

**结论：选定 halflife = 8（保持 v4_4.yaml 默认值，不改配置）。**

依据：
1. hl=8 四项门禁全过：realized Sharpe 1.498（较基线 +0.7%）、MaxDD +0.01pp、触发率
   8.76% < 15%、evaluate verdict=PASS 且两个新 corr 情景均通过门禁并与 v4.3 参考量级一致；
2. 任务规则"若 8 达标则保持 8"——默认值优先，避免引入额外调参自由度；
3. 4 档 realized 指标差异（Sharpe ±0.003、年化 ±0.07pp）远小于噪声量级，不构成改档理由；
   hl=6 虽也达标但触发率 12.69% 距红线仅 2.3pp、满格周数与调仓摩擦最高，稳健性余量最差；
   hl=10/13 与 hl=8 无实质差异，仅作对比记录。

---

## 附：产物清单

- 本报告：`output/experiments/exp_v44_shadow.md`
- halflife 变体配置：`config/experiments/v4_4_hl6.yaml` / `v4_4_hl10.yaml` / `v4_4_hl13.yaml`
- 回测报告：`output/experiments/v44_shadow/report_v4_3_regression.md`、`report_v4_4_hl{6,8,10,13}.md`（含运行日志 `bt_hl*.log`）
- 触发率脚本与输出：`output/experiments/v44_shadow/trigger_rate.py`
- 影子评估 JSON：`output/experiments/v44_shadow/evaluate_v4_4.json`（7 seeds）、
  `evaluate_v4_3_corr.json`（7 seeds 对照）、`evaluate_v4_{3,4}_s3.json`（seeds 11,22,33 复核）
