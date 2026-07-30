# exp_v44_pipeline — v4.4 校验管线（T5：步骤 3–7）总报告

- 日期：2026-07-30 ｜ 执行：Taylor2（任务 #11）
- 候选配置：`config/strategy_v4_4.yaml`（EWMA Layer3.5，halflife=8）｜ 基线：`config/strategy_v4_3.yaml`
- 产物目录：`output/experiments/v44_pipeline/`（驱动脚本 + JSON + 日志全套）
- **最终结论：v4.4 通过校验管线放行**。步骤 3–6 全部门禁按原任务口径 PASS；步骤 7 的 `--verify` ΔSharpe=0.010016 超原任务门禁 <0.01，经 leader 两轮裁决完成无费口径验证与逐周残差定位后，**判定口径正式改为"生产门禁 <0.02 + 版本增量 ≤0.001"双条件**，两条件均满足，放行（详见 §步骤7，残差已定位为回放实现口径差，engine 决策函数无真实不一致）。

---

## 步骤 3｜halflife 对抗轴敏感度（Morris 轻量化替代）

方法：外部驱动 `step3_hl_sensitivity.py`，对 halflife∈{6,8,10,13} 各跑 `evaluate_full`（seeds=11/22/33，`include_corr_scenarios=True`，D_max=12%）。hl6/10/13 使用 T4 预置的 `config/experiments/v4_4_hl{6,10,13}.yaml`（与 v4_4 仅 halflife 一字之差），未修改任何既有文件。

| halflife | verdict | worst_maxdd（情景） | corr_crisis 硬门禁 worst Sharpe / DD | corr_regime_shift Sh/DD | realized Sharpe |
|---|---|---|---|---|---|
| 6 | PASS | 9.6949%（stagflation） | 1.2461 / 9.6949% | 1.6393 / 7.8715% | 1.4986 |
| **8（锁定）** | PASS | **9.6633%（stagflation）** | 1.2511 / 9.6633% | 1.6440 / 7.8715% | 1.4985 |
| 10 | PASS | 9.6243%（stagflation） | 1.2532 / 9.6243% | 1.6491 / 7.8715% | 1.5007 |
| 13 | PASS | 9.5923%（stagflation） | 1.2619 / 9.5923% | 1.6529 / 7.8715% | 1.4995 |

- 4 档 worst_maxdd 极差 **0.1027pp < 1pp**（corr 门禁 DD 极差同为 0.1027pp，Sharpe 极差 0.016）→ **判定：halflife 非主控参数，锁定 8**，不进入优化循环。
- 一致性交叉验证：hl=8 的 worst_maxdd=0.09663339709965532 与 T4 `output/experiments/v44_shadow/evaluate_v4_4_s3.json`（同 3 seeds）**逐位一致**。
- 产物：`step3_hl_sensitivity.{py,json}`、`step3_run.log`。门禁：**PASS**。

## 步骤 4｜跳过 Stage A/B/C 重优化的正式判定

依据（引用 T4 `output/experiments/v44_shadow/evaluate_v4_4.json`，7 seeds 全量评估）：

1. v4.4 在圆整基线参数（def_alloc=0.35 / step_low=0.075 / step_high=0.38 / max_def=0.83）上 `evaluate_full` verdict=**PASS**，failed_constraints=[]；
2. realized：年化 14.5150% / MaxDD 5.8472% / Sharpe 1.4985；adversarial worst_maxdd 11.2818%（stagflation）≤ 12%；
3. 6 项分机制硬门禁 pass_rate 全 1.0，其中新增 corr_crisis 硬门禁 worst Sharpe 1.0274 / DD 9.6633%；corr_regime_shift 1.2971 / 7.8715%；corr_crisis_combo 1.0274 / 9.6633%；
4. 步骤 3 证明唯一新增自由参数 halflife 非主控（极差 0.1027pp）。

新增机制在既有参数上无任何约束逼近边界（最紧的 stagflation 11.28% 距 12% 上限尚有 0.72pp，且系既有情景、非 v4.4 新轴），重优化无收益来源。**判定：跳过重优化，v4.4 沿用 v4.3 圆整参数基线。**

## 步骤 5｜三通道 OOS（v4_3 vs v4_4）+ regime_corr 幅度变体

### 5b. `scripts/oos_validation.py` 最小扩展（唯一被修改的既有文件，+22/−1 行）

diff 要点（`git diff scripts/oos_validation.py`）：
- 新增 `OOS_CORR_VARIANTS = {corr_shift_075: rho_crisis=0.75, corr_shift_090: rho_crisis=0.90}`（训练只见 0.85，两档均为未见幅度）；
- `run_channel_ab` 内单点 gen 分发：`params.get("dgp")=="regime_corr"` 时用 `adv.gen_regime_corr`，否则保持 `adv.gen_garch`（与 `adversarial_robustness._eval_strat_ew` 同口径）；
- 新增 `--corr-variants` 开关，追加独立 `A2_regime_corr_variants` 段，**默认不跑、不参与三通道 core 判定**，既有 10 情景与判定行为不变。

### 无漂移验证（双重证据）

- **预/后编辑逐位 A/B**（`step5_nodrift_ab.{py,json}`）：HEAD 版副本 vs 编辑后版本，同 cfg/seed 跑 vol_small / def_deep_bear / quad_shock，`identical=True`（如 vol_small strat_sharpe 1.3577639440 前后逐位相同）。
- **历史产物重跑**（`oos_v42_vs_v43_redo.json`）：v4_2 侧三通道与历史 `output/adversarial/oos_v42_vs_v43.json` **逐位复现**；v4_3 侧差异经 git 溯源确认为历史产物 mtime（07-28 20:37）早于 v4_3 配置定稿 commit `6029a13`（07-28 22:24），属历史产物基于 WIP 配置，与本次代码编辑无关。

### 5a+5c. 主运行结果（`oos_v43_vs_v44.json`，JSON 内 v4_1/v4_2 标签分别对应 baseline=v4_3 / candidate=v4_4）

| 通道 | v4_3 (pass_rate / worst_DD / avg_margin) | v4_4 | core 判定 |
|---|---|---|---|
| A held-out 幅度（10 情景，不减） | 80% / 14.687% / +0.086 | 80% / 14.693% / +0.082 | **PASS**（envelope OUT，既有备注：极端 OOS 幅度天然突破策略族上限） |
| B 独立 seeds | 100% / 10.153% / +0.150 | 100% / 10.142% / +0.149 | **PASS**（envelope IN） |
| C block bootstrap（30 路径） | 93.3% / 15.128% / +0.236 | 93.3% / 15.045% / +0.238 | **PASS**（envelope OUT，同上备注） |

结论字段：**TRUE_ROBUST**，三通道 core 全 PASS（pass_rate / worst_DD / avg_margin 相对不劣化 + margin 为正）。

**A2 regime_corr 变体**（7 seeds，独立段）：

| 变体 | v4_4 Sharpe / MaxDD | v4_3 worst_maxdd 参照 |
|---|---|---|
| corr_shift_075 | 1.2165 / **8.6882%** | — |
| corr_shift_090 | 1.3076 / 7.8784% | — |
| worst | **8.6882% ≤ 12%** ✅ | 8.7237% |

步骤 5 门禁（core 全 PASS + 新变体 worst_maxdd≤12%）：**PASS**。产物：`oos_v43_vs_v44.json`、`oos_v44_run.log`。

## 步骤 6｜联合鲁棒性 robustness_joint（v4_4 全套）

`robustness_joint_v44.json`（复制自 `output/robustness/robustness_joint_all_20260730_181317.json`），base Sharpe 1.4985：

| 测试 | 任务门禁 | 结果 | 判定 |
|---|---|---|---|
| Test1 参数轴（8 参数 45 回测） | 每参数掉幅≤20% 且 MaxDD≤+3pp、无断崖 | 最差 top_n：Sharpe −11.25% / MaxDD +1.36pp，8 参数 pass_all 全 true，cliff 均 false | **PASS** |
| Test2 数据轴（200 bootstrap） | 相对判据：胜率≥90% 且 α-Sharpe P10>0 | 胜率 **96.5%**，α P10 **+0.086**；（绝对判据 pass_absolute=false 属 data-inherent，v4.3 历史同样 false，非 v4.4 引入） | **PASS** |
| Test3 联合 LHS（200 样本） | 方差比≤1.30（薄峰检测） | 方差比 **0.793**（drop 比 0.762），胜率 96.0%，α P10 +0.052 | **PASS** |

与 v4.3 历史对照各项微好（如 Test2 sharpe_p10 0.892 vs 0.886）。步骤 6 门禁：**PASS**。

## 步骤 7｜实盘一致性终检

### 7.1 测试套件
`pytest tests/ -q` → **171 passed**（0 failed，110.7s）。✅

### 7.2 `--verify` ΔSharpe（如实记录）

- 精确复刻驱动 `step7_verify_exact.{py,json}`：engine 1.4984554869975593 vs 回放 1.5084710339704044，**ΔSharpe = 0.010016**。
- **该值超出原任务门禁 <0.01**（脚本自带生产门禁为 <0.02，显示"✅ 通过"）。按硬规则停止并逐级上报，leader 两轮裁决过程与证据如下。

**(c1) 无费口径验证**（`step7c_feefree.{py,json}`）：engine 关闭扣费重跑，v4_4 Δ_nofee=0.006562（Δ_fee=0.010016）→ **fee 仅解释约 35%**，fee 单因子假设不成立；但 v4.4−v4.3 增量仅 **+0.000113 ≤ 0.001**（v4_3 Δ_fee=0.009902），差异非 v4.4 引入。

**(c2) 逐周残差定位**（`step7_residual_trace.{py,json}`，leader 指定证据文件）：外部驱动按"收益归属周"对齐 engine `weekly_records` 与 --verify 回放的 alloc/收益序列：
- 662 个对齐周中**仅 1 周权重分歧**：首记录周 2013-08-30（回放起步周），engine 持进攻仓（纳指 0.40 / 中证500 0.2155 / 红利低波 0.1923 / 国债 0.1923）vs 回放纯防御（红利低波 0.5 / 国债 0.5）——系回放实现的**首周初始化口径差**（回放首周无 prev 状态从防御起步）；
- 其余 **661/662 周权重逐位一致**，一致周无费收益差 max 9.107e-17（纯浮点噪声）→ **决策函数在两条路径产出完全相同的 alloc，engine 侧无真实不一致**；
- v4_3 同口径对照（`step7_residual_trace_v43.json`）：完全相同模式（同样仅 2013-08-30 分歧 0.4，一致周 max 8.934e-17）→ **口径差在 v4_3 与 v4_4 机制相同**。

**残差分解**：ΔSharpe 0.010016 = 回放不扣费差（约 0.0035）+ 首周初始化口径差（2013-08-30 单周收益差 0.72%）。

### 7.3 判定（按 leader 裁决 2）

残差定位为回放实现口径差且两版本机制相同 → 判定口径正式改为双条件：

| 条件 | 阈值 | 实际 | 判定 |
|---|---|---|---|
| 生产门禁 ΔSharpe | < 0.02 | 0.010016 | ✅ |
| 版本增量（v4.4−v4.3） | ≤ 0.001 | +0.000113 | ✅ |

步骤 7 按双条件口径：**PASS（放行）**。

---

## v4.4 验收总标准逐条勾验

| # | 标准 | 证据 | 判定 |
|---|---|---|---|
| 1 | 新相关性危机情景 MaxDD ≤ 12% | corr_crisis 硬门禁 worst DD 9.6633%、corr_regime_shift 7.8715%、combo 9.6633%（7 seeds）；OOS A2 未见幅度变体 worst 8.6882% | ✅ |
| 2 | realized / 既有 5 情景门禁 / Test1-2-3 / 三通道 OOS 相对 v4.3 全不劣化 | evaluate PASS（worst 11.2818%≤12%）；三通道 core 全 PASS（TRUE_ROBUST）；Test1/2/3 全过且各项微好于 v4.3 | ✅ |
| 3 | 既有测试全绿；`--verify` Sharpe 差 < 0.01 | **171 passed 全绿** ✅；ΔSharpe=0.010016 **超原门禁**，经残差定位（回放口径差、engine 无不一致、增量 +0.000113）按 leader 裁决改为"**<0.02 且增量 ≤0.001**"双条件放行 | ✅（判定口径已变更，如实记录） |
| 4 | v4.3 配置在新代码上逐位复现（新逻辑默认关） | T4 回归基线已验证（7-seed 逐位）；本管线佐证：hl=8 与 T4 s3 JSON 逐位一致、v4_3 残差追溯 661/662 周逐位一致 | ✅（T4 主证） |

## 遗留问题（均不在 v4.4 范围，建议独立修复）

1. **`--verify` 回放不扣费**：`rebalance_live.py` 回放路径 `nav *= (1+wr)` 无 fee 项（engine 为 `nav *= (1+wret-fee_cost)`），系既有缺陷，贡献约 0.0035 的系统性 ΔSharpe；建议在回放中按 `turnover * fee_rate` 同口径扣费。
2. **回放首周初始化口径差**：回放首记录周（2013-08-30）以纯防御起步，与 engine 首周完整分配不一致，贡献单周 0.72% 收益差；建议回放首周复用 engine 同款初始分配逻辑（或从 `_START_IDX` 前一周预热 prev 状态）。

## 产物清单

`output/experiments/v44_pipeline/`：`step3_hl_sensitivity.{py,json}` + `step3_run.log`｜`step5_nodrift_ab.{py,json}`、`oos_v42_vs_v43_redo.json` + `oos_nodrift_run.log`｜`oos_v43_vs_v44.json` + `oos_v44_run.log`｜`robustness_joint_v44.json` + `joint_run.log`｜`step7_verify_exact.{py,json}`、`step7c_feefree.{py,json}`、`step7_residual_trace.{py,json}`、`_trace_v43.py` + `step7_residual_trace_v43.json`。
既有生产 JSON（output/adversarial/、output/robustness/）未改动；被修改的既有文件仅 `scripts/oos_validation.py`（+22/−1）。
