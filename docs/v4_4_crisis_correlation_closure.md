# v4.4 相关性危机轴闭环 — 动机、设计与验收全记录

> **一句话**：CCC-GARCH 常相关框架在数学上**无法生成"危机中相关性真实飙过 0.6 阈值"的情景**（正定性上限内进攻对相关最高只能推到 ≈0.31），Layer 3.5 危机相关性加成因此 13 年从未被定向压测。v4.4 用 **regime-switching 相关 DGP** 补上这条压测轴、用**无状态 EWMA 加权相关**让 Layer 3.5 更快捕捉相关性飙升，并把 `corr_crisis` 升级为硬门禁——默认关闭时与 v4.3 逐位一致。

- 版本：v4.4（`config/strategy_v4_4.yaml`，EWMA halflife=8，圆整防御参数）
- 基线：v4.3（`config/strategy_v4_3.yaml`，Sharpe 1.488 / 年化 14.52% / MaxDD 5.84%）
- 证据链：`output/experiments/exp_crisis_corr.md`（缺口实证）、`exp_v44_shadow.md`（T4 影子对照）、
  `exp_v44_pipeline.md`（T5 全管线）、`exp_rounded_robust.md` + `exp_config_variants.md`（圆整参数三轴校验）
- 方法学上位文档：[`adversarial_robustness_methodology.md`](adversarial_robustness_methodology.md)

---

## 1. 动机与缺口实证

### 1.1 Layer 3.5 是什么

Layer 3.5（`engine_core.compute_crisis_boost`）：26 周窗口内进攻资产（纳指/中证500/黄金）两两相关
max|ρ| > 0.6 时，防御比例线性加成（斜率 1.875），上限 +0.15。设计目标是捕捉"危机中相关性收敛、
分散化失效"的时变风险。真实历史（650 有效周）触发率 **5.7%**（37 周，满上限仅 1 周），集中在
2016 / 2018 / 2020 / 2025-26 等风险时段——机制在真实数据上按设计意图工作。

### 1.2 缺口：CCC 常相关框架原则上压不到 Layer 3.5

对 v4.0 对抗框架的定向审查（`exp_crisis_corr.md`）确认了三层缺口：

1. **旧 5 情景从未主动压测过它**。σ/μ 旋钮（vol_stress/offense_cooldown/bond_bear/stagflation）
   不改变相关结构，5 情景中除 decorrelation 外 Layer 3.5 触发统计**逐 seed 完全相同**（中位 6.62%）；
   触发完全来自肥尾 t(ν≈4.5) 创新在 26 周窗口里的样本相关噪声（DGP 真实进攻对相关仅 ≈0.24，
   远低于 0.6 阈值），而非情景设计的危机相关飙升。
2. **c_mult 旋钮有数学天花板**。`_scale_corr` 的正定性上限 cap = 1/(1−λ_min) ≈ **1.693**
   （λ_min=0.409）；即使 c_mult=1.30 未被截断，进攻对 DGP 相关也只从 ≈0.24 推到 **≈0.31**，
   仍远低于 0.6 触发阈值——**CCC 常相关框架内无论怎么拧旋钮都无法让真实相关穿越触发区**，
   只能抬高噪声触发的频率（corr_up_severe 触发率中位也只到 8.77%）。
3. **时变触发时机原则上不可验证**。Layer 3.5 捕捉的是*时变*相关收敛，而 CCC 相关恒定，
   其入场/退出时机的有效性在该框架内无法评估。

结论：Layer 3.5 的阈值/斜率/上限参数 **13 年（662 周回测跨度）从未被定向压测**，
只是被采样噪声顺带触发。

### 1.3 红线穿破证据：旧 c_mult 情景下 MaxDD 13.78%

在旧框架内把 c_mult 拧到 1.30 并叠加 σ×1.2 + μoff×0.8（7 seeds × v4.3）：

| 情景（CCC c_mult 版） | 策略 Sharpe 中位 | 等权 | 中位 MaxDD | 最差 seed MaxDD |
|---|---|---|---|---|
| corr_up_mild (c=1.15) | 1.146 | 0.977 | 10.15% | 16.84% |
| corr_up_severe (c=1.30) | 1.083 | 0.938 | 10.22% | 18.46% |
| **corr_crisis_combo (c=1.30 + σ1.2 + μoff0.8)** | 0.757 | 0.659 | **13.78%** | **20.17%** |

corr_crisis_combo 的**中位** MaxDD 13.78% 已穿破 12% 红线（既有 5 情景基线 worst 仅 11.95%），
最差 seed 达 20.17%。

> ⚠️ **口径注意**：上表的 `corr_crisis_combo` 是缺口实证阶段的 **CCC c_mult 缩放版**（相关只被
> 整体拉伸、依然到不了 0.6 阈值），与 v4.4 正式入禁的 **regime DGP 版** `corr_crisis_combo`
> （两状态 Markov、危机态进攻对相关真实抬到 0.85）**同名不同物**。前者是"半吊子相关 + 波动放大"
> 的组合效应（见 §6），后者才是对 Layer 3.5 的定向压测。

---

## 2. 方案选型

### 2.1 压测 DGP：regime-switching 而非 DCC

补相关性压测轴的两个候选是 DCC-GARCH 与 regime-switching 相关。选 **regime-switching**：

- **压测需要的是可控性，不是拟真度**。目标是定向验证"相关真实飙过 0.6 阈值时 Layer 3.5
  的行为"，regime DGP 直接把危机态相关设为 `rho_crisis`（可精确控制穿越幅度与停留时长），
  情景语义清晰、结果可归因；DCC 的相关路径由拟合出的 a/b 动态参数内生决定，
  危机相关的峰值与持续期都不可直接指定，作为门禁情景不可控。
- **DCC 引入额外拟合风险**。DCC 需在 13 年周频数据上估计相关动态参数，样本量对 5 资产
  相关动态而言偏薄，拟合不稳会把噪声写进门禁；regime DGP 只在既有 CCC 拟合产物（R、GARCH
  参数、t 尾）之上加一个两状态开关，改动面最小、可复现性最强。
- regime-switching 恰好也是 methodology 文档第 8 节早已声明的"CCC 覆盖不到"的方向之一，
  本次是把已知边界补成显式压测轴。

### 2.2 引擎侧：无状态 EWMA，而非有状态机制/阈值下调/max_def 联动

引擎侧候选方案与否决理由：

| 候选 | 否决理由 |
|---|---|
| **有状态机制**（危机计数器/滞回） | 引入路径依赖，破坏"逐周独立可复算"性质：`--verify` 回放、离线触发率统计、单测 pin 全部要携带状态；出错面大、审计成本高 |
| **阈值下调**（0.6 → 更低） | 平时噪声触发大增（真实历史 max\|ρ\| 中位 0.358、p95 0.603，阈值贴着 p95 设计），改变 realized 全期行为，等于重调一个 Morris μ*≈0 的参数 |
| **max_def 联动**（相关高时抬防御天花板） | 跨层耦合 Layer 3 与 3.5，扩大参数曲面且与 crisis boost 现有"可突破 max_def"语义重叠 |
| **EWMA 加权相关（采纳）** | **无状态**：同窗口、同阈值、同斜率，仅把等权 Pearson 换成半衰期加权——对相关性飙升的响应更快（新近周权重高），而机制骨架与可验证性不变；halflife→∞ 数学上收敛回等权 Pearson |

### 2.3 防过拟合：关键常数固定，不进搜索空间

regime DGP 的 `p_enter=0.03`（平常→危机/周）、`p_stay=0.92`（危机停留，均期 ~12 周）、
`rho_crisis=0.85` 为**固定常数，不进任何搜索空间**：压测情景是门禁不是目标函数，若让优化器
接触这些旋钮，等于允许策略参数对特定危机形态定向调参——重蹈"在自己出的考题上刷分"覆辙。
幅度的稳健性改由 OOS 通道用**训练未见的 rho_crisis ∈ {0.75, 0.90} 变体**独立检验（§4.4）。
同理，halflife 经步骤 3 敏感度证明非主控后**锁定 8、不进入优化循环**（§4.1）。

---

## 3. 实现摘要

### 3.1 分发器模式（默认关 = v4.3 逐位复现）

`engine_core.compute_crisis_boost` 改为分发器：`crisis_corr_ewma_enabled=False`（默认）走
`_compute_crisis_boost_classic`（v4.3 原逻辑原样搬移），`True` 走 `_compute_crisis_boost_ewma`。
配置缺失 `crisis_correlation_ewma` 段时默认关——v4.3/v4.2/v4.1 全部 yaml 在新代码上
**逐位复现**（T4 回归 + T5 残差追溯 661/662 周逐位一致佐证）。

### 3.2 EWMA 加权相关公式

与 classic 同窗口（26 周）、同早退条件，仅相关估计器不同。窗口内第 t 周（t=0 最老）权重：

```
w_t = 0.5 ** ((window − 1 − t) / halflife)      # 最新周权重 1，向过去指数衰减
```

对每个进攻资产对，在 NaN mask 上重归一化权重后计算加权相关：

```
x̄ = Σ w·x,  ȳ = Σ w·y
ρ = Σ w·(x−x̄)(y−ȳ) / (√(Σ w·(x−x̄)² · Σ w·(y−ȳ)²) + 1e-12)
```

数学自检：halflife → ∞ 时权重收敛为等权，加权相关收敛为 Pearson 相关，即该函数收敛到
classic 路径（单测 `TestEwmaWeightedCorrMath` 固化此性质）。触发规则不变：
max|ρ| > 0.6 → boost = min((max|ρ| − 0.6) × 1.875, 0.15)。

### 3.3 regime DGP 机理（`adversarial_robustness.gen_regime_corr`）

两状态 Markov 相关切换的 CCC-GARCH 变体：

- 平常态用拟合 R；危机态把进攻对两两相关抬到 `rho_crisis=0.85`，经 PSD 修复 + 相关阵
  归一化后取 Cholesky，两态各持一个因子；
- Markov 链：`p_enter=0.03`、`p_stay=0.92`（危机均期 ~12 周），逐周切换所用 Cholesky；
- `c_mult` 在此 DGP 中被忽略（相关由状态机控制）；其余 4 轴旋钮
  （rho_mult/sig_mult/mudef_mult/muoff_mult）行为与 `gen_garch` 完全一致，
  故 `corr_crisis_combo` 可叠加 σ×1.2 + μoff×0.8 构成复合情景。

### 3.4 门禁扩展

- 新情景独立注册于 `CORR_STRESS_SCENARIOS`：`corr_regime_shift`（纯相关切换）、
  `corr_crisis_combo`（相关切换 + σ1.2 + μoff0.8），机制标签 `corr_crisis`；
- `evaluate.py` 的 `MECHANISM_GATE` 将 **`corr_crisis` 设为硬门禁**（与 vol_defense/
  defense_asset/dispersion/composite 同级，必须过）；经 `--corr-scenarios`
  （API：`include_corr_scenarios=True`）显式并入评估，**默认 False 时输出与 v4.3 逐位一致**
  （单测 `TestRobustnessScoreBackwardCompat` 固化）；
- `oos_validation.py`（+22/−1 行，v4.4 唯一被修改的既有脚本）：新增 `OOS_CORR_VARIANTS`
  （rho_crisis 0.75/0.90，训练只见 0.85）与 `--corr-variants` 开关，追加独立
  `A2_regime_corr_variants` 段，**默认不跑、不参与三通道 core 判定**；`run_channel_ab` 内
  单点 gen 分发（`params["dgp"]=="regime_corr"` → `gen_regime_corr`）。编辑前后
  预/后逐位 A/B 与历史产物重跑双重验证无漂移。

### 3.5 配置段与测试

`config/strategy_v4_4.yaml` 相对 v4.3 的全部差异：

```yaml
defense:                     # 四参数圆整(v4.3: 0.3492/0.0764/0.384/0.8299)
  def_alloc: 0.35
  step_low: 0.075
  step_high: 0.38
  max_def: 0.83
crisis_correlation_ewma:     # v4.4 新增段
  ewma_enabled: true
  ewma_halflife: 8
```

`tests/test_v44_crisis_corr.py` 覆盖六类：classic 路径逐位不变 + pin 值、EWMA 激活行为、
EWMA 加权相关数学性质（halflife→∞ 收敛）、regime DGP 统计性质、`robustness_score`
向后兼容（默认关时逐位一致）、v4.3/v4.4 配置加载。

---

## 4. 验收结果（全部为实测数字）

### 4.1 halflife 4 档对比与锁定 8 的依据

真实历史 662 周回测 + 触发率统计（T4）：

| 档位 | Sharpe | 年化 | MaxDD | Layer3.5 触发率 | 满格(0.15)周数 | 调仓次数 |
|---|---|---|---|---|---|---|
| v4.3 classic（对照） | 1.488 | 14.52% | 5.84% | 5.59% | 1 | 377 |
| hl=6 | 1.499 | 14.47% | 5.85% | 12.69% | 23 | 387 |
| **hl=8（锁定）** | **1.498** | **14.51%** | **5.85%** | **8.76%** | 17 | 378 |
| hl=10 | 1.501 | 14.54% | 5.85% | 8.16% | 12 | 375 |
| hl=13 | 1.500 | 14.54% | 5.85% | 7.10% | 12 | 373 |

锁定 hl=8 的依据：
1. 四项门禁全过（realized Sharpe +0.7% / MaxDD +0.01pp / 触发率 8.76% < 15% 红线 /
   evaluate verdict=PASS）；
2. 对抗轴敏感度（T5 步骤 3，hl∈{6,8,10,13} × 3 seeds 全量 evaluate）：4 档 worst_maxdd
   极差仅 **0.1027pp < 1pp** → **halflife 非主控参数**，按"默认值优先"规则保持 8，
   不进入优化循环；
3. 4 档 realized 指标差异（Sharpe ±0.003）远小于噪声量级，不构成改档理由；hl=6 触发率
   距 15% 红线仅 2.3pp 且满格/摩擦最高，稳健余量最差。

realized 对 halflife 不敏感恰说明 EWMA 的价值在危机情景而非历史路径——不应据 realized
微小差异调参。

### 4.2 v4.4 realized 与 7-seed 评估（verdict=PASS）

- **realized**：Sharpe **1.498** / 年化 **14.51%** / MaxDD **5.85%**（662 周，回撤谷底
  2024-09-13 与 v4.3 相同，回撤事件结构未变）。
- **`evaluate.py --corr-scenarios`（7 seeds 11-77）**：verdict = **PASS**，
  failed_constraints = []；adversarial pass_rate 100%（8/8 情景 beats_ew），
  baseline_retention 0.9042（v4.3 同口径 0.9047）；全情景 worst MaxDD
  **11.28%（stagflation）≤ 12%**——最紧约束仍是既有情景，非 v4.4 新轴。
- **新 corr_crisis 硬门禁**：corr_regime_shift Sharpe **1.297** / MaxDD **7.87%**；
  corr_crisis_combo **1.027** / **9.66%**（均为 7-seed 中位，均胜等权）。
- 既有 5 情景无劣化：v4.4 vs v4.3 ΔSharpe 在 −0.003 ~ −0.027（seed 噪声量级），
  MaxDD 4/6 情景反而更低；v4.3 侧复跑与 `baseline_metrics.json` 存档逐位一致（管线无漂移）。

### 4.3 圆整参数三轴校验

v4.4 沿用的圆整防御参数（0.35/0.075/0.38/0.83）此前已通过三个数据维度的无损验证：

- **realized**：Sharpe 1.489 vs 基线 1.488，MaxDD 5.85% vs 5.84%，MaxDD 事件相同；
- **对抗轴**：evaluate 总判定 PASS，5/5 情景胜等权，worst MaxDD 11.42% < 12%；
- **bootstrap 轴**（200 路径逐 seed 配对）：胜率 95.5% / α P10 +0.077（基线 96.0% / +0.078），
  配对 ΔSharpe 均值 +0.0008、P10~P90 = [−0.0021, +0.0036]——扰动远小于数据轴方差，
  统计上不可区分。粗圆整（≤5% 扰动）同样无损 → 参数邻域为平坦高原，四位小数精度无实质意义，
  圆整降低过拟合表面积且提升配置可读性。

T5 步骤 4 据此 + halflife 非主控，正式判定**跳过 Stage A/B/C 重优化**：新增机制在既有
参数上无任何约束逼近边界（最紧的 stagflation 11.28% 距上限 0.72pp），重优化无收益来源。

### 4.4 三通道 OOS：TRUE_ROBUST + 未见幅度变体

`oos_validation.py`（baseline=v4.3, candidate=v4.4）：

| 通道 | v4.3 | v4.4 | core 判定 |
|---|---|---|---|
| A held-out 幅度（10 情景） | 80% / 14.687% / +0.086 | 80% / 14.693% / +0.082 | PASS |
| B 独立 seeds | 100% / 10.153% / +0.150 | 100% / 10.142% / +0.149 | PASS |
| C block bootstrap（30 路径） | 93.3% / 15.128% / +0.236 | 93.3% / 15.045% / +0.238 | PASS |

（各格为 pass_rate / worst_DD / avg_margin）结论字段 **TRUE_ROBUST**，三通道 core 全 PASS。

**A2 rho_crisis 未见幅度变体**（7 seeds，训练只见 0.85）：corr_shift_075 Sharpe 1.2165 /
MaxDD 8.6882%，corr_shift_090 Sharpe 1.3076 / 7.8784%——**worst 8.69% ≤ 12%**，
Layer 3.5 的防御不依赖被压测过的特定相关幅度。

### 4.5 联合鲁棒性与测试套件

`robustness_joint.py` v4.4 全套（base Sharpe 1.4985）：

| 测试 | 结果 | 判定 |
|---|---|---|
| Test1 参数轴（8 参数 45 回测） | 最差 top_n：Sharpe −11.25% / MaxDD +1.36pp，无断崖 | PASS |
| Test2 数据轴（200 bootstrap） | 胜率 **96.5%**，α-Sharpe P10 **+0.086** | PASS |
| Test3 联合 LHS（200 样本） | 方差比 **0.793** ≤ 1.30（无薄峰），胜率 96.0% | PASS |

各项相对 v4.3 历史基线微好（如 Test2 sharpe_p10 0.892 vs 0.886）。
测试套件：`pytest tests/ -q` → **171 passed，全绿**（含 v4.4 新增 `test_v44_crisis_corr.py`）。

---

## 5. 一致性终检与遗留问题

### 5.1 `--verify` ΔSharpe = 0.010016 的裁决全过程

如实记录：T5 步骤 7 精确复刻 `--verify`，engine Sharpe 1.4984554869975593 vs 回放
1.5084710339704044，**ΔSharpe = 0.010016**——超出原任务门禁 <0.01（脚本自带生产门禁 <0.02，
显示通过）。按硬规则停止并逐级上报，经两轮裁决：

1. **无费口径验证**：engine 关闭扣费重跑，Δ_nofee = 0.006562（Δ_fee = 0.010016）→
   fee 仅解释约 35%，"回放不扣费"单因子假设不成立；但 **v4.4−v4.3 增量仅 +0.000113**
   （v4.3 同口径 Δ_fee = 0.009902）→ 差异非 v4.4 引入。
2. **逐周残差定位**：按收益归属周对齐 engine `weekly_records` 与回放序列——662 周中
   **仅首记录周 2013-08-30 权重分歧**（回放无 prev 状态从纯防御 0.5/0.5 起步，engine 持
   完整进攻仓），其余 **661/662 周权重逐位一致**，一致周无费收益差 max 9.107e-17
   （纯浮点噪声）→ **决策函数在两条路径产出完全相同的 alloc，engine 侧无真实不一致**；
   v4.3 同口径对照呈完全相同模式（同样仅首周分歧，max 8.934e-17）→ 口径差两版本机制相同。
3. **残差分解**：ΔSharpe 0.010016 ≈ 回放不扣费差（约 0.0035）+ 首周初始化口径差
   （2013-08-30 单周收益差 0.72%）。

**最终判定口径（leader 裁决 2）改为双条件**：生产门禁 ΔSharpe < 0.02（实际 0.010016 ✅）
**且** 版本增量 v4.4−v4.3 ≤ 0.001（实际 +0.000113 ✅）→ 步骤 7 放行。

### 5.2 遗留问题（既有缺陷，不在 v4.4 范围，建议独立修复）

1. **`--verify` 回放不扣费**：`rebalance_live.py` 回放路径 `nav *= (1+wr)` 无 fee 项
   （engine 为 `nav *= (1+wret−fee_cost)`），贡献约 0.0035 的系统性 ΔSharpe；
   建议回放按 `turnover × fee_rate` 同口径扣费。
2. **回放首周初始化口径差**：回放首记录周以纯防御起步、与 engine 首周完整分配不一致，
   贡献单周 0.72% 收益差；建议回放首周复用 engine 同款初始分配逻辑
   （或从 `_START_IDX` 前一周预热 prev 状态）。

---

## 6. 有趣发现：真实相关飙升情景下策略反而结构性占优

一个反直觉结果：缺口实证阶段（CCC c_mult 版）corr_crisis_combo 中位 MaxDD 13.78% 穿破红线，
而换成**更狠**的 regime DGP（进攻对相关真实抬到 0.85）后，v4.3/v4.4 反而**轻松通过门禁**
（corr_regime_shift MaxDD 7.87%、corr_crisis_combo 9.66%，甚至低于 baseline 情景的 9.39%）。

机理拆解：

- **相关性真实飙升重创的是等权基准**（等权 MaxDD 被推到 26-33%）：等权无防御机制，
  三条进攻腿同涨同跌时分散化完全失效，只能硬吃回撤；
- **策略侧 vol 防御更早触发**：危机态相关收敛必然伴随组合波动放大，Layer 3（纳指
  tapered-vol 三档）在相关飙升传导到回撤之前就已把仓位推向防御，Layer 3.5 EWMA 再叠加
  快速加成——防御链条对"真危机"的响应是结构性的；
- 反观旧 CCC c_mult 版的 13.78% 穿破，实为"**半吊子相关 + 波动放大**"的组合效应：
  c_mult=1.3 把相关抬到 ≈0.31——不足以触发 Layer 3.5（0.6 阈值），却足以让分散化收益
  持续劣化；叠加 σ×1.2 后策略既吃了波动亏损、又拿不到危机加成，落在防御机制的盲区。

这个发现本身就是压测通道价值的证明：**最危险的不是相关性飙到 0.85 的显性危机
（防御链条接得住），而是相关性停在 0.3-0.5 灰色地带的隐性劣化**（触发不了任何危机机制）。
regime DGP 补上了显性危机轴的验证；灰色地带的暴露此前已由旧 corr_up 系列情景刻画，
两者共同构成相关性轴的完整压测覆盖。

---

## 附：产物索引

| 类别 | 路径 |
|---|---|
| 生产配置 | `config/strategy_v4_4.yaml`（halflife 变体：`config/experiments/v4_4_hl{6,10,13}.yaml`） |
| 引擎实现 | `src/engine_core.py`（分发器 + `_compute_crisis_boost_ewma`）、`src/strategy.py`（`crisis_correlation_ewma` 配置段） |
| 压测/门禁 | `scripts/adversarial_robustness.py`（`gen_regime_corr`/`CORR_STRESS_SCENARIOS`）、`scripts/evaluate.py`（`--corr-scenarios`、corr_crisis 硬门禁）、`scripts/oos_validation.py`（`--corr-variants`） |
| 测试 | `tests/test_v44_crisis_corr.py`（套件合计 171 passed） |
| 实验/验收报告 | `output/experiments/exp_crisis_corr.md`、`exp_v44_shadow.md`、`exp_v44_pipeline.md`、`exp_rounded_robust.md`、`exp_config_variants.md` |
| 全链路产物 | `output/experiments/v44_shadow/`、`output/experiments/v44_pipeline/` |
