# 虾池ETF轮动策略 v4.6 — 定向 boost 分级应用生产版

基于 **5只ETF** 的周频动量轮动策略，全连续/零门控/四层架构（含 DefAlloc）。
**Sharpe 1.609 / 年化 15.46% / 最大回撤 5.76% / Calmar 2.68**（realized 2013-08 ~ 2026-08）；
**在 v4.5-pvd 基座上新增定向 boost：Layer 3.5 危机相关性保护分级应用——显性危机
（EWMA corr>0.60）满额防御 boost，灰区（≤0.60）定向降进攻不推高防御绝对水平，
修复持续中相关灰区缺口（合成对抗 grey_corr_combo 中位 MaxDD 12.79%→11.80%，回到 12% 红线内）**。

v4.6 由 **E0→E1→E2→Gate→E3 研究管线 + 完整对抗验证管线** 产出（定向 boost 预研
7 变体矩阵中唯一全门禁 PASS 的 T1+V3 混合分级；PE 估值防御调制 E2 通过但 E3 对抗
门禁失败已裁出；R² 动量替换 E2 未过门禁）。完整验证链：evaluate --corr-scenarios
verdict PASS（worst 11.80%≤12%，机制门禁全过）+ block bootstrap 200 路径胜率 96.0%
（alpha P10 +0.091）+ OOS B 通道 PASS + 联合鲁棒性 t1/t3 PASS + 量价联合 DGP 复核
无结构性损害 + --verify Δ=0.0066。详见
[`docs/v4_6_directed_boost_closure.md`](docs/v4_6_directed_boost_closure.md) 与
[`docs/adversarial_robustness_methodology.md`](docs/adversarial_robustness_methodology.md)。

**生产状态**：`config/strategy_v4_6.yaml` 为**默认生产 config**（`rebalance_live.py` /
`run_backtest.py` / `evaluate.py` 等默认路径已全部切换，实盘脚本经 `--verify` 确认与回测
引擎一致 Δ=0.0066）。前代 `config/strategy_v4_5_pvd.yaml`（PVD 条件激活）保留为
**前代已验证配置**与影子对照（周度调仓并行输出，4-8 周复盘）；如需切回：
`python scripts/rebalance_live.py --config config/strategy_v4_5_pvd.yaml`。

## 版本演进（一）：v4.1（历史基线）→ v4.2（前代生产）

针对"realized 高 Sharpe 只代表历史这一条路径通过"的结构性风险，v4.0 框架把
"另一条路径下也不崩"变成可优化、可门禁的量化指标。**v4.1 → v4.2 是这个框架
第一次端到端跑通的产物**：

| 维度 | v4.1（历史基线） | v4.2（当前生产） |
|---|---|---|
| realized 年化 | 17.05% | 15.84%（-1.21pp，换鲁棒代价） |
| realized MaxDD | 6.97% | 6.75%（+0.22pp） |
| realized Sharpe | 1.610 | **1.635**（反涨；波动降幅 > 收益降幅） |
| 全情景对抗 worst_DD | 12.19% ✗ | **11.60% ✓**（12% 门槛） |
| 硬机制 Sharpe 门禁 | 2/4 FAIL | **4/4 PASS** |
| Adv verdict | FAIL | **PASS** |
| OOS 通道 A 相对不劣化 | 基线 | PASS（pass_rate 50%→70%） |
| OOS 通道 B 相对不劣化 | 基线 | PASS（DD 12.09%→11.63%） |
| OOS 通道 C 相对不劣化（最独立） | 基线 | **PASS（DD 18.89%→13.97%，-4.92pp）** |
| 过拟合判定 | — | **TRUE_ROBUST**（三通道 core PASS） |

**参数变化**（**"轻&快防御"胜过"重&深防御"**，节点 2+3 的涌现结果，颠覆直觉）：

| 参数 | v4.1 | v4.2 | 方向 |
|---|---|---|---|
| `def_alloc` | 0.25 | 0.145 | 基础防御更低 |
| `step_low` | 0.15 | 0.095 | 触发更早 |
| `step_high` | 0.35 | 0.193 | 档间距更紧 |
| `max_def` | 0.95 | 0.811 | 峰值防御更低 |
| `vol_window` | 11 | 10 | vol 信号更快 |

**版本沿革**：`config/strategy_v4_2.yaml`（rolling）曾是生产配置，现已被后续版本取代、降为
**前代已验证配置**（对抗 OOS 对照 + 回归 pin 保留）；`config/strategy_v4_1.yaml` 仍作**历史基线**
（对抗 OOS 验证的对照组、回归测试的历史行为参照）。当前默认生产 = v4.5-pvd（见顶部）。

## 版本演进（二）：v4.2 → v4.3（前代生产，当前回归基线）

rolling `vol_window` 有一个**设计层面的固有缺陷**：当最老一周滚出窗口时波动率会阶跃跳变。
v4.3 用 **tapered vol**（窗口内最老若干周线性降权）替代硬截断窗口，实测把纳指 vol 的
周环比跳变**均值降 27%、p95 降 42%**（vol 绝对水平仅 +1.9%）。

v4.3 经 **7 维 taper 搜索（含 `vol_taper_window`/`vol_taper_len`）+ max-Sharpe 目标 +
OOS 泛化门** 优化得到。方法学上有两个关键教训（详见 methodology 文档）：
1. **max-年化目标会过拟合对抗测试**：第一次用"max 年化"跑出的 taper config 在 in-sample
   对抗 PASS，但独立 block bootstrap 上通过率从 90% 崩到 63% —— 典型过拟合。
2. **OOS 入环门修复**：改用 max-Sharpe 目标 + Stage C（独立 seed 泛化门），5 个 in-sample
   PASS 候选里**拒掉 4 个过拟合**，只留 1 个在独立 seed 上仍 PASS 的。

**v4.2（前代） vs v4.3（当前生产）**：

| 指标 | v4.2 (rolling10) | v4.3 (taper14+7) |
|---|---|---|
| realized Sharpe | **1.635** | 1.488 |
| realized 年化 | **15.84%** | 14.52% |
| realized MaxDD | 6.75% | **5.84%** |
| Calmar | 2.35 | **2.49** |
| vol 窗口跳变 | 有 | **消除 -27~42%** |
| OOS 三通道通过率 | 基线 | **全 ≥ v4.2**（A 80%>70%, B 100%=, C 93%>90%） |
| def_alloc | 0.145 | 0.349（更防御） |

**选型**：**v4.3 已被 v4.5-pvd 取代为默认生产**（v4.5-pvd 以 v4.3 为基座，PVD 条件激活
进一步提升 Sharpe 且 MaxDD 不恶化）。v4.3 保留为前代已验证配置（无跳变 + 更低回撤 + 高 Calmar，
且经消融确认因子级鲁棒优势）；若更偏好风险调整收益（更高 Sharpe/年化），可切回前代 v4.2：
`python scripts/rebalance_live.py --config config/strategy_v4_2.yaml`。

## 版本演进（三）：v4.3 → v4.4（相关性危机轴闭环，已就绪待切换）

v4.3 的 Layer 3.5（危机相关性加成）存在一个**方法论缺口**：CCC-GARCH 常相关框架的正定性上限
（cap≈1.69）内，c_mult 只能把进攻对相关推到 ≈0.31，远低于 0.6 触发阈值——Layer 3.5 的参数
13 年从未被定向压测（真实触发率仅 5.7%）；旧 c_mult 版 corr_crisis_combo 情景中位 MaxDD
13.78% 穿破 12% 红线。v4.4 闭环该轴：**regime-switching 相关 DGP 压测 + 无状态 EWMA 加权相关
（halflife=8）+ corr_crisis 硬门禁 + 防御四参数圆整**。EWMA 默认关闭时 v4.3 配置在新代码上
**逐位复现**。

| 维度 | v4.3（前代生产，当前回归基线） | v4.4（已就绪） |
|---|---|---|
| realized Sharpe / 年化 / MaxDD | 1.488 / 14.52% / 5.84% | **1.498 / 14.51% / 5.85%** |
| 防御参数 | 0.3492 / 0.0764 / 0.384 / 0.8299 | **圆整 0.35 / 0.075 / 0.38 / 0.83**（三轴校验无损） |
| Layer 3.5 相关估计 | 等权 Pearson（26 周） | **EWMA 半衰期加权（hl=8，无状态，同窗口/阈值/斜率）** |
| Layer 3.5 触发率（真实历史） | 5.59% | 8.76%（< 15% 红线） |
| corr_crisis 硬门禁（7-seed，regime DGP） | —（无此压测轴） | **corr_regime_shift 1.297/7.87%、corr_crisis_combo 1.027/9.66%** |
| evaluate verdict（7-seed，8 情景） | PASS | **PASS**（worst MaxDD 11.28% ≤ 12%） |
| 三通道 OOS（vs v4.3） | 基线 | **TRUE_ROBUST**（core 全 PASS；rho 0.75/0.90 未见变体 worst 8.69%） |
| 联合鲁棒性 Test1/2/3 | 全 PASS | **全 PASS**（胜率 96.5%、α P10 +0.086、方差比 0.793） |
| 测试套件 | 全绿 | **171 passed 全绿** |

**生产状态**：`config/strategy_v4_4.yaml` 已通过全链路校验（影子对照、全管线、一致性终检）；
v4.4 的 EWMA Layer3.5 成果已随 v4.5-pvd 配置继承（crisis_correlation_ewma 段）并切换为生产
默认的一部分。单独试用 v4.4：`--config config/strategy_v4_4.yaml`。完整动机、选型、实现与验收
记录见 [`docs/v4_4_crisis_correlation_closure.md`](docs/v4_4_crisis_correlation_closure.md)。

> **v4.5 灰区相关保护**：曾评估降阈值(M-C)与条件门控(M-D)两条路径修复 grey_corr_combo
> （持续中相关×σ1.5, MaxDD 12.81%>12%），经 36 格实验证实结构性不可行（bond_bear DGP
> 自然相关≈0.40 与灰区物理冲突），**立项中止**。grey_corr_combo 保留为监控情景。
> 详见 [`docs/v4_5_grey_corr_abort.md`](docs/v4_5_grey_corr_abort.md)。

> **v4.5-pvd PVD 条件激活因子（前代生产，当前影子对照）**：量价背离(Price-Volume Divergence)作为动量接近时的 tiebreaker。
> 条件：纳指成交额 ∈ [p25, p75]（expanding 无前视门限）且 top-2 动量 gap < 0.05 时注入
> `0.15×PVD`，否则完全退化为 v4.3 基座。Realized Sharpe 1.603（同窗口 +0.09 vs v4.3），
> MaxDD 5.80%，block bootstrap 200 路径胜率 94.5%。**2026-08 曾切换为生产默认，
> 2026-08-11 由 v4.6 接续**；回退：`python scripts/run_backtest.py --config config/strategy_v4_5_pvd.yaml`。
> 详见 [`docs/v4_5_pvd_factor_closure.md`](docs/v4_5_pvd_factor_closure.md)。

> **v4.6 定向 boost 分级应用（当前生产）**：Layer 3.5 危机相关性保护的**应用点重构**（预研
> 7 变体矩阵唯一全门禁 PASS 的 T1+V3 混合分级）：EWMA(hl=8) corr 触发（thr 0.45 / slope 0.75，
> 覆盖灰区 0.3-0.5）；corr>0.60 显性危机满额防御 boost，corr≤0.60 灰区定向降进攻
> `def += b×(1−def)` 不推高防御绝对水平——修复 v4.5 的灰区缺口（grey_corr_combo 中位
> MaxDD 12.79%→11.80% 回到 12% 红线内）且 bond_bear 不恶化。Realized Sharpe 1.609 /
> MaxDD 5.76%；同期 PE 估值防御调制 E2 通过但 E3 对抗门禁失败已裁出（代码保留默认关），
> R² 动量替换 E2 未过门禁。完整验证链与裁决见
> [`docs/v4_6_directed_boost_closure.md`](docs/v4_6_directed_boost_closure.md)。

> **周内波动(High/Low)探索**：评估 Parkinson/GK 估计器替代 CC-tapered vol，经 E1 信息增量
> 评估（纳指 QDII 溢价扭曲 corr=0.30）+ E2 分资产回测（Mixed Sharpe -0.38）双 NO-GO，
> **课题中止**。详见 [`docs/hl_vol_exploration_abort.md`](docs/hl_vol_exploration_abort.md)。

> **ETF 份额脉冲因子（观察层，不接入资金分配）**：把主力份额的净申赎变化当**稀疏脉冲**
> （滚动 q0.95 低尾，36 次 / 5.28% 的周、横跨 10 个年份）而非每周连续因子来测，共实跑 **150 次回测**。
> E1' 事件研究预注册假说显著（`pool_net_flow|low` → 策略前瞻 4 周回撤，效应量 −0.254、p=0.0103），
> 且与股指期货线相反：市场层面不显著、**策略层面显著**（即 Layer3/3.5 未吸收的那部分回撤）。
> E2 轨 B（脉冲→双向调防御）9/9 格 Ulcer 改善，但 **E3 四门禁仅 2/4**（分期 OOS 符号反转、
> block bootstrap CI 跨零；安慰剂重排 0/50 与 10bp 成本稳健通过）→ **NO-GO（集成）/ GO（观察层）**。
> 轨 A（份额→选股 tiebreaker）是 **no-op、未测出结论**（加分量 0.001~0.003 相对 momentum 0.1~0.3
> 从未翻转排序），**不是证否**。下一轮预注册候选：单边“净流出→多防御” `delta=0.10, hold=4`
> （单边 3×3 网格 9/9 稳健、bootstrap CI 上界仅差 0.0148pp、约需再累积 2 年样本定案）。
> 详见 [`docs/share_pulse_factor_closure.md`](docs/share_pulse_factor_closure.md)，
> 全量表格见 `output/experiments/exp_share_pulse_e2.md`；脚本 `scripts/_exp_share_pulse_e2.py`。

> **股指期货衍生信号（A 线基差 / B 线多空龙虎榜）**：两条线均 **NO-GO**。基差是**同步指标**
> （k=0 同步相关远强于所有领先项，与 tapered_vol14 相关 −0.49）；IF 长样本对照线连一个组合都未过
> 单项门禁。龙虎榜净空变动对**市场**风险有效应（fwd_vol_4w +3.10pp）但对**策略**回撤几乎无效应
> （策略/市场效应量比 0.204）→ 信息已被 Layer3/3.5 吸收。这也是上面份额线的反面对照组。
> 证据见 `output/experiments/exp_futures_basis.md` 与 `exp_futures_holding.md`
> （脚本 `scripts/_exp_futures_basis_study.py` / `_exp_futures_holding_study.py`）。

注：`fut_holding` 是**前 20 名排行榜**而非全市场持仓，数据里不存在真实的 0、只有 NaN；
用 `groupby().sum()` 会把“未上榜”读成“持仓为 0”，必须逐字段 `sum(min_count=1)`。

## v4.0 对抗鲁棒性框架

四节点 + 收尾，实现"realized + adversarial 双维度评估 + 多目标约束优化"：

- **评估入口** (`scripts/evaluate.py`) — 统一双维度评估：realized 历史 + CCC-GARCH 合成对抗；多目标约束判定 `max realized 年化 s.t. 全情景MaxDD≤D_max & realized收益>等权 & 硬机制Sharpe≥等权`；5 机制分维门禁（vol_defense/defense_asset/dispersion/composite 硬门禁，selection 软门禁）。
- **维度约简** (`scripts/dim_reduction.py`) — Morris Elementary Effects，12 候选超参 × 4 轨迹筛出 6 主控。反直觉发现：`vol_w`/`mom_w` 对对抗鲁棒 μ*≈0，Layer1 打分权重与鲁棒性完全正交，鲁棒性只由 Layer3 防御深度/触发点决定。
- **约束优化** (`scripts/optimize.py`) — 6D LHS + 双阶段（3-seed 粗筛 + 7-seed 严验）。v4.1 → v4.2 就是本节点的产物。
- **OOS 验证** (`scripts/oos_validation.py`) — 三通道独立验证：held-out 扰动幅度、独立 seed 集、block bootstrap（跳出 CCC-GARCH 参数族）。核心判定用**相对不劣化**（过拟合假设的直接反驳），envelope 独立记录。

## 策略原理

### 四层架构

| 层次 | 决策 | 方法（v4.5-pvd 生产） |
|------|------|------|
| **Layer 1** 买什么 | 进攻层选 TOP2 | `score = 1.10×mom6 − 1.10×tapered_vol`，PVD 条件激活时 `+0.15×PVD`（纳指 amount ∈[p25,p75] 且 top-2 gap<0.05）；vol 用 tapered14+7 无跳变 |
| **Layer 2** 买多少 | 进攻层权重分配 | inv-vol14（波动率倒数加权，ddof=0，窗口 14 周，与 taper 窗口一致） |
| **Layer 3** 防多少 | 进攻 vs 防御比例 | 纳指 tapered_vol ∈ [**0.076**, **0.384**] → 防御 [**0.349**, **0.830**] 线性插值 |
| **DefAlloc** 防什么 | 红利低波 vs 国债 | `hl_ratio = clip(0.80 − 2.67 × tapered_vol_红利低波, 0, 0.80)` — T=0.30（vol>30%→全国债） |

### 最终参数（v4.5-pvd 生产，附前代对照；v4.5 相对 v4.3 仅 mom_w 1.0→1.1 + pvd_factor 段）

| 参数 | v4.3（前代生产） | v4.2（前代） | v4.1（基线） | 说明 |
|------|:--:|:--:|:--:|------|
| `mom_w` | **1.0** | 1.0 | 1.0 | 动量权重（Morris μ*≈0，不敏感） |
| `vol_w` | **1.10** | 1.10 | 1.10 | 波动率惩罚权重（Morris μ*≈0，不敏感） |
| `mom_window` | **6** | 6 | 6 | 动量计算窗口 |
| vol 估计器 | **tapered 14+7** | rolling 10 | rolling 11 | v4.3 用 tapered 消除窗口跳变 |
| `inv_vol_window` | **14** | 10 | 10 | 波动率倒数平滑窗口（随 vol 有效窗口联动） |
| `step_low` | **0.076** | 0.095 | 0.15 | 防御起效 vol 下限 |
| `step_high` | **0.384** | 0.193 | 0.35 | 极限防御 vol 上限 |
| `def_alloc` | **0.349** | 0.145 | 0.25 | 基准防御比例（v4.3 更高，配宽档距） |
| `max_def` | **0.830** | 0.811 | 0.95 | 峰值防御比例（vol-tier 天花板，crisis boost 可突破） |
| `top_n` | **2** | 2 | 2 | 进攻资产数 |
| `max_single_alloc` | **0.40** | 0.40 | 0.40 | 单只进攻 ETF 权重上限 |
| `rebalance_threshold` | **2.5%** | 2.5% | 2.5% | 调仓触发阈值 |
| `score_margin` | **0.02** | 0.02 | 0.02 | TOP_N 分数差距门槛(防噪声换仓) |
| `dynamic_margin_sensitivity` | **1.0** | 1.0 | 1.0 | 动态 margin 对 score gap 波动率敏感度 |
| `dynamic_margin_window` | **3** | 3 | 3 | 动态 margin 的 score gap 回看窗口(周) |
| `fee_rate` | **0.005%** | 0.005% | 0.005% | 交易费率（单边） |
| `T` (DefAlloc) | **0.30** | 0.30 | 0.30 | 红利低波 vol 红线（领域选择，非调优） |

**参数演化的机制归因**（节点 2 Morris + 节点 3 优化 + 消融）：Layer1 打分权重（`vol_w`/`mom_w`）
对对抗鲁棒 μ*≈0（Layer1 与鲁棒性正交），鲁棒性只由 Layer3 防御参数 + vol 估计器决定。消融
证明 tapered vol 是**因子级真实优势**——同方法下 rolling 找不到可泛化鲁棒配置，唯 taper 能。
v4.3 的"高 def_alloc + 宽档距"配 tapered 平滑 vol，实现无跳变 + 更低 realized 回撤（5.84%）。

### DefAlloc 逻辑

```
hl_ratio = clip(0.80 − 2.67 × tapered_vol(红利低波), 0, 0.80)
T = 0.30: vol ≥ 30% → hl_ratio = 0 → 全国债

T 是领域选择，非超参数：
  · 红利低波 vol 的历史 p90 ≈ 26.89%，取整到 0.30（tapered 与 rolling 分布差 <0.6pp）
  · vol > 30% 的周仅占 ~5%，均为股灾级行情
  · T 在 0.25~0.35 区间 Sharpe 变化 < 0.007，不敏感
```

### 资产池

| ETF | 代码 | 角色 |
|-----|------|------|
| 纳指ETF | 513100.SH | 进攻—海外科技成长 |
| **中证500ETF** | **510500.SH** | **进攻—A股中盘成长（替代沪深300）** |
| 黄金ETF | 518880.SH | 进攻—商品/避险 |
| 红利低波ETF | 512890.SH | 防御—高股息+低波 |
| 国债ETF | 511010.SH | 防御—利率避险 |

## 核心指标

三个口径均由 `scripts/benchmark_compare.py` 统一生成（净值对齐到策略有效区间、共同起点归一化、同一 `compute_metrics`，risk_free=2.5%，默认加载 v4.6 生产 config），可复现。

| 指标 | 策略 v4.6 | 等权(每周再均衡) | 买入持有(不调仓) |
|------|:---:|:---:|:---:|
| Sharpe（标准） | **1.609** | 0.931 | 0.838 |
| 年化收益 | **15.46%** | 12.02% | 12.93% |
| 累计收益 | **526.7%** | 326.0% | 372.5% |
| 最大回撤 | **5.76%** | 20.20% | 29.93% |
| 年化波动 | **7.57%** | 10.07% | 12.48% |
| Calmar | **2.68** | 0.60 | 0.43 |
| 周胜率 | 62.3% | 60.5% | 59.3% |
| 回测区间 | 2013-08-30 ~ 2026-08-07（664周，随周度刷新延伸） | 同左 | 同左 |
| 数据源 | QFQ 前复权 | 同左 | 同左 |

> 注：v4.6 继承 v4.3 的 `def_alloc=0.349` 防御**下限**（vol<step_low 时的基础防御比例），
> 故防御盘常态占比较高（"防御周数"口径下几乎每周都在基线之上，不再是有区分度的指标）；
> 这也是 realized 波动 7.57%、MaxDD 5.76% 显著低于早期前代的结构性原因。

> **两个"什么都不做"基准的对比**：真·买入持有累计收益（372.6%）高于每周再平衡（326.0%），
> 但代价是最大回撤高出近 10pp（29.9% vs 20.2%）、年化波动更大，风险调整后 Sharpe 反而更低
> （0.838 vs 0.931）。原因是买入持有让权重自然漂移、越来越集中于涨幅最大也波动最大的纳指——
> 赢在牛市复利，输在回撤深度；每周再平衡靠强制"高抛低吸 + 维持分散"压住风险、牺牲部分收益。
> 故风险调整后排序为 **策略 > 每周再平衡 > 买入持有**。买入持有的收益优势高度依赖本轮单边科技牛，
> 在 OOS（2024+）段该优势消失（收益反低于再平衡、Sharpe 亦更低），详见 `benchmark_compare.py` 输出。

## 鲁棒性评估（v4.0 对抗框架；以下为 v4.3 口径历史评估记录，v4.5-pvd 验收见 closure 文档）

当前主鲁棒性口径 = **v4.0 对抗鲁棒框架**（`scripts/evaluate.py` + `scripts/oos_validation.py`），
取代了 v4.1 时代的 DSR/MC/PSS 静态评估。v4.3 的评估结果：

| 维度 | 结果（v4.3, 7-seed, D_max=12%） |
|------|:---|
| **多目标约束判定** | **PASS** — realized 收益>等权 ✓、realized DD 5.84%≤12% ✓ |
| **全情景对抗 worst_DD** | **11.95% ≤ 12%** ✓（CCC-GARCH 6 情景 × 7 seed 中位数） |
| **5 机制 Sharpe 门禁** | **全 PASS**：vol_defense 11.03% / defense_asset 11.11% / dispersion 8.24% / composite 11.41% / selection 11.95%（worstDD，Sharpe 胜率均 100%） |
| **3 通道 OOS（vs 前代 v4.2）** | 通过率**全 ≥ v4.2**：A(held-out 幅度) 80%>70%、B(独立 seed) 100%=、C(block bootstrap) 93%>90% |
| **因子级优势（消融）** | 同 max-Sharpe+OOS 门方法下 rolling 0/3 泛化、taper 1/5 泛化 → tapered vol 是真实因子优势非方法假象 |
| **联合鲁棒性（参数×数据）** | **A + B 全 PASS** — 详见下文 |
| **综合结论** | **可上生产**：realized 达标 + 对抗全情景 DD 达标 + 5 机制门禁全过 + OOS 不劣化于前代 + 联合鲁棒 A+B 全过 |

方法学（多目标约束、机制分维门禁、OOS core/envelope 判定、对抗空间局限、消融、**联合鲁棒性 §12**）详见
[`docs/adversarial_robustness_methodology.md`](docs/adversarial_robustness_methodology.md)。

### 联合鲁棒性检验（v4.3 首次基线）

回答一个更本质的问题：**参数邻域 + 数据邻域 + 联合曲面**是否都光滑？(`scripts/robustness_joint.py`)

| 层次 | 设计 | v4.3 结果 | 判据 |
|------|------|:---:|:---:|
| **Test 1** 参数轴 | 8 活参 × ±15% 单参扫描（45 次回测） | 最大 Sharpe 掉幅 −10.3%（top_n 离散跳跃），其他 <6%；无断崖；参数×Sharpe \|ρ\| < 0.13 | **PASS** |
| **Test 2** 数据轴 | 200 次 block bootstrap（block=13w） | 绝对 P10 Sharpe=0.886 但 **EW baseline 自己 P10=0.564**；策略 alpha vs EW 96% 路径为正、alpha P10 = +0.078 | **相对 PASS** |
| **Test 3** 联合 | LHS 200 组 (Δparams ±10%, seed) | 联合方差 0.0774 < 边缘和 0.0970；**交互项 ≈ 0**（QQ 图沿 y=x）；无薄峰 | **PASS** |

**关键判据**（诊断过拟合的正确工具）：

- **联合/边缘方差比 = 0.80 ≤ 1.30**（无薄峰，参数扰动与数据扰动几乎正交）
- **策略跑赢 EW 的 bootstrap 路径 = 96.0%**（真正的策略 alpha 稳健性）
- **参数轴单独方差 = 0.0021**（σ ≈ 0.046，参数最优点位于扁平谷底而非针尖山峰）

绝对分位数（Sharpe P10、MaxDD P90）不区分"策略脆弱"与"历史路径本身的方差"——EW 自己在 bootstrap 下 P10 都掉到 0.564。真正的过拟合诊断是**同一 bootstrap 路径下策略 vs EW 的 alpha**，v4.3 alpha P10=+0.078 / P50=+0.350 / P90=+0.606，96% 胜率。

完整报告（含 445 次回测原始数据、可视化 3 图）：[`output/robustness/report.md`](output/robustness/report.md)。

> **历史评估（v4.1 lineage，仅存档参考，非 v4.3 口径）**：早期用 DSR≈0.999 / MC 生存率 /
> PSS 分位 / 9 窗 Walk-Forward（7/9 vs 再平衡、55.6% reoptimize）评估 v4.1。这套静态指标已被
> 上面的 v4.0 对抗框架取代为主口径。若需按 v4.3 参数复跑 Walk-Forward：
> `python scripts/run_walkforward.py --benchmark`（固定参数稳定性）或 `--reoptimize`（真重选参 OOS）。

## 目录结构

```
claw_etf_strategy/
├── README.md
├── config/
│   ├── strategy_v4_6.yaml               # 当前生产配置 (定向 boost 分级应用, 基于 v4.5-pvd)
│   ├── strategy_v4_5_pvd.yaml           # 前代已验证配置 (PVD 条件激活, 影子对照)
│   ├── strategy_v4_4.yaml               # 已验证配置 (EWMA Layer3.5 + 圆整参数, 全链路校验通过)
│   ├── strategy_v4_3.yaml               # 前代生产/回归基线 (tapered-vol, 无跳变/低回撤)
│   ├── strategy_v4_2.yaml               # 前代已验证配置 (rolling, 高 Sharpe)
│   ├── strategy_v4_1.yaml               # 历史基线 (对抗 OOS 对照, 回归测试参照)
│   └── strategy_v3_1.yaml               # 更早历史版本
├── docs/
│   ├── adversarial_robustness_methodology.md   # v4.0 完整方法学
│   ├── v4_4_crisis_correlation_closure.md      # v4.4 相关性危机轴闭环 (动机/选型/实现/验收)
│   └── premium_management_sop.md               # QDII 溢价管理 SOP (每周运维, 任务23)
├── src/
│   ├── backtest.py                      # 回测引擎
│   ├── strategy.py                      # 策略逻辑 + 配置加载
│   ├── data_loader.py                   # 数据加载
│   ├── factors.py                       # 因子引擎（唯一 ddof=0 源）
│   ├── report.py                        # 报告生成
│   ├── robustness.py                    # 鲁棒性评估(realized 侧)
│   └── utils.py                         # 工具函数
├── scripts/
│   ├── rebalance_live.py                # 实时调仓（每周一用）
│   ├── run_backtest.py                  # 单次回测
│   ├── calc_performance.py              # 绩效对比（当年/近1年/当前回撤）
│   ├── benchmark_compare.py             # 基准对比（策略/每周再平衡/真买入持有，全期+OOS）
│   ├── run_walkforward.py               # Walk-Forward（--reoptimize/--benchmark）
│   ├── cost_sensitivity.py              # 交易成本敏感性分析
│   ├── update_etf_data_tushare.py       # Tushare 数据更新
│   ├── premium_sentinel.py              # 调仓日溢价哨兵 (只提示不切换, rebalance_live --premium-check 调用)
│   ├── # ---- v4.0 对抗鲁棒框架 ----
│   ├── data_manifold.py                 # 真实数据流形拟合(VAR+GARCH), 供合成对抗共享
│   ├── adversarial_robustness.py        # 对抗评估内核(robustness_score + 机制分组)
│   ├── evaluate.py                      # 统一双维度评估 + 多目标约束判定
│   ├── dim_reduction.py                 # Morris 敏感度筛选(12→6 主控)
│   ├── optimize.py                      # 6D LHS 约束优化, 产出下一代候选 config
│   └── oos_validation.py                # 三通道 OOS 验证(held-out/独立seed/block bootstrap)
├── tests/
│   ├── test_factors.py                  # 因子计算单元测试
│   ├── test_strategy.py                 # 策略逻辑单元测试
│   ├── test_engine_core.py              # 回测引擎核心测试
│   ├── test_consistency.py              # 引擎-实盘一致性测试
│   ├── test_benchmark_compare.py        # 三方基准口径回归测试
│   ├── test_no_lookahead.py             # 无前视回归测试
│   ├── test_data_loader.py              # 数据加载测试
│   ├── test_robustness.py               # realized 鲁棒性测试
│   ├── test_config_and_enhancements.py  # 配置与增强测试
│   └── test_audit_fixes.py              # 审计整改 + v4.0 框架结构回归(机制分组/evaluate_full)
├── data/
│   ├── all_etfs_nav_latest.csv          # QFQ 前复权净值
│   └── 300etf_pe_percentile_weekly.csv
├── output/
│   └── adversarial/                     # v4.0 框架产出物
│       ├── baseline_metrics.json        # 基线快照 (当前 = v4.2, 供 --vs-baseline 回归对比)
│       ├── morris_sensitivity.json      # 节点2 敏感度原始数据
│       ├── optimize_stageA.json         # 节点3 LHS 200 点评估结果
│       ├── optimize_stageB.json         # 节点3 精验候选结果
│       └── oos_validation.json          # 节点4 三通道完整证据链
└── .gitignore
```

## 如何运行

### 单次回测
```bash
cd /home/ubuntu/claw_etf_strategy
python scripts/run_backtest.py                                        # 默认 = v4.6 生产 config
python scripts/run_backtest.py --config config/strategy_v4_5_pvd.yaml  # 回退前代 v4.5-pvd
```
> `config/strategy_v4_6.yaml` 已过完整对抗验证管线并于 2026-08-11 切换为生产默认；
> 前代 v4.5-pvd/v4.4/v4.3/v4.2 配置保留，可用 `--config` 随时回退对比。

### 绩效对比（当年/近1年/当前回撤）
```bash
python scripts/calc_performance.py
```

### 基准对比（策略 vs 每周再平衡等权 vs 真·买入持有）
```bash
python scripts/benchmark_compare.py          # 全期(in-sample) + OOS(2024+) 两段
python scripts/benchmark_compare.py --json   # JSON（供回归测试/程序化）
```
三方净值同口径对齐对比；其中"真·买入持有"为一次买入后永不调仓（权重自然漂移），是最朴素的"什么都不做"基准。

### Walk-Forward 基准对比（9 个滚动窗口）
```bash
python scripts/run_walkforward.py --reoptimize --windows 10   # 真·重选参 OOS（每窗训练选参→test，防过拟合）
python scripts/run_walkforward.py --benchmark --windows 10    # 固定参数 vs rebal/buyhold（稳定性，非防过拟合）
python scripts/run_walkforward.py --json                      # JSON 输出
```
`--reoptimize` 每窗用训练段重新选参再到测试段验证，是唯一能回答"参数是否过拟合"的模式；`--benchmark` 用固定生产参数，衡量相对基准的稳定性。

### 实时调仓计算
```bash
python scripts/rebalance_live.py                # 查看调仓
python scripts/rebalance_live.py --save-state   # 确认调仓并保存状态
```
输出下周一持仓方案，含 Layer 1~4 分解 + 阈值基准说明。

### 数据更新
```bash
python scripts/update_etf_data_tushare.py
```

### v4.0 对抗鲁棒框架运行入口

```bash
# 统一评估: realized + adversarial 双维度 + 多目标约束判定 (默认已切换到 v4_5_pvd)
python scripts/evaluate.py --dmax 0.12                                     # 默认 = v4_5_pvd 生产 config
python scripts/evaluate.py --config config/strategy_v4_1.yaml --dmax 0.12  # 历史基线复现
python scripts/evaluate.py --save-baseline                                 # 覆盖 baseline_metrics.json
python scripts/evaluate.py --config <某新yaml> --vs-baseline               # 与基线快照对比

# Morris 敏感度筛选(默认 r=4 轨迹, 12 候选超参 → 6 主控, 约 20-30 min)
python scripts/dim_reduction.py --r 4 --seeds 11,22,33
# 结果: output/adversarial/morris_sensitivity.json

# 6D 约束优化器(默认从 v4_2 起搜, N=200 LHS + top-K=15 精验, 约 1 小时)
# 产出 config/strategy_v4_next.yaml (不覆盖生产, 供 OOS 验证后手动切换)
python scripts/optimize.py --n 200 --k 15 --dmax 0.12

# OOS 三通道验证 (默认 v4_1 历史基线 vs v4_2 当前生产, 约 5 min)
python scripts/oos_validation.py
# 结果: output/adversarial/oos_validation.json (含 core=过拟合假设直接测试 + envelope=独立记录)
```

方法学与设计边界（对抗空间局限、机制分组、决策依据）详见 [`docs/adversarial_robustness_methodology.md`](docs/adversarial_robustness_methodology.md)。

## 运维：QDII 溢价管理（哨兵 + SOP，任务 #19~#23）

### 溢价哨兵（每周调仓日例行）
```bash
python scripts/rebalance_live.py --premium-check   # 调仓计算 + 拉取 513100 及 6 只候选最新溢价
python scripts/premium_sentinel.py                 # 手工诊断入口（仅溢价，直接运行时才联网）
```
哨兵**只提示、不自动切换**，数据获取失败自动降级、不中断调仓主流程。读输出时以**溢价数值**
对照 SOP 纪律表执行（哨兵 2.0%/2.5% 双阈值仅是告警分层，运营纪律线为 **1.5%**，详见下）。

### 操作纪律与 SOP
完整每周流程、纪律表（溢价区间 × 新增/存量 × 动作）、场外联接申赎要点（T+2 时序/对账/限购）、
异常预案见 [`docs/premium_management_sop.md`](docs/premium_management_sop.md)。核心纪律一句话：
**纳指腿新增/加仓在溢价 >1.5% 时走场外联接申购；卖出照常场内；存量只监控不自动动**
（依据与完整决策链见 [`output/experiments/premium_decision.md`](output/experiments/premium_decision.md)，
其中含 E3 对自动开关机制的否决、p*≈2.1% 旧框架的证伪记录、以及存量高溢价仓位"轮换变现"
的人工决策账目）。本节纪律取代早前"溢价 >2% 延迟买入"的旧提示口径。

### 月度复评提示
每月首个调仓日按 SOP §5 清单复评：溢价水平与存量回吐敞口、QDII 额度/限购政策、候选溢价对比、
（若溢价持续 ≥5%）存量变现议题。

### 回测口径披露（重要）
生产数据的纳指列为**含溢价的场内市价**（非净值），因此**所有历史回测业绩含溢价 beta
≈ +1.2pp/年**（E1 口径A，2013-08~2026-07；2024 年后窗口更高）——这部分是 QDII 溢价单边扩张的
红利而非策略 alpha，与"无溢价世界"（NAV 反事实年化 13.30%）或外部基准对比时须先扣除；
各版本（v4.1~v4.4）同口径互比不受影响。详见 `output/experiments/premium_e1_erosion.md` §6
与 `premium_decision.md` §6。

## 注意事项

- 数据列顺序：`日期,纳指ETF,红利低波ETF,中证500ETF,黄金ETF,国债ETF`
- **v4.6 生产沿用 tapered vol**（`vol_taper_enabled: true`, window14+len7），非 rolling、非 EWMA。
  三种 vol 估计器互斥，引擎优先级 ewma > taper > rolling；v4.5-pvd 关 ewma、开 taper。前代 v4.2 用 rolling(10)、v4.1 用 rolling(11)，EWMA 分支保留但默认关闭
- 阈值基准使用状态文件 `data/.last_alloc.json`（上次实仓），首次无状态文件时降级到上周理论仓位
- 确认调仓后请带 `--save-state` 参数保存仓位状态，下次阈值判断更准
- 如有多日频分析需求，日频 DD 比周频高约 0.3~2pp
- 纳指ETF如出现 QDII 溢价 >1.5%，新增按 SOP §3 纪律表走场外联接（哨兵阈值已与纪律线对齐）
- PE 分位因子(pe_percentile)已算但**未接入决策**——策略为纯价量(动量+波动率)，无估值逻辑
- 数据更新脚本基线为 `data/all_etfs_nav_latest.csv`(可用 `ETF_BASE_FILE` 覆盖)，勿指向已弃用的 h20269_scaled 文件
- 策略基于历史回测，**不保证未来收益**
- **v4.0 对抗鲁棒框架**基于 CCC-GARCH 合成 + block bootstrap 的重采样评估，覆盖同分布下的路径不确定性；不能内生地产生 regime switching / DCC / 非对称尾相依（详见 methodology 文档第 8 节）。σ×1.4 类极端复合冲击处于策略族架构上界，需资产池/杠杆结构层面解决而非超参调整
- **v4.6 是当前生产 config**：所有默认路径已切换（实盘脚本支持定向 boost 分级应用并经 `--verify` 与引擎对齐，Δ=0.0066）。回退前代 v4.5-pvd：`--config config/strategy_v4_5_pvd.yaml`；v4.3：`--config config/strategy_v4_3.yaml`；v4.2：`--config config/strategy_v4_2.yaml`；历史基线 v4.1：`--config config/strategy_v4_1.yaml`。完整取向对比见上方版本演进章节
- **v4.6 裁出项留档**：PE 估值防御调制（pe_defense，E2 PASS 但 E3 对抗门禁 FAIL：bond_bear/stagflation 机制 Sharpe 跑输等权）代码保留默认关，待重新设计（如危机条件交互/更低 δ）；R² 动量替换 E2 NO-GO 归档。见 `docs/v4_6_directed_boost_closure.md`
- **稀疏脉冲类因子的评估教训（份额/期货两条线共同固化）**：① 稀疏脉冲不能用全样本连续 IC 判死（隐含“每周都线性有效”），改事件研究 + 真回测；② 非平稳序列禁用 expanding 分位（只认“历史新高”会把触发全挤在少数年份），改滚动分位 + 触发年份跳数门禁；③ MaxDD 不能当唯一主指标（被单一历史极值锚死、对干预不敏感），用 Ulcer / 平均回撤 / 条件回撤；④ 安慰剂“只重排触发日期”比单纯 block bootstrap 更能排除“改善来自动作本身”的混淆；⑤ 区分“证否”与“未测出结论（no-op）”；⑥ 事后拆解必须标注已用自由度与校正后阈值。详见 `docs/share_pulse_factor_closure.md` 第 9 节
- **v4.4 成果已随 v4.5-pvd 继承**：EWMA Layer3.5 + 圆整防御参数均包含在 v4.5-pvd 配置中（v4.4 单独配置保留作对照）。压测入口：`evaluate.py --corr-scenarios`（corr_crisis 硬门禁）、`oos_validation.py --corr-variants`。详见 `docs/v4_4_crisis_correlation_closure.md`