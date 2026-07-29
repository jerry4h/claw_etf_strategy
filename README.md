# 虾池ETF轮动策略 v4.3 — tapered-vol 生产版（无窗口跳变）

基于 **5只ETF** 的周频动量轮动策略，全连续/零门控/四层架构（含 DefAlloc）。
**Sharpe 1.488 / 年化 14.52% / 最大回撤 5.84% / Calmar 2.49**（realized 2013-08 ~ 2026-07）；
**tapered vol 消除 rolling 窗口跳变（-27~42%）；对抗 3 通道 OOS 通过率全 ≥ 前代 v4.2**。

v4.3 由 **v4.0 对抗鲁棒性框架 + max-Sharpe 目标 + OOS 泛化门** 产出。经**控制变量消融实验**
确认：在相同优化方法下，rolling 找不到任何可泛化的鲁棒配置（0/3 过 OOS 门），唯 tapered vol
能（1/5）——**taper 是因子级的真实优势，不是方法的假象**。方法学详见
[`docs/adversarial_robustness_methodology.md`](docs/adversarial_robustness_methodology.md)。

**生产状态**：`config/strategy_v4_3.yaml` 为**默认生产 config**（`rebalance_live.py` /
`run_backtest.py` / `evaluate.py` 等默认路径已全部切换，实盘脚本已支持 tapered vol 且经
`--verify` 确认与回测引擎一致 ΔSharpe<0.01）。前代 `config/strategy_v4_2.yaml`（rolling，
Sharpe 1.635/更高年化但 vol 有跳变）保留为**已验证替代配置**；如需切回：
`python scripts/rebalance_live.py --config config/strategy_v4_2.yaml`。

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

**版本沿革**：`config/strategy_v4_2.yaml`（rolling）曾是生产配置，现已被 **v4.3 取代**、降为
**前代已验证配置**（对抗 OOS 对照 + 回归 pin 保留）；`config/strategy_v4_1.yaml` 仍作**历史基线**
（对抗 OOS 验证的对照组、回归测试的历史行为参照）。当前默认生产 = v4.3（见顶部）。

## 版本演进（二）：v4.2 → v4.3（当前生产）

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

**选型**：**v4.3 已是默认生产**（无跳变 + 更低回撤 + 更高 Calmar，且经消融确认因子级鲁棒优势）。
若更偏好风险调整收益（更高 Sharpe/年化），可切回前代 v4.2：
`python scripts/rebalance_live.py --config config/strategy_v4_2.yaml`。

## v4.0 对抗鲁棒性框架

四节点 + 收尾，实现"realized + adversarial 双维度评估 + 多目标约束优化"：

- **评估入口** (`scripts/evaluate.py`) — 统一双维度评估：realized 历史 + CCC-GARCH 合成对抗；多目标约束判定 `max realized 年化 s.t. 全情景MaxDD≤D_max & realized收益>等权 & 硬机制Sharpe≥等权`；5 机制分维门禁（vol_defense/defense_asset/dispersion/composite 硬门禁，selection 软门禁）。
- **维度约简** (`scripts/dim_reduction.py`) — Morris Elementary Effects，12 候选超参 × 4 轨迹筛出 6 主控。反直觉发现：`vol_w`/`mom_w` 对对抗鲁棒 μ*≈0，Layer1 打分权重与鲁棒性完全正交，鲁棒性只由 Layer3 防御深度/触发点决定。
- **约束优化** (`scripts/optimize.py`) — 6D LHS + 双阶段（3-seed 粗筛 + 7-seed 严验）。v4.1 → v4.2 就是本节点的产物。
- **OOS 验证** (`scripts/oos_validation.py`) — 三通道独立验证：held-out 扰动幅度、独立 seed 集、block bootstrap（跳出 CCC-GARCH 参数族）。核心判定用**相对不劣化**（过拟合假设的直接反驳），envelope 独立记录。

## 策略原理

### 四层架构

| 层次 | 决策 | 方法（v4.3 生产） |
|------|------|------|
| **Layer 1** 买什么 | 进攻层选 TOP2 | `score = mom6 − 1.10×tapered_vol`（mom_w=1 固定，vol_w=1.10；vol 用 tapered14+7 无跳变） |
| **Layer 2** 买多少 | 进攻层权重分配 | inv-vol14（波动率倒数加权，ddof=0，窗口 14 周，与 taper 窗口一致） |
| **Layer 3** 防多少 | 进攻 vs 防御比例 | 纳指 tapered_vol ∈ [**0.076**, **0.384**] → 防御 [**0.349**, **0.830**] 线性插值 |
| **DefAlloc** 防什么 | 红利低波 vs 国债 | `hl_ratio = clip(0.80 − 2.67 × tapered_vol_红利低波, 0, 0.80)` — T=0.30（vol>30%→全国债） |

### 最终参数（v4.3 生产，附前代对照）

| 参数 | v4.3（当前生产） | v4.2（前代） | v4.1（基线） | 说明 |
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

三个口径均由 `scripts/benchmark_compare.py` 统一生成（净值对齐到策略有效区间、共同起点归一化、同一 `compute_metrics`，risk_free=2.5%，默认加载 v4.3 生产 config），可复现。

| 指标 | 策略 v4.3 | 等权(每周再均衡) | 买入持有(不调仓) |
|------|:---:|:---:|:---:|
| Sharpe（标准） | **1.488** | 0.906 | 0.812 |
| 年化收益 | **14.52%** | 11.72% | 12.54% |
| 累计收益 | **461.9%** | 309.9% | 350.1% |
| 最大回撤 | **5.84%** | 20.20% | 29.93% |
| 年化波动 | **7.64%** | 10.05% | 12.46% |
| Calmar | **2.49** | 0.58 | 0.42 |
| 周胜率 | 60.7% | 60.4% | 59.2% |
| 回测区间 | 2013-08-30 ~ 2026-07-24（662周） | 同左 | 同左 |
| 数据源 | QFQ 前复权 | 同左 | 同左 |

> 注：v4.3 `def_alloc=0.349` 是防御**下限**（vol<step_low 时的基础防御比例），
> 故防御盘常态占比较高（"防御周数"口径下几乎每周都在基线之上，不再是有区分度的指标）；
> 这也是 v4.3 realized 波动 7.64%、MaxDD 5.84% 显著低于前代的结构性原因。

> **两个"什么都不做"基准的对比**：真·买入持有累计收益（350.1%）高于每周再平衡（309.9%），
> 但代价是最大回撤高出近 10pp（29.9% vs 20.2%）、年化波动更大，风险调整后 Sharpe 反而更低
> （0.812 vs 0.906）。原因是买入持有让权重自然漂移、越来越集中于涨幅最大也波动最大的纳指——
> 赢在牛市复利，输在回撤深度；每周再平衡靠强制"高抛低吸 + 维持分散"压住风险、牺牲部分收益。
> 故风险调整后排序为 **策略 > 每周再平衡 > 买入持有**。买入持有的收益优势高度依赖本轮单边科技牛，
> 在 OOS（2024+）段该优势消失（收益反低于再平衡、Sharpe 亦更低），详见 `benchmark_compare.py` 输出。

## 鲁棒性评估（v4.0 对抗框架，v4.3 生产）

当前主鲁棒性口径 = **v4.0 对抗鲁棒框架**（`scripts/evaluate.py` + `scripts/oos_validation.py`），
取代了 v4.1 时代的 DSR/MC/PSS 静态评估。v4.3 的评估结果：

| 维度 | 结果（v4.3, 7-seed, D_max=12%） |
|------|:---|
| **多目标约束判定** | **PASS** — realized 收益>等权 ✓、realized DD 5.84%≤12% ✓ |
| **全情景对抗 worst_DD** | **11.95% ≤ 12%** ✓（CCC-GARCH 6 情景 × 7 seed 中位数） |
| **5 机制 Sharpe 门禁** | **全 PASS**：vol_defense 11.03% / defense_asset 11.11% / dispersion 8.24% / composite 11.41% / selection 11.95%（worstDD，Sharpe 胜率均 100%） |
| **3 通道 OOS（vs 前代 v4.2）** | 通过率**全 ≥ v4.2**：A(held-out 幅度) 80%>70%、B(独立 seed) 100%=、C(block bootstrap) 93%>90% |
| **因子级优势（消融）** | 同 max-Sharpe+OOS 门方法下 rolling 0/3 泛化、taper 1/5 泛化 → tapered vol 是真实因子优势非方法假象 |
| **综合结论** | **可上生产**：realized 达标 + 对抗全情景 DD 达标 + 5 机制门禁全过 + OOS 不劣化于前代 |

方法学（多目标约束、机制分维门禁、OOS core/envelope 判定、对抗空间局限、消融）详见
[`docs/adversarial_robustness_methodology.md`](docs/adversarial_robustness_methodology.md)。

> **历史评估（v4.1 lineage，仅存档参考，非 v4.3 口径）**：早期用 DSR≈0.999 / MC 生存率 /
> PSS 分位 / 9 窗 Walk-Forward（7/9 vs 再平衡、55.6% reoptimize）评估 v4.1。这套静态指标已被
> 上面的 v4.0 对抗框架取代为主口径。若需按 v4.3 参数复跑 Walk-Forward：
> `python scripts/run_walkforward.py --benchmark`（固定参数稳定性）或 `--reoptimize`（真重选参 OOS）。

## 目录结构

```
claw_etf_strategy/
├── README.md
├── config/
│   ├── strategy_v4_3.yaml               # 当前生产配置 (tapered-vol, 无跳变/低回撤)
│   ├── strategy_v4_2.yaml               # 前代已验证配置 (rolling, 高 Sharpe)
│   ├── strategy_v4_1.yaml               # 历史基线 (对抗 OOS 对照, 回归测试参照)
│   └── strategy_v3_1.yaml               # 更早历史版本
├── docs/
│   └── adversarial_robustness_methodology.md   # v4.0 完整方法学
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
python scripts/run_backtest.py
```

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
# 统一评估: realized + adversarial 双维度 + 多目标约束判定 (默认已切换到 v4_3)
python scripts/evaluate.py --dmax 0.12                                     # 默认 = v4_3 生产 config
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

## 注意事项

- 数据列顺序：`日期,纳指ETF,红利低波ETF,中证500ETF,黄金ETF,国债ETF`
- **v4.3 生产用 tapered vol**（`vol_taper_enabled: true`, window14+len7），非 rolling、非 EWMA。
  三种 vol 估计器互斥，引擎优先级 ewma > taper > rolling；v4.3 关 ewma、开 taper。前代 v4.2 用 rolling(10)、v4.1 用 rolling(11)，EWMA 分支保留但默认关闭
- 阈值基准使用状态文件 `data/.last_alloc.json`（上次实仓），首次无状态文件时降级到上周理论仓位
- 确认调仓后请带 `--save-state` 参数保存仓位状态，下次阈值判断更准
- 如有多日频分析需求，日频 DD 比周频高约 0.3~2pp
- 纳指ETF如出现 QDII 溢价 >2%，需人工判断是否延迟买入
- PE 分位因子(pe_percentile)已算但**未接入决策**——策略为纯价量(动量+波动率)，无估值逻辑
- 数据更新脚本基线为 `data/all_etfs_nav_latest.csv`(可用 `ETF_BASE_FILE` 覆盖)，勿指向已弃用的 h20269_scaled 文件
- 策略基于历史回测，**不保证未来收益**
- **v4.0 对抗鲁棒框架**基于 CCC-GARCH 合成 + block bootstrap 的重采样评估，覆盖同分布下的路径不确定性；不能内生地产生 regime switching / DCC / 非对称尾相依（详见 methodology 文档第 8 节）。σ×1.4 类极端复合冲击处于策略族架构上界，需资产池/杠杆结构层面解决而非超参调整
- **v4.3 是当前生产 config**：所有默认路径已切换（实盘脚本已支持 tapered vol 并经 `--verify` 与引擎对齐）。回退前代 v4.2：`--config config/strategy_v4_2.yaml`；用历史基线 v4.1：`--config config/strategy_v4_1.yaml`。完整取向对比见上方版本演进章节