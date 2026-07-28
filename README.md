# 虾池ETF轮动策略 v4.1 — EWMA因子 + 中证500ETF版

基于 **5只ETF** 的周频动量轮动策略，全连续/零门控/四层架构（含 DefAlloc）。
**Sharpe 1.656 / 年化 17.56% / 最大回撤 6.97%**（2013-05-17 ~ 2026-07，665周）。

2026-07 迁移：沪深300ETF → 中证500ETF。

v4.1 升级（2026-07）：因子层从 rolling 窗口切换到 EWMA（指数加权移动平均），消除窗口截断导致的因子跳变。Sharpe 1.610→1.656，对抗性 σ 轴安全余量 +50%，WF OOS 4/5 无过拟合。超参数鲁棒：mom_hl 15-19 宽安全区，vol_hl 4-16 完全不影响。新标的相关性更低（0.245 vs 0.55）、年化收益更高（8.31% vs 6~8%），策略参数同步优化后 Sharpe +0.088, 年化 +1.63pp。

## v4.0 对抗鲁棒性框架（2026-07 新增）

针对"realized 高 Sharpe 只代表历史这一条路径通过"这一结构性风险，v4.0 引入了**对抗性重采样评估 + 多目标约束优化 + 三通道 OOS 验证**的完整框架，把"另一条路径下也不崩"变成可优化、可门禁的量化指标。

四节点 + 收尾：

- **评估入口** (`scripts/evaluate.py`) — 统一双维度评估：realized 历史 + CCC-GARCH 合成对抗；多目标约束判定 `max realized 年化 s.t. 全情景MaxDD≤D_max & realized收益>等权 & 硬机制Sharpe≥等权`；5 机制分维门禁（vol_defense/defense_asset/dispersion/composite 硬门禁，selection 软门禁）。
- **维度约简** (`scripts/dim_reduction.py`) — Morris Elementary Effects，12 候选超参 × 4 轨迹筛出 6 主控。反直觉发现：`vol_w`/`mom_w` 对对抗鲁棒 μ*≈0，Layer1 打分权重与鲁棒性完全正交，鲁棒性只由 Layer3 防御深度/触发点决定。
- **约束优化** (`scripts/optimize.py`) — 6D LHS + 双阶段（3-seed 粗筛 + 7-seed 严验）。产出 `config/strategy_v4_2_robust.yaml`。
- **OOS 验证** (`scripts/oos_validation.py`) — 三通道独立验证：held-out 扰动幅度、独立 seed 集、block bootstrap（跳出 CCC-GARCH 参数族）。核心判定用**相对不劣化**（过拟合假设的直接反驳），envelope 独立记录。

v4_2_robust 相对 v4_1（7-seed 严格对抗 + 3 通道 OOS）：

| 维度 | v4_1（当前生产） | v4_2_robust（鲁棒候选） |
|---|---|---|
| realized 年化 | 17.05% | 15.84%（-1.21pp，换鲁棒代价） |
| realized MaxDD | 6.97% | 6.75% |
| realized Sharpe | 1.610 | **1.635**（反涨；波动降幅 > 收益降幅） |
| 全情景对抗 worst_DD | 12.19% ✗ | **11.60% ✓**（12% 门槛） |
| 硬机制 Sharpe 门禁 | 2/4 FAIL | **4/4 PASS** |
| verdict | FAIL | **PASS** |
| OOS 通道 A 相对不劣化 | 基线 | PASS（pass_rate 50%→70%） |
| OOS 通道 B 相对不劣化 | 基线 | PASS（DD 12.09%→11.63%） |
| OOS 通道 C 相对不劣化（最独立） | 基线 | **PASS（DD 18.89%→13.97%，-4.92pp）** |
| 过拟合判定 | — | **TRUE_ROBUST**（三通道 core PASS） |

**参数变化**（**"轻&快防御"胜过"重&深防御"**，是节点 2+3 的涌现结果，颠覆直觉）：

| 参数 | v4_1 | v4_2_robust | 方向 |
|---|---|---|---|
| `def_alloc` | 0.25 | 0.145 | 基础防御更低 |
| `step_low` | 0.15 | 0.095 | 触发更早 |
| `step_high` | 0.35 | 0.193 | 档间距更紧 |
| `max_def` | 0.95 | 0.811 | 峰值防御更低 |
| `vol_window` | 11 | 10 | vol 信号更快 |

**当前状态**：`v4_1` 仍是**当前生产配置**（README 上方描述与实盘调仓脚本 `rebalance_live.py` 默认加载它）；`v4_2_robust` 是**经完整对抗验证的鲁棒候选**，供用户在评估切换代价（-1.21pp 年化）与鲁棒收益（对抗全过、通道 C DD -4.92pp）之间做决策。切换用 `python scripts/rebalance_live.py --config config/strategy_v4_2_robust.yaml`。

**方法学、决策依据、对抗空间局限**：见 [`docs/adversarial_robustness_methodology.md`](docs/adversarial_robustness_methodology.md)，含 σ×1.4 这类"策略族架构上界"与超参优化边界的完整讨论。

## 策略原理

### 四层架构

| 层次 | 决策 | 方法 |
|------|------|------|
| **Layer 1** 买什么 | 进攻层选 TOP2 | `score = mom6 − 1.10×vol11`（mom_w=1 固定，vol_w=1.10, vol_window=11） |
| **Layer 2** 买多少 | 进攻层权重分配 | inv-vol10（波动率倒数加权，ddof=0，窗口 10 周） |
| **Layer 3** 防多少 | 进攻 vs 防御比例 | 纳指 vol11 ∈ [15%, 35%] → 防御 [25%, 95%] 线性插值 |
| **DefAlloc** 防什么 | 红利低波 vs 国债 | `hl_ratio = clip(0.80 − 2.67 × vol11_红利低波, 0, 0.80)` — T=0.30（vol>30%→全国债） |

### 最终参数

| 参数 | 值 | 安全边界 | 说明 |
|------|:--:|:--------:|------|
| `mom_w` | **1.0** | 固定 | 动量权重 |
| `vol_w` | **1.10** | 0.80~1.20（MC=100%） | 波动率惩罚权重 |
| `mom_window` | **6** | 3~6 安全 | 动量计算窗口 |
| `vol_window` | **11** | 10~16 安全 | 波动率计算窗口 |
| `inv_vol_window` | **10** | ≥8 安全 | 波动率倒数平滑窗口 |
| `step_low` | **0.15** | 0.12~0.20 安全 | 防御起效的 vol 下限 |
| `step_high` | **0.35** | 0.25~0.45 安全 | 极限防御的 vol 上限 |
| `def_alloc` | **0.25** | 0.20~0.35 安全 | 基准防御比例 |
| `max_def` | **0.95** | — | 极限防御比例 |
| `max_single_alloc` | **0.40** | 0.35~0.50 零影响 | 单只进攻ETF 权重上限 |
| `rebalance_threshold` | **2.5%** | 0~5% 波动 <0.003 Sharpe | 调仓触发阈值 |
| `score_margin` | **0.02** | 0.005~0.05 有效, 0.02最优 | TOP_N 分数差距门槛(防噪声换仓) |
| `dynamic_margin_sensitivity` | **1.0** | 0.5~1.5 | 动态 margin 对 score gap 波动率的敏感度 |
| `dynamic_margin_window` | **3** | 2~5 | 动态 margin 的 score gap 回看窗口(周) |
| `fee_rate` | **0.005%** | 单边 | 交易费率 |
| `T` (DefAlloc) | **0.30** | 0.25~0.35 稳定 | 红利低波 vol 红线（领域选择，非调优） |

### DefAlloc 逻辑

```
hl_ratio = clip(0.80 − 2.67 × vol11(红利低波), 0, 0.80)
T = 0.30: vol ≥ 30% → hl_ratio = 0 → 全国债

T 是领域选择，非超参数：
  · 红利低波 vol11 的历史 p90 ≈ 26.89%，取整到 0.30
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

三个口径均由 `scripts/benchmark_compare.py` 统一生成（净值对齐到策略有效区间、共同起点归一化、同一 `compute_metrics`，risk_free=2.5%），可复现。

| 指标 | 策略 | 等权(每周再均衡) | 买入持有(不调仓) |
|------|:---:|:---:|:---:|
| Sharpe（标准） | **1.610** | 0.918 | 0.818 |
| 年化收益 | 17.05% | 11.84% | 12.59% |
| 累计收益 | **649.1%** | 318.3% | 355.7% |
| 最大回撤 | **6.97%** | 20.20% | 30.12% |
| 年化波动 | **8.47%** | 10.03% | 12.41% |
| Calmar | **2.45** | 0.59 | 0.42 |
| 周胜率 | 61.1% | 60.6% | 59.5% |
| 防御周数 | **349 / 665（52%）** | N/A | N/A |
| 回测区间 | 2013-08-09 ~ 2026-07-24（665周） | 同左 | 同左 |
| 数据源 | QFQ 前复权 | 同左 | 同左 |

> **两个"什么都不做"基准的对比**：真·买入持有累计收益（355.7%）高于每周再平衡（318.3%），
> 但代价是最大回撤高出近 10pp（30.1% vs 20.2%）、年化波动更大，风险调整后 Sharpe 反而更低
> （0.818 vs 0.918）。原因是买入持有让权重自然漂移、越来越集中于涨幅最大也波动最大的纳指——
> 赢在牛市复利，输在回撤深度；每周再平衡靠强制"高抛低吸 + 维持分散"压住风险、牺牲部分收益。
> 故风险调整后排序为 **策略 > 每周再平衡 > 买入持有**。买入持有的收益优势高度依赖本轮单边科技牛，
> 在 OOS（2024+）段该优势消失（收益反低于再平衡、Sharpe 亦更低），详见 `benchmark_compare.py` 输出。

## 鲁棒性评估

| 指标 | 结果 |
|------|:---:|
| DSR（Deflated Sharpe，n_trials=30 保守矫正） | **≈0.999** 🟡（已修正两处口径 bug：年化SR→每期SR 与 n_obs 同频、超额峰度口径；正确口径下仍≈0.999 统计显著。真正敏感点是 n_trials：若真实调参次数达上千，DSR 降至 ~0.92——乐观来自试验次数低估而非公式） |
| MC 生存率（400次±15%扰动，仅活跃参数） | **见鲁棒性报告** 🟡（已修正：仅扰动 11 个真正生效参数，剔除 D4 no-op；生存标准 Sharpe≥1.0 & DD<10%） |
| PSS 收益 P50 / P10 / P90 | 15.9% / 10.2% / 23.7% |
| PSS DD P50 / P10 / P90 | 6.7% / 5.5% / 9.2% |
| PSS Sharpe P50 / P10 / P90 | 1.520 / 1.102 / 1.793 |
| WF（固定参数稳定性，`--benchmark`） | 9 窗：**7/9 vs 每周再平衡**、**8/9 vs 真·买入持有**。衡量固定策略相对基准的稳定性，非防过拟合 |
| WF（真·重选参 OOS，`--reoptimize`） | 9 窗每窗 anchored 训练重选参→test 验证(不重拟合)：**7/9 vs 每周再平衡**、**7/9 vs 真·买入持有**；IS→OOS Sharpe 退化 **-0.18**(测试期反更高，无过拟合)；9 窗中 8 窗独立收敛到同组参数。唯一两败窗 2015-12~2017-04(低波慢牛、动量失效)；熊市窗 2021-03~2022-07 两基准皆负、策略仍正(防御层真实贡献) |
| **综合评级** | **🟡 基本可上（WF 55.6%，train/test 无过拟合，策略在低风险环境中跑输等权）** |

## 目录结构

```
claw_etf_strategy/
├── README.md
├── config/
│   ├── strategy_v4_1.yaml               # 当前生产配置
│   ├── strategy_v4_2_robust.yaml        # v4.0 框架产出鲁棒候选
│   └── strategy_v3_1.yaml               # 历史版本
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
│   ├── optimize.py                      # 6D LHS 约束优化, 产出 v4_2_robust config
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
│       ├── baseline_metrics.json        # 基线快照(当前=v4_2_robust)
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
# 统一评估: realized + adversarial 双维度 + 多目标约束判定
python scripts/evaluate.py --config config/strategy_v4_1.yaml --dmax 0.12
python scripts/evaluate.py --config config/strategy_v4_2_robust.yaml --dmax 0.12
python scripts/evaluate.py --config config/strategy_v4_2_robust.yaml --save-baseline   # 存基线快照
python scripts/evaluate.py --config <某新yaml> --vs-baseline                            # 与基线快照对比

# Morris 敏感度筛选(默认 r=4 轨迹, 12 候选超参 → 6 主控, 约 20-30 min)
python scripts/dim_reduction.py --r 4 --seeds 11,22,33
# 结果: output/adversarial/morris_sensitivity.json

# 6D 约束优化器(默认 N=200 LHS + top-K=15 精验, 约 1 小时)
python scripts/optimize.py --n 200 --k 15 --dmax 0.12
# 结果: output/adversarial/optimize_stage{A,B}.json + config/strategy_v4_2_robust.yaml

# OOS 三通道验证(默认 v4_1 vs v4_2_robust, 约 5 min)
python scripts/oos_validation.py
# 结果: output/adversarial/oos_validation.json (含 core=过拟合假设直接测试 + envelope=独立记录)
```

方法学与设计边界（对抗空间局限、机制分组、决策依据）详见 [`docs/adversarial_robustness_methodology.md`](docs/adversarial_robustness_methodology.md)。

## 注意事项

- 数据列顺序：`日期,纳指ETF,红利低波ETF,中证500ETF,黄金ETF,国债ETF`
- **v4.1 默认使用 EWMA 因子（ewma_mom_halflife=16, ewma_vol_halflife=6）**，消除 rolling 窗口截断跳变；可通过 `ewma_factors_enabled: false` 回退到 rolling
- 阈值基准使用状态文件 `data/.last_alloc.json`（上次实仓），首次无状态文件时降级到上周理论仓位
- 确认调仓后请带 `--save-state` 参数保存仓位状态，下次阈值判断更准
- 如有多日频分析需求，日频 DD 比周频高约 0.3~2pp
- 纳指ETF如出现 QDII 溢价 >2%，需人工判断是否延迟买入
- PE 分位因子(pe_percentile)已算但**未接入决策**——策略为纯价量(动量+波动率)，无估值逻辑
- 数据更新脚本基线为 `data/all_etfs_nav_latest.csv`(可用 `ETF_BASE_FILE` 覆盖)，勿指向已弃用的 h20269_scaled 文件
- 策略基于历史回测，**不保证未来收益**
- **v4.0 对抗鲁棒框架**基于 CCC-GARCH 合成 + block bootstrap 的重采样评估，覆盖同分布下的路径不确定性；不能内生地产生 regime switching / DCC / 非对称尾相依（详见 methodology 文档第 8 节）。σ×1.4 类极端复合冲击处于策略族架构上界，需资产池/杠杆结构层面解决而非超参调整
- **v4_2_robust 是候选而非当前生产**：`rebalance_live.py` 默认加载 v4_1；切换请显式指定 `--config config/strategy_v4_2_robust.yaml`，切换权衡见上方 v4.0 章节对比表