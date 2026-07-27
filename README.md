# 虾池ETF轮动策略 v3.1 — 中证500ETF版

基于 **5只ETF** 的周频动量轮动策略，全连续/零门控/四层架构（含 DefAlloc）。
**Sharpe 1.610 / 年化 17.05% / 最大回撤 6.97%**（2013-05-17 ~ 2026-07，665周）。

2026-07 迁移：沪深300ETF → 中证500ETF。新标的相关性更低（0.245 vs 0.55）、年化收益更高（8.31% vs 6~8%），策略参数同步优化后 Sharpe +0.088, 年化 +1.63pp。

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
| DSR（Deflated Sharpe，n_trials=30 保守矫正） | **≈1.0000** 🟡（已修正公式：n=664 周使 SR 估计标准误极小→真实显著，非钳制；n_trials 为保守估计，真实变体数可能更大） |
| MC 生存率（400次±15%扰动，仅活跃参数） | **见鲁棒性报告** 🟡（已修正：仅扰动 11 个真正生效参数，剔除 D4 no-op；生存标准 Sharpe≥1.0 & DD<10%） |
| PSS 收益 P50 / P10 / P90 | 15.9% / 10.2% / 23.7% |
| PSS DD P50 / P10 / P90 | 6.7% / 5.5% / 9.2% |
| PSS Sharpe P50 / P10 / P90 | 1.520 / 1.102 / 1.793 |
| WF 相对胜率（9个滚动窗口，生产参数固定） | **7/9 vs 每周再平衡（77.8%）**，**8/9 vs 真·买入持有（88.9%）**；唯一两败窗口为 2015-12~2017-04（低波慢牛、动量失效）；熊市窗口（2021-03~2022-07）基准皆负、策略仍正（防御层真实贡献） |
| **综合评级** | **🟡 基本可上（WF 55.6%，train/test 无过拟合，策略在低风险环境中跑输等权）** |

## 目录结构

```
claw_etf_strategy/
├── README.md
├── config/
│   └── strategy_v3_1.yaml               # 当前配置
├── src/
│   ├── backtest.py                      # 回测引擎
│   ├── strategy.py                      # 策略逻辑 + 配置加载
│   ├── data_loader.py                   # 数据加载
│   ├── factors.py                       # 因子引擎（唯一 ddof=0 源）
│   ├── report.py                        # 报告生成
│   ├── robustness.py                    # 鲁棒性评估
│   └── utils.py                         # 工具函数
├── scripts/
│   ├── rebalance_live.py                # 实时调仓（每周一用）
│   ├── run_backtest.py                  # 单次回测
│   ├── calc_performance.py              # 绩效对比（当年/近1年/当前回撤）
│   ├── benchmark_compare.py             # 基准对比（策略/每周再平衡/真买入持有，全期+OOS）
│   ├── run_walkforward.py               # Walk-Forward 验证（含 --benchmark 基准对比模式）
│   ├── cost_sensitivity.py              # 交易成本敏感性分析
│   └── update_etf_data_tushare.py       # Tushare 数据更新
├── tests/
│   ├── test_factors.py                  # 因子计算单元测试
│   ├── test_strategy.py                 # 策略逻辑单元测试
│   ├── test_consistency.py              # 引擎-实盘一致性测试
│   └── test_benchmark_compare.py        # 三方基准口径回归测试
├── data/
│   ├── all_etfs_nav_latest.csv          # QFQ 前复权净值
│   └── 300etf_pe_percentile_weekly.csv
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
python scripts/run_walkforward.py --rolling --windows 10 --benchmark   # 9 窗 vs rebal / buyhold
python scripts/run_walkforward.py --json                               # JSON 输出
```
逐窗判定策略 Sharpe 是否跑赢两个基准，汇总胜率。默认生产参数固定（不重拟合）。

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

## 注意事项

- 数据列顺序：`日期,纳指ETF,红利低波ETF,中证500ETF,黄金ETF,国债ETF`
- **所有因子计算强制使用 `src/factors.py`（ddof=0）**，禁止自行实现
- 阈值基准使用状态文件 `data/.last_alloc.json`（上次实仓），首次无状态文件时降级到上周理论仓位
- 确认调仓后请带 `--save-state` 参数保存仓位状态，下次阈值判断更准
- 如有多日频分析需求，日频 DD 比周频高约 0.3~2pp
- 纳指ETF如出现 QDII 溢价 >2%，需人工判断是否延迟买入
- 策略基于历史回测，**不保证未来收益**