# 虾池ETF轮动策略 v3.0

基于**5只ETF**的周频动量轮动策略，全新 QFQ 前复权数据回测：
**Sharpe 1.311 / 年化 14.38% / 最大回撤 7.52%**（2013-2026）。

## 策略原理

### 三层架构，全部零门控/零阈值

| 层次 | 决策 | 方法 |
|------|------|------|
| **Layer 1** 买什么 | 进攻层选 TOP2 | `score = mom4 − 1.05×vol20`（mom_w=1 固定） |
| **Layer 2** 买多少 | 进攻层内权重分配 | inv-vol12（波动率倒数加权，ddof=0） |
| **Layer 3** 防多少 | 进攻 vs 防御比例 | 纳指 vol20 ∈ [15%, 35%] → 防御 [25%, 95%] 线性插值 |

> 相比等权持仓（TOP2 各 50%），inv-vol12 提升 Sharpe +9.9%（1.311 vs 1.193），
> 终值差距 +8.3%。

### 超参数最终版

| 参数 | 值 | 安全边界 | 说明 |
|------|:--:|:--------:|------|
| `mom_w` | **1.0** | 固定 | 动量权重 |
| `vol_w` | **1.05** | 0.80~1.20（MC=100%） | 波动率惩罚权重 |
| `inv_vol_window` | **12** | ≥8 安全，≤5 MC 降至 17% | 波动率倒数平滑窗口 |
| `step_low` | **0.15** | 0.12~0.20 安全 | 防御开始触发的 vol 下限 |
| `step_high` | **0.35** | 0.25~0.45 安全 | 极限防御的 vol 上限 |
| `max_single_alloc` | **0.40** | 0.35~0.50 零影响 | 单只 ETF 权重上限 |
| `rebalance_threshold` | **7%** | 0%~10% 波动<0.04 Sharpe | 调仓触发阈值 |
| `fee_rate` | **0.005%** | ×2 双边 | 单边交易费率 |

### 资产池

| ETF | 代码 | 角色 |
|-----|------|------|
| 纳指ETF | 513100.SH | 进攻—海外成长 |
| 沪深300ETF | 510300.SH | 进攻—A股大盘 |
| 黄金ETF | 518880.SH | 进攻—商品/避险 |
| 红利低波ETF | 512890.SH | 防御—低波+股息 |
| 国债ETF | 511010.SH | 防御—利率避险 |

### 调仓阈值的影响

| 阈值 | Sharpe | 年化 | DD | 交易次数 |
|:---:|:-----:|:---:|:--:|:------:|
| **0%** | **1.313** | 14.3% | 7.6% | 648 |
| 1% | 1.312 | 14.3% | 7.5% | 648 |
| 2% | 1.311 | 14.3% | 7.5% | 647 |
| **7%** | **1.311** | 14.4% | 7.5% | 642 |
| 10% | 1.312 | 14.4% | 7.4% | 636 |

> **结论**：阈值 0~10% 的 Sharpe 波动 <0.003，**几乎无影响**。当前 7% 已充分。
> 0% 阈值（每周调仓）全年多交易 ~0.5 次，Sharpe 无实质提升。

## 目录结构

```
claw_etf_strategy/
├── README.md
├── config/
│   └── strategy_v3_0_final.yaml    # 最终版配置
├── src/
│   ├── backtest.py                 # 回测引擎
│   ├── strategy.py                 # 策略逻辑 + 配置加载
│   ├── data_loader.py              # 数据加载
│   ├── factors.py                  # 因子引擎（唯一 ddof=0 源）
│   ├── report.py                   # 报告生成
│   ├── robustness.py               # 鲁棒性评估（DSR/PSS/MC/WF/SPS）
│   └── utils.py                    # 工具函数
├── scripts/
│   ├── rebalance_live.py           # **实时调仓**（每周一用）
│   ├── run_backtest.py             # 单次回测
│   ├── run_robustness.py           # 鲁棒性评估 CLI
│   ├── run_grid_search.py          # 网格搜索
│   ├── optimize_v3_0_newdata.py    # 超参优化
│   ├── compare_layer2.py           # inv-vol vs 等权对比
│   ├── update_etf_data_tushare.py  # tushare 数据更新
│   └── ...
├── data/
│   ├── all_etfs_nav_2013_20260626.csv  # QFQ 前复权净值（主数据）
│   └── 300etf_pe_percentile_weekly.csv
├── output/
│   └── hyperparam_newdata.csv          # 超参扫描全量结果
└── docs/
    └── etf_data_build.md              # PM 数据构建说明
```

## 如何运行

### 单次回测

```bash
cd /home/ubuntu/claw_etf_strategy
python scripts/run_backtest.py --config config/strategy_v3_0_final.yaml
```

### 实时调仓计算

```bash
python scripts/rebalance_live.py
```

输出下周一持仓方案，含 Layer 1~3 分解。

### 超参优化

```bash
python scripts/optimize_v3_0_newdata.py
```

### 鲁棒性评估

```bash
python scripts/run_robustness.py --config config/strategy_v3_0_final.yaml \
  --output output/robustness_v3_final/ --n-mc 400 --oat --n-wf 9
```

## 核心指标

| 指标 | 值 |
|------|:--:|
| Sharpe | **1.311** |
| 年化收益 | **14.38%** |
| 最大回撤 | **7.52%** |
| 年化波动 | **8.64%** |
| 周胜率 | **60.4%** |
| 防御周数 | **398 / 652（61%）** |
| 回测区间 | 2013-05 ~ 2026-06 |
| 数据源 | QFQ 前复权（见 docs/etf_data_build.md） |

## 鲁棒性评估结果

| 指标 | 结果 |
|------|:---:|
| DSR（Deflated Sharpe） | **1.0000** 🟢（>95% 真实alpha） |
| MC 生存率（400次扰动） | **100%** 🟢 |
| WF 相对胜率（9窗口） | **77.8%** 🟡 |
| PSS CV（收益/DD/Sharpe） | **0.01~0.02** 🟢 |
| **综合评级** | **🟢 实盘可上** |

### 四维超参边界

| 参数 | 安全范围 | 失效边界 |
|------|:--------:|:--------:|
| `vol_w` | **0.80~1.20** MC 全 100% | **未发现** |
| `inv_vol_window` | **≥8** | **≤5** MC 17% 🔴 |
| `step_low` | **0.12~0.20** | **0.10** Sharpe 降至 1.201 |
| `step_high` | **0.25~0.45** | **未发现** |
| `rebalance_threshold` | **0%~10%** Sharpe 波动 <0.04 | **0%** 交易量最多但收益相近 |

## 注意事项

- 数据列顺序：`日期,纳指ETF,红利低波ETF,沪深300ETF,黄金ETF,国债ETF`
- 所有因子计算强制使用 `src/factors.py`（ddof=0），禁止自行实现
- 如需多日频分析，考虑日频 DD 比周频高约 0.3~2pp
- 纳指ETF 如出现 QDII 溢价 >2%，需人工判断是否延迟买入
- 策略基于历史回测，不保证未来收益
