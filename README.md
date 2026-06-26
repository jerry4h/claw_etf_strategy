# 虾池ETF轮动策略 (Shrimp Pool ETF Rotation)

> 周频调仓、多资产、动量驱动的 ETF 组合策略 — 年化 14.3% / 最大回撤 8.8% / 夏普 1.21

---

## 一、策略概述

基于 5 只 ETF（纳指、红利低波、沪深300、黄金、国债）的周频动量轮动策略。三层决策架构，全部连续无门控：

| 层次 | 决策 | 方法 | 参数 |
|------|------|------|------|
| **Layer 1** | 买什么 | `score = mom4 − 0.857×vol20`，选 top 2 | 零超参 |
| **Layer 2** | 买多少 | inv-vol8 风险预算权重 | window=8 |
| **Layer 3** | 防多少 | 纳指 vol20 三段式线性插值 → 防御比例 25%~95% | step_low=20%, step_high=35% |

**核心特点**：零门控、零阈值、纯连续 — 所有决策都是线性或 sigmoid 连续函数，无参数悬崖。

---

## 二、核心指标（v3.0 inv-vol8）

| 指标 | 值 | 说明 |
|------|:---:|------|
| 年化收益 | **14.28%** | 2013-2026, 13年回测 |
| 最大回撤 | **8.79%** | 全区间 |
| 标准夏普 | **1.211** | 扣 2.5% 无风险利率 |
| DSR | 1.000 | 真 alpha（Bailey & López de Prado 2014） |
| SPS 最差起点 | >5% | 114 个逐月起点投 3 年，零亏损 |
| MC 生存率 | 58.5% | ±15% 参数扰动后 年化>10% AND DD<15% |
| WF 相对胜率 | 77.8% | 9 窗口 vs 等权 5 ETF 基准 |

---

## 三、项目结构

```
claw_etf_strategy/
├── src/                         # 核心引擎
│   ├── backtest.py              #   统一回测引擎
│   ├── strategy.py              #   策略配置 + 评分/分配/D4过滤
│   ├── data_loader.py           #   数据加载 (日频清洗 + 周频降采样)
│   ├── factors.py               #   因子计算 (mom4/vol20/PE分位)
│   ├── report.py                #   绩效报告 + 可视化
│   ├── robustness.py            #   鲁棒性评估 (DSR/MC/WF/PSS/SPS/OAT)
│   └── utils.py                 #   共享工具函数
├── config/                      # 策略 YAML 配置
│   ├── strategy_v2_3_cap040.yaml            # v2.3 基线
│   ├── strategy_v2_3_cap040_D4_tuned.yaml   # v2.3 + D4 门控 (历史最佳候选)
│   ├── strategy_v3_0_invvol.yaml            # v3.0 inv-vol8 (当前推荐)
│   └── ...                                  # 历史实验变体
├── scripts/                     # CLI 入口
│   ├── run_backtest.py          #   单策略回测
│   ├── run_grid_search.py       #   网格搜索
│   ├── run_ablation.py          #   消融实验
│   ├── rebalance_live.py        #   实时调仓计算 ★
│   └── robustness_evaluation.py #   鲁棒性评估
├── data/                        # 数据文件
│   ├── all_etfs_nav_2013_20260622_scaled.csv  # 主数据 (2013-2026, 678周)
│   └── 300etf_pe_percentile_weekly.csv        # PE 分位数据
├── output/                      # 输出报告
│   ├── robustness_v5b/          #   最新鲁棒性评估
│   └── ...                      #   历史评估报告
├── docs/                        # 设计文档
│   ├── ROBUSTNESS_3_METRICS.md  #   三指标定义
│   └── ROBUSTNESS_IMPL_SPEC.md  #   实现规范
├── 虾池ETF轮动策略 —— 技术方案定稿.md   # 策略技术方案 v2.3
├── goal.md                      # 项目目标 + 数据清洗说明
├── ARCHITECTURE.md              # 工程架构设计
└── README.md                    # 本文档
```

---

## 四、回测说明

### 输入格式

CSV 文件, 每周一行, 列名: `日期,纳指ETF,红利低波ETF,沪深300ETF,黄金ETF,国债ETF`

```
日期,纳指ETF,红利低波ETF,沪深300ETF,黄金ETF,国债ETF
2013-05-20,0.1993003,0.3145183869016145,2.0916279862404,2.626,97.355602616144
...
```

日期为周一, 净值用周五收盘价填充。数据为前复权/scaled 净值 (百分比收益不变)。

### 核心引擎

```python
from src.backtest import run_backtest
from src.strategy import load_config

cfg = load_config('config/strategy_v3_0_invvol.yaml')
result = run_backtest(cfg)
# result.metrics → {sharpe_ratio, annual_return, max_drawdown, ...}
# result.nav_series → 逐周净值、回撤、仓位
```

### 评估体系（5 指标）

| 指标 | 问什么 | Python 函数 |
|------|--------|------------|
| DSR | 真 alpha 还是运气？| `robustness.compute_dsr()` |
| PSS/MC | 参数飘了还能赚？| `robustness.run_mc_survival_test()` |
| WF | 各窗口跑赢等权基准？| `robustness.compute_benchmark_relative_win_rate()` |
| SPS | 最倒霉入场还能赚？| `robustness.compute_starting_point_sensitivity()` |
| OAT | 每个参数悬崖在哪？| `robustness.run_oat_sensitivity()` |

### 运行命令

```bash
# 基准回测
python scripts/run_backtest.py --config config/strategy_v3_0_invvol.yaml

# 消融实验
python scripts/run_ablation.py --config config/strategy_v3_0_invvol.yaml

# 网格搜索
python scripts/run_grid_search.py --param-space '{"inv_vol_window":[4,6,8,10,12]}'

# 鲁棒性评估
python scripts/robustness_evaluation.py --config config/strategy_v3_0_invvol.yaml
```

---

## 五、实时调仓

### 使用方法

```bash
python scripts/rebalance_live.py                      # 最新数据 → 下周一调仓
python scripts/rebalance_live.py --verify             # 全量回测 vs 引擎验证
python scripts/rebalance_live.py --week 2026-06-22    # 查看特定周
```

### 需要准备

1. **数据文件**: CSV 末尾追加最新一周数据 (手动填入 5 只 ETF 周五净值)
2. **运行时机**: 周日晚或周一早上 — 用最新数据跑脚本
3. **执行调仓**: 脚本输出即下周持仓比例, 按比例下单

### 输出示例

```
══════════════════════════════════════════════════════════════════════
 虾池ETF轮动 v3.0  实时调仓计算
══════════════════════════════════════════════════════════════════════
 基准日: 2026-06-22  (本周净值)
 调仓日: 下周一

 Layer 1 (买什么): score = mom4 − 0.857×vol20
  ETF            mom4    vol20    score   rank
  ──────────────────────────────────────────
  沪深300ETF      3.12%    19.2%  -0.1337    TOP
  纳指ETF         0.85%    33.9%  -0.2816    TOP
  黄金ETF        -8.37%    43.5%  -0.4565

 Layer 3 (防多少): 纳指vol20 =  33.9% → 线性插值 → 防御  90%

 ── 本周计算结果(目标持仓) ──
  纳指ETF        3.1%  ≈   15,405元
  红利低波ETF    44.8%  ≈  224,086元
  沪深300ETF     7.3%  ≈   36,424元
  国债ETF       44.8%  ≈  224,086元
 ══════════════════════════════════════════════════════════════════════
 ✅ 下周一按此比例调仓
```

### 注意事项

| 要点 | 说明 |
|------|------|
| **数据用周五净值** | 回测使用 W-MON 锚点, 实际操作周五收盘价代替周一开盘价 |
| **CSV 末尾追加** | 每周手动加一行新数据, 不需要 Tushare 等数据源 |
| **最少 20 周数据** | vol20 窗口需要 20 周历史 |
| **调仓阈值 7%** | 若本周 vs 上周最大仓位变化 <7%, 不调仓 (减少摩擦成本) |
| **防御比例** | 纳指 vol20 <20%→基准25%, 20~35%→线性插值, >35%→极限95% |
| **权重上限** | 单个 ETF 不超过 40%, 防过度集中 |

---

## 六、关键设计决策

| 决策 | 原因 |
|------|------|
| 零门控 | 消除 OAT 参数悬崖, 提高鲁棒性 |
| inv-vol8 权重 | 单一因子决定买多少, 天然连续 |
| score = mom4 − 0.857×vol20 | 固定 mom_w=1 → 只有 1 个参数 (vol_w), 减少冗余测试 |
| 周五净值 → 周一调仓 | 日频 DD 与周频差异 0.1pp, 可忽略 |
| 固定 top_n=2 | 消融实验验证唯一最优解 (top_n=1 Sharpe 暴跌, top_n=3 灾难) |
| cap 0.40 | 防单 ETF 过度集中 |

---

## 七、历史演进

| 版本 | 策略 | Sharpe | 年化 | DD | 说明 |
|------|------|:-----:|:---:|:--:|------|
| v2.3 基线 | scoring + vol3tier | 1.102 | 14.1% | 7.4% | 原始基准 |
| v2.3+cap040 | +单标的上限 0.40 | 1.102 | 14.1% | 7.4% | 加风控 |
| v2.3+cap040+D4 | +8周动量门控 | 1.216 | 15.7% | 7.6% | 峰值 Sharpe, 有悬崖 |
| **v3.0 inv-vol8** | **scoring + invvol + vol3tier** | **1.211** | **14.3%** | **8.8%** | ★ 当前推荐: 零悬崖 |
| v3.0 inv-vol (新数据) | 扩展至 2026-06-22 | **1.211** | **14.3%** | **8.8%** | 新 2 月确认稳定 |

---

## 八、相关文档

- `goal.md` — 项目目标与基础要求
- `虾池ETF轮动策略 —— 技术方案定稿.md` — v2.3 技术方案 (含消融实验)
- `ARCHITECTURE.md` — 工程架构设计
- `docs/ROBUSTNESS_3_METRICS.md` — 鲁棒性三指标定义
- `docs/ROBUSTNESS_IMPL_SPEC.md` — 鲁棒性评估实现规范

---

*策略设计: quant-se | 工程实现: quant-coder | 测试验证: quant-tester | 项目管理: quant-pm*

*策略风险提示: 过往回测表现不代表未来收益。实盘需考虑滑点、冲击成本、QDII 溢价等因素。建议初期仓位 50% 验证 3 个月后调整。*