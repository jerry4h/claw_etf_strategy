# 虾池ETF轮动策略 — 工程架构设计

> 版本：v1.0
> 日期：2026-06-15
> 设计者：quant-se（架构师）
> Kanban 任务：T1
> 基于：技术方案定稿 v2.3

---

## 一、设计目标

1. **模块化**：清晰的模块边界，每个模块单一职责
2. **可配置**：策略参数通过 YAML 配置，零硬编码
3. **可扩展**：策略变体通过新配置文件即可支持，无需改代码
4. **统一引擎**：所有回测走同一个 `BacktestEngine`，保证数据一致性
5. **可测试**：每个模块独立可测
6. **可复现**：给定配置 + 数据，输出确定

---

## 二、项目结构

```
/home/ubuntu/claw_etf_strategy/
├── config/
│   ├── strategy_v2_3.yaml              # v2.3 基准策略配置（默认）
│   ├── strategy_step_high_50.yaml      # 变体：放宽vol阈值
│   └── strategy_pure_offensive.yaml    # 变体：纯进攻零防御
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # 数据加载 & 预处理
│   ├── factors.py                      # 因子计算
│   ├── strategy.py                     # 策略逻辑（评分、选股、分配、防御）
│   ├── backtest.py                     # 统一回测引擎
│   ├── report.py                       # 绩效报告 & 可视化 & 贡献分析
│   └── utils.py                        # 共享工具函数
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # 共享 fixtures
│   ├── test_data_loader.py
│   ├── test_factors.py
│   ├── test_strategy.py
│   ├── test_backtest.py
│   └── test_report.py
├── data/                               # 数据目录（数据文件移至此）
│   ├── all_etfs_nav_2013_2026_h20269_scaled.csv
│   └── 300etf_pe_percentile_weekly.csv
├── output/                             # 回测输出
├── run_backtest.py                     # CLI 入口：单次回测
├── run_grid_search.py                  # CLI 入口：网格搜索
├── run_ablation.py                     # CLI 入口：消融实验
├── goal.md
├── 虾池ETF轮动策略 —— 技术方案定稿.md
├── ARCHITECTURE.md                     # 本文档
└── kimi_strategy_study/                # 原研究代码（归档，不再更新）
```

---

## 三、模块接口定义

### 3.1 `src/data_loader.py` — 数据加载与预处理

**职责**：加载 ETF 净值数据，清洗，生成周频序列。

```python
# === 常量 ===
ETFS: list[str]           # 全量 ETF 列表
OFFENSIVE: list[str]      # 进攻层 ETF
DEFENSIVE: list[str]      # 防御层 ETF

# === 接口 ===
def load_nav_data(data_path: str | Path) -> pd.DataFrame:
    """
    加载 ETF 日净值数据，执行清洗。

    处理步骤：
    1. 读 CSV → datetime index
    2. 删除全市场休市行（所有 ETF 为 NaN）  # goal.md §3 修复
    3. 单只 ETF 缺失值 ffill
    4. 截断至所有 ETF 都有数据之后

    Returns: DataFrame, index=日期(datetime), columns=ETF名称, values=净值(float)
    """

def calculate_daily_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算日收益率。

    Returns: DataFrame, index=日期, columns=ETF名称, values=日收益率(float)
    """

def resample_weekly(nav_df: pd.DataFrame, anchor: str = "W-MON") -> pd.DataFrame:
    """
    将日净值降采样为周净值（每周一收盘价）。

    Args:
        anchor: pandas 周锚点 ("W-MON", "W-TUE", ...)

    Returns: DataFrame, index=周一日期, columns=ETF名称, values=周收盘净值
    """

def load_pe_percentile(pe_path: str | Path) -> pd.DataFrame:
    """
    加载沪深300 PE-TTM 5年滚动分位数。

    Returns: DataFrame, index=日期, 单列 PE分位数(float, 0~1)
    """
```

**数据流**：CSV → `load_nav_data()` → 日净值 → `resample_weekly()` → 周净值 → 下游所有模块

---

### 3.2 `src/factors.py` — 因子计算

**职责**：从周净值计算动量、波动率、PE 分位数。纯函数，无副作用。

```python
def calculate_momentum(
    weekly_nav: pd.DataFrame,
    window: int = 4
) -> pd.DataFrame:
    """
    4 周动量（v2.3 公式）。

    计算：prod(1 + wrets[i−4:i]) − 1

    Returns: DataFrame, index=日期, columns=ETF, values=动量(float)
    """

def calculate_volatility(
    weekly_nav: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    20 周年化波动率。

    计算：std(wrets[i−20:i]) × √52

    Returns: DataFrame, index=日期, columns=ETF, values=年化波动率(float)
    """

def calculate_pe_percentile(
    pe_df: pd.DataFrame,
    window_years: int = 5
) -> pd.DataFrame:
    """
    沪深300 PE-TTM 滚动分位数。

    ⚠️ 必须 shift(1) 确保无前视偏差：本周调仓只能用上周及之前的分位数。

    Returns: DataFrame, index=日期, 单列 PE分位数(float, 0~1)
    """

def compute_all_factors(
    weekly_nav: pd.DataFrame,
    pe_df: pd.DataFrame,
    config: dict
) -> dict[str, pd.DataFrame]:
    """
    一次计算所有因子，自动 shift(1) 防前视。

    Args:
        config: 策略配置字典（从 YAML 加载），含 mom_window, vol_window 等

    Returns: {
        "momentum":     DataFrame (shifted),
        "volatility":   DataFrame (shifted),
        "pe_percentile": DataFrame (shifted)
    }
    """
```

**数据流**：周净值 + PE数据 → `compute_all_factors()` → {momentum, volatility, pe_percentile} → `strategy.py`

---

### 3.3 `src/strategy.py` — 策略逻辑

**职责**：评分计算、进攻 ETF 选择、防御比例计算、仓位分配。

```python
# === 数据类 ===
@dataclass
class StrategyConfig:
    """从 YAML 加载的策略参数"""
    mom_w: float            # 动量权重
    vol_w: float            # 波动率权重
    top_n: int              # 选几只进攻 ETF
    def_alloc: float        # 基准防御比例
    step_low: float         # vol 三段式下限
    step_high: float        # vol 三段式上限
    max_def: float          # 极限防御比例
    hongli_ratio: float     # 防御层中红利低波占比
    rebalance_threshold: float  # 调仓阈值
    fee_rate: float         # 单边费率
    stop_loss: float        # 止损阈值
    recovery_weeks: int     # 止损恢复观察周数

# === 接口 ===
def load_config(config_path: str | Path) -> StrategyConfig:
    """从 YAML 文件加载策略配置"""

def score_offensive(
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    date: pd.Timestamp,
    config: StrategyConfig
) -> dict[str, float]:
    """
    计算进攻层 ETF 综合得分（v2.3 公式，已移除 val_w）。

    score = mom_w × mom4 − vol_w × vol20

    仅对 OFFENSIVE ETF 计算。

    Returns: {"纳指ETF": 0.12, "沪深300ETF": 0.05, "黄金ETF": 0.08}
    """

def select_top(
    scores: dict[str, float],
    top_n: int
) -> list[str]:
    """
    选择得分最高的 top_n 只进攻 ETF。

    Returns: ["纳指ETF", "黄金ETF"]
    """

def calculate_defense_ratio(
    nasdaq_vol: float,
    config: StrategyConfig
) -> float:
    """
    vol 三段式防御比例计算。

    if nasdaq_vol < step_low:     → def_alloc
    elif nasdaq_vol > step_high:  → max_def
    else:                          → 线性插值

    Returns: 防御比例 (0~1)
    """

def allocate(
    selected: list[str],
    defense_ratio: float,
    config: StrategyConfig
) -> dict[str, float]:
    """
    计算完整仓位分配。

    防御层: 红利低波(defense_ratio × hongli_ratio) + 国债(defense_ratio × (1-hongli_ratio))
    进攻层: selected ETFs 平分 (1-defense_ratio)

    Returns: {"纳指ETF": 0.375, "黄金ETF": 0.375, "红利低波ETF": 0.125, "国债ETF": 0.125}
    """

def check_rebalance(
    current_alloc: dict[str, float],
    new_alloc: dict[str, float],
    threshold: float
) -> bool:
    """检查是否有 ETF 仓位变化超过阈值，决定是否调仓"""

def check_stop_loss(
    current_nav: float,
    peak_nav: float,
    threshold: float
) -> bool:
    """检查是否触发止损"""
```

**数据流**：{factors} + config → `score_offensive()` → `select_top()` → `allocate()` → 仓位字典

---

### 3.4 `src/backtest.py` — 统一回测引擎

**职责**：唯一回测引擎，所有回测（单次、网格搜索、消融实验）都走这里。

```python
@dataclass
class BacktestResult:
    """回测结果"""
    nav_series: pd.DataFrame         # 逐周净值、回撤、仓位
    metrics: dict                    # 年化、回撤、夏普、胜率等
    config: StrategyConfig           # 使用的配置（可复现）

def run_backtest(
    config: StrategyConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    data_path: str | Path | None = None,
    pe_path: str | Path | None = None
) -> BacktestResult:
    """
    统一回测引擎。

    流程：
    1. data_loader.load_nav_data() + load_pe_percentile()
    2. data_loader.resample_weekly()
    3. factors.compute_all_factors()
    4. 逐周循环：
       a. strategy.score_offensive()    → 评分
       b. strategy.select_top()         → 选前 N
       c. strategy.calculate_defense_ratio() → vol 防御
       d. strategy.check_stop_loss()    → 止损检查
       e. strategy.allocate()           → 仓位分配
       f. strategy.check_rebalance()    → 调仓阈值过滤
       g. 计算周收益，扣双边手续费
       h. 记录持仓明细
    5. 汇总绩效指标

    Returns: BacktestResult (nav_series + metrics)
    """

def compute_metrics(nav_series: pd.DataFrame) -> dict:
    """
    从净值序列计算核心指标。

    Returns: {
        "total_return": 4.40,
        "annual_return": 0.1406,
        "max_drawdown": 0.0821,
        "sharpe_ratio": 1.102,      # 标准夏普（扣 2.5% 无风险利率）
        "simple_sharpe": 1.60,      # 简化夏普（仅供参考）
        "calmar_ratio": 1.71,
        "win_rate": 0.583,
        "avg_weekly_return": 0.0027,
        "std_weekly_return": 0.018,
        "total_weeks": 674,
        "defensive_weeks": 135,
        "rebalance_count": 455,
    }
    """

def grid_search(
    param_space: dict[str, list],
    base_config_path: str | Path,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    网格搜索。每个参数组合调用一次 run_backtest()。
    使用 multiprocessing 并行。

    Args:
        param_space: {"mom_w": [0.30, 0.35, 0.40], "step_high": [0.35, 0.40, 0.45]}
        base_config_path: 基准配置文件

    Returns: DataFrame, 每行一组参数 + 绩效指标
    """

# === 消融实验 ===
def run_ablation(config_path: str | Path) -> dict:
    """
    逐特性移除消融实验。

    依次移除：动量选择、vol防御、红利低波防御、调仓阈值、止损兜底。
    每次移除后运行完整回测，记录绩效变化。

    Returns: {"完整策略": {...}, "去掉动量": {...}, ...}
    """
```

**数据流**：Config → `run_backtest()` → BacktestResult

---

### 3.5 `src/report.py` — 绩效报告 & 可视化

**职责**：从 BacktestResult 生成指标、图表和 Markdown 报告。

```python
def generate_metrics_table(result: BacktestResult) -> str:
    """生成核心指标 Markdown 表格"""

def generate_annual_breakdown(result: BacktestResult) -> str:
    """生成年度收益分解 Markdown 表格"""

def generate_drawdown_analysis(result: BacktestResult) -> str:
    """生成回撤来源分析 Markdown 表格"""

def generate_contribution_analysis(
    result: BacktestResult,
    nav_df: pd.DataFrame
) -> str:
    """
    生成 ETF 贡献分析：每年各 ETF 持有周数、平均权重、收益贡献。
    """

# === 图表（base64 内嵌，适合 Feishu/Markdown 渲染）===
def chart_nav_curve(result: BacktestResult) -> str:
    """净值曲线 + 基准对比，返回 base64 PNG"""

def chart_drawdown(result: BacktestResult) -> str:
    """回撤曲线，返回 base64 PNG"""

def chart_annual_returns(result: BacktestResult) -> str:
    """年度收益柱状图，返回 base64 PNG"""

def chart_monthly_heatmap(result: BacktestResult) -> str:
    """月度收益热力图，返回 base64 PNG"""

def chart_risk_dashboard(result: BacktestResult) -> str:
    """风险仪表盘（VaR、CVaR、连续亏损、回撤持续），返回 base64 PNG"""

# === 完整报告 ===
def generate_full_report(
    result: BacktestResult,
    output_path: str | Path,
    include_charts: bool = True
) -> str:
    """
    生成完整 Markdown 报告，写入文件。

    包含：指标总览、年度分解、回撤分析、贡献分析、图表。
    """
```

**数据流**：BacktestResult → `generate_full_report()` → Markdown 报告

---

### 3.6 `src/utils.py` — 共享工具

```python
def annualize_return(total_return: float, n_weeks: int) -> float:
    """年化收益率: (1 + r) ^ (52 / n) − 1"""

def compute_max_drawdown(nav: pd.Series) -> float:
    """最大回撤"""

def compute_sharpe(returns: pd.Series, risk_free: float = 0.025) -> float:
    """标准夏普比率（扣无风险利率）"""

def compute_calmar(annual_return: float, max_drawdown: float) -> float:
    """卡尔马比率"""
```

---

## 四、配置文件格式

### `config/strategy_v2_3.yaml`（基准配置）

```yaml
# 虾池ETF轮动策略 v2.3 — 基准配置
# 对应技术方案定稿 v2.3 的核心参数

strategy:
  name: "虾池ETF轮动 v2.3"
  version: "2.3"

scoring:
  mom_w: 0.35        # 4周动量权重
  vol_w: 0.30        # 20周波动率权重
  # val_w 已移除（v2.3消融验证不影响排序）

selection:
  top_n: 2           # 从进攻层选2只ETF

factors:
  mom_window: 4      # 动量窗口（周）
  vol_window: 20     # 波动率窗口（周）
  pe_window_years: 5 # PE分位数窗口（年）

defense:
  def_alloc: 0.25    # 基准防御比例
  step_low: 0.20     # vol三段式下限
  step_high: 0.35    # vol三段式上限
  max_def: 0.95      # 极限防御比例
  hongli_ratio: 0.50 # 防御层中红利低波占比

rebalance:
  threshold: 0.07    # 调仓阈值（单只仓位变化≥7%才调）
  fee_rate: 0.00005  # 单边费率（双边×2 = 0.01%）
  anchor: "W-MON"    # 调仓日锚点

risk_control:
  stop_loss: 0.08    # 止损阈值（净值从峰值下跌8%）
  recovery_weeks: 4  # 止损后恢复观察周数

data:
  nav_path: "data/all_etfs_nav_2013_2026_h20269_scaled.csv"
  pe_path: "data/300etf_pe_percentile_weekly.csv"
  start_date: "2013-01-01"
  end_date: "2026-05-01"

reporting:
  risk_free_rate: 0.025  # 无风险利率（用于标准夏普）
```

### `config/strategy_step_high_50.yaml`（变体示例）

```yaml
# 变体：放宽vol防御阈值 → 提高收益
# 继承 v2.3 全部参数，仅覆盖 defense

strategy:
  name: "虾池ETF轮动 v2.3 — 放宽vol防御"
  version: "2.3"
  base: "strategy_v2_3"  # 继承基准配置

defense:
  step_high: 0.50    # 从 0.35 放宽到 0.50
```

---

## 五、数据流图

```
                          ┌──────────────────────┐
                          │   YAML 配置文件       │
                          │  (strategy_v2_3.yaml) │
                          └──────────┬───────────┘
                                     │ StrategyConfig
                                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  data/       │───▶│ data_loader  │───▶│   factors    │
│  CSV 文件    │    │ .py          │    │   .py        │
└──────────────┘    │ load_nav()   │    │ compute_all  │
                    │ resample_w() │    │ _factors()   │
                    └──────┬───────┘    └──────┬───────┘
                           │ 周净值             │ {momentum,
                           │                    │  volatility,
                           │                    │  pe_percentile}
                           │                    │
                           ▼                    ▼
                    ┌──────────────────────────────────┐
                    │         strategy.py              │
                    │  score → select → defense →      │
                    │  allocate → check_rebalance      │
                    └──────────────┬───────────────────┘
                                   │ 每周仓位字典
                                   ▼
                    ┌──────────────────────────────────┐
                    │         backtest.py              │
                    │  run_backtest()  ← 统一引擎      │
                    │  逐周循环 + 绩效汇总              │
                    └──────────────┬───────────────────┘
                                   │ BacktestResult
                                   ▼
                    ┌──────────────────────────────────┐
                    │          report.py               │
                    │  metrics / charts / markdown     │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │          output/                 │
                    │  report_*.md / nav_history.csv   │
                    └──────────────────────────────────┘
```

---

## 六、关键技术决策

### 6.1 数据清洗：先删全 NaN 行，再 ffill 个别缺失

**来源**：`goal.md` §3 数据清洗与交易日历处理

```python
# 正确顺序：
# 1. 先删所有 ETF 都为 NaN 的行（周末 + A 股节假日）
all_nan = df[ETFS].isna().all(axis=1)
df = df[~all_nan].copy()

# 2. 再对个别缺失 ffill（如 QDII 暂停申赎）
for col in ETFS:
    df[col] = df[col].ffill()
```

⚠️ **禁止**：先 ffill 再删 NaN——那样会把周末填充成幽灵交易日，导致 vol 低估 2-3%。

### 6.2 shift(1) 防前视偏差

所有因子计算后必须 `.shift(1)`，确保第 t 周调仓只用第 t-1 周及之前的信息。

### 6.3 年化乘数 √252

量化业界统一标准。A 股实际交易日 ~244 天，但 √244 与 √252 差异仅 1.6%，对策略排序无影响。

### 6.4 PE 分位数统一使用

v2.3 消融实验证实：val_w（估值因子权重）完全不影响进攻层 ETF 排序——三个进攻 ETF 共用同一 PE 分位数，评分中加的是常数。**已从评分公式移除**。PE 分位数保留在因子层，供未来版本（如防御比例、趋势过滤器）使用。

### 6.5 双边手续费

买入 + 卖出各收 0.005%，turnover 计算后 × 2。手续费在每周收益中扣减。

### 6.6 vol 防御用纳指代表全部进攻层

技术方案 §8.2 已知缺陷：防御触发只看纳指 vol。这是有意的简化——纳指 vol 是进攻层波动的主要驱动。未来版本可考虑加权平均。

---

## 七、迁移计划（从 kimi_strategy_study/ 到新架构）

### Phase 1：基础设施搭建

- [ ] 创建目录结构（`src/`, `config/`, `tests/`, `data/`, `output/`）
- [ ] 移动数据文件到 `data/`
- [ ] 创建 `config/strategy_v2_3.yaml`
- [ ] 创建 `src/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- [ ] 创建 `src/utils.py`

### Phase 2：核心模块迁移

按依赖顺序逐模块迁移：

1. **`src/data_loader.py`**（无依赖）
   - 从 `kimi_strategy_study/data_loader.py` 迁移
   - 添加 `load_pe_percentile()`
   - 确保 `load_nav_data()` 已包含 goal.md 的 NaN 修复

2. **`src/factors.py`**（依赖 data_loader）
   - 从 `kimi_strategy_study/factors.py` 迁移
   - 4 周动量确认使用 `prod(1+wrets) - 1`（原代码用 `rolling.sum`，需对齐 v2.3 公式）
   - 添加 `compute_all_factors()` 统一入口（含自动 shift）

3. **`src/strategy.py`**（依赖 factors）
   - 从 `kimi_strategy_study/strategy.py` 迁移
   - 添加 `StrategyConfig` dataclass + `load_config()`
   - 评分公式移除 val_w
   - vol 三段式防御从配置读取参数

4. **`src/backtest.py`**（依赖 strategy、data_loader、factors）
   - 从 `kimi_strategy_study/backtest.py` 迁移
   - 改为接收 `StrategyConfig` 而非散列参数
   - 添加 `BacktestResult` dataclass
   - 保留 `grid_search()` 和 `run_ablation()`

5. **`src/report.py`**（依赖 backtest）
   - 从 `kimi_strategy_study/analysis.py` 拆分迁移
   - 拆分为多个独立函数，每个一个职责

### Phase 3：CLI 入口

- [ ] `run_backtest.py`：加载配置 → 运行回测 → 输出报告
- [ ] `run_grid_search.py`：参数空间定义 → 网格搜索 → CSV 结果
- [ ] `run_ablation.py`：消融实验 → 表格输出

### Phase 4：测试

- [ ] `test_data_loader.py`：验证 NaN 处理、数据形状
- [ ] `test_factors.py`：验证动量/波动率计算，验证 shift 防前视
- [ ] `test_strategy.py`：验证评分、选股、vol 三段式边界
- [ ] `test_backtest.py`：验证完整回测输出与 v2.3 基准一致
- [ ] `test_report.py`：验证报告生成、指标计算

### Phase 5：验证

- [ ] 新架构跑 v2.3 基准 → 结果必须与文档一致（14.06% / 8.21% / 1.102）
- [ ] 网格搜索 → 确认最优参数不变
- [ ] 消融实验 → 确认结论不变

---

## 八、关键差异：新架构 vs 旧代码

| 项目 | `kimi_strategy_study/`（旧） | 新架构 `src/` |
|------|---------------------------|-------------|
| 参数管理 | 硬编码在代码各处 | YAML 配置，一处修改全局生效 |
| 评分公式 | `0.4*mom - 0.4*vol + 0.2*val` | `0.35*mom - 0.30*vol`（v2.3 基准） |
| val_w | 存在，有硬编码值 | 已移除（v2.3 消融验证不影响排序） |
| 动量计算 | `rolling(20).sum()`（日频） | `prod(1+wrets[i−4:i])−1`（周频，v2.3 公式） |
| 防御逻辑 | 固定 55% 防御 + 止损额外逻辑 | vol 三段式（25%~95%），配置驱动 |
| 回测引擎 | 参数散列式，多版本并存 | 统一 `run_backtest(config)` |
| 可视化 | 1200 行单文件 | 拆分为独立函数 |
| 测试 | 无 | 独立可测 |

---

## 九、下一步：quant-coder 实现指南

1. **先读 `goal.md` 和 技术方案定稿**，理解业务逻辑
2. **从 `src/utils.py` 开始**（最简单，无依赖）
3. **然后是 `src/data_loader.py`**（确认 NaN 修复正确）
4. **依次 `factors.py` → `strategy.py` → `backtest.py` → `report.py`**
5. **每完成一个模块，写对应测试**（TDD 推荐）
6. **写完 `backtest.py` 后立即验证**：v2.3 配置 → 输出必须 = 14.06%/8.21%/1.102
7. **差异 > 0.05% 即视为不一致**，需排查

验收标准：
```bash
# 基准回测
python run_backtest.py --config config/strategy_v2_3.yaml
# 预期：年化 14.06%, 回撤 8.21%, 夏普 1.102

# 消融实验
python run_ablation.py --config config/strategy_v2_3.yaml
# 预期：与 §10 消融实验表一致

# 测试
python -m pytest tests/ -v
# 预期：全部通过
```

---

## 附录 A：模块依赖图

```
utils.py          ← 零依赖
    ↓
data_loader.py    ← utils.py
    ↓
factors.py        ← data_loader.py, utils.py
    ↓
strategy.py       ← factors.py, utils.py
    ↓
backtest.py       ← strategy.py, data_loader.py, factors.py, utils.py
    ↓
report.py         ← backtest.py, utils.py
    ↓
run_backtest.py   ← backtest.py, report.py, strategy.py
run_grid_search.py ← backtest.py, strategy.py
run_ablation.py   ← backtest.py, strategy.py
```

---

*本文档由 quant-se（架构师）编写，基于技术方案定稿 v2.3。交付给 quant-pm 进行下一步任务分派。*
