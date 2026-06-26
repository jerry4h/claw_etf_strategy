# 鲁棒性评估 v2 — 增强版 MC 规范

> 版本：v2.0
> 日期：2026-06-17
> 设计者：quant-se
> 改动范围：`src/robustness.py` + `scripts/run_robustness.py`

---

## 一、改动清单

### 1. MC 参数列表扩展

从 5 个参数扩展为 7 个：

```python
# 旧
MC_PARAMS = ['mom_w', 'vol_w', 'def_alloc', 'step_high', 'step_low']

# 新
MC_PARAMS = [
    'mom_w',                # 动量权重 (0.35)
    'vol_w',                # 波动率权重 (0.30)
    'def_alloc',            # 基准防御比例 (0.25)
    'step_high',            # vol三段式上限 (0.35)
    'step_low',             # vol三段式下限 (0.20)
    'momentum_window',      # D4动量窗口 (8, 整数)
    'momentum_threshold',   # D4动量阈值 (-0.075)
]
```

`momentum_window` 是整数，扰动后需 `round()` 到 int。范围 7~9（±1）。

### 2. 扰动幅度变更

```python
# 旧
PERTURBATION = 0.10  # ±10%

# 新
PERTURBATION = 0.15  # ±15%
```

### 3. 新增：OAT 多级敏感度分析

除全参数同时扰动的 MC 外，增加 One-At-a-Time 多级敏感度：

```
对每个参数，在 7 个扰动级别下单独测试：
  -15%, -10%, -5%, 0%, +5%, +10%, +15%

其他参数保持基线值不变。
```

输出：每个参数一条敏感度曲线（Sharpe / Return / MaxDD vs 扰动级别）。

**实现**：`run_mc_survival_test()` 新增 `mode` 参数：
- `mode='mc'`（默认）：全参数同时随机扰动，n_runs 次
- `mode='oat'`：OAT 多级敏感度，7×7=49 次

### 4. MC 运行次数

```
n_runs = 400  # (原 100)
```

理由：p≈0.5 时 95% CI 半宽 ≈ ±5%（n=100 时为 ±10%，不可接受）。

### 5. MC 生存率判定标准收紧

```python
# 旧：仅 Sharpe > 0
survival = (sharpe > 0)

# 新：年化 > 10% AND DD < 15%（对应 goal.md 原始约束）
survival = (annual_return > 0.10 and max_drawdown < 0.15)
```

### 6. 门控阈值重新校准

| 指标 | 🟢 | 🟡 | 🔴 |
|------|:--:|:--:|:--:|
| MC 生存率（新标准） | > 80% | 50-80% | < 50% |

---

## 二、接口变更

### `src/robustness.py`

```python
# 常量
MC_PARAMS = ['mom_w', 'vol_w', 'def_alloc', 'step_high', 'step_low',
             'momentum_window', 'momentum_threshold']
PERTURBATION = 0.15  # ±15%

# 新增函数
def run_oat_sensitivity(
    config_path: str,
    perturbation_levels: list[float] = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15],
) -> dict[str, list[dict]]:
    """OAT多级敏感度。返回 {param_name: [{level, sharpe, ret, dd}, ...], ...}"""

# 修改函数签名
def run_mc_survival_test(
    config_path: str,
    n_runs: int = 400,       # 原 100
    perturbation: float = 0.15,  # 原 0.10
    n_jobs: int = -1,
    mode: str = 'mc',        # 新增: 'mc' | 'oat'
) -> tuple[float, list[dict]]:

# 修改判定
# 第 281 行: sharpe > 0 → annual_return > 0.10 AND max_drawdown < 0.15
```

### `scripts/run_robustness.py`

```python
# 新增参数
parser.add_argument('--n-mc', type=int, default=400)   # 原 100
parser.add_argument('--perturbation', type=float, default=0.15)  # 新增
parser.add_argument('--oat', action='store_true')      # 新增: 启用 OAT
```

### `RobustnessResult` dataclass

```python
@dataclass
class RobustnessResult:
    dsr: float
    mc_survival_rate: float
    benchmark_relative_win_rate: float
    oat_sensitivity: dict | None = None     # 新增
    strategy_config: str
    strategy_metrics: dict
    details: dict
```

---

## 三、报告变更

新增 OAT 敏感度章节：

```markdown
## OAT 多级敏感度分析

### mom_w

| 扰动 | 值 | Sharpe | 年化收益 | 最大回撤 |
|------|:--:|:------:|:-------:|:-------:|
| -15% | 0.2975 | X.XXX | XX.X% | XX.X% |
| -10% | 0.3150 | X.XXX | XX.X% | XX.X% |
| ... | ... | ... | ... | ... |
| +15% | 0.4025 | X.XXX | XX.X% | XX.X% |

（7 个参数各一张表）
```

---

## 四、CLI 用法

```bash
# 完整评估（MC 400次 + OAT 49次/策略 + WF 9窗口）
python scripts/run_robustness.py \
    --configs config/strategy_v2_3_cap040.yaml,config/strategy_v2_3_cap040_D4_tuned.yaml \
    --labels "v2.3 基线","v2.3+cap040+D4 tuned" \
    --output output/robustness_v2/ \
    --n-mc 400 \
    --perturbation 0.15 \
    --oat \
    --n-wf 9

# 预估耗时：~25 分钟（400+49+49 ≈ 500 次回测 × 2 策略，多进程并行）
```

---

## 五、验收标准

1. MC 参数包含 D4 的 `momentum_window` 和 `momentum_threshold`
2. 扰动幅度 ±15%
3. OAT 输出 7 参数 × 7 级别的完整矩阵
4. MC 判定标准为 `年化>10% AND DD<15%`
5. MC n_runs=400，报告显示 95% CI 半宽
6. MC 门控阈值按新标准（🟢>80%, 🟡50-80%, 🔴<50%）
7. D4 tuned 的 OAT 显示 `momentum_threshold` 在 -0.0675 ~ -0.0825 范围内的行为

---

*本文档由 quant-se 设计，作为 `src/robustness.py` v2 的实现规范。*