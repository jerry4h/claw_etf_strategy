# 鲁棒性评估模块 — 实现规范

> 版本：v1.0
> 设计者：quant-se
> 目标：实现三指标简化版鲁棒性评估，替换旧 5 项二元门控

---

## 一、模块设计

### 文件位置
```
src/robustness.py          # 新增：鲁棒性评估核心模块
scripts/run_robustness.py  # 新增：CLI 入口
```

### 模块接口

```python
# src/robustness.py

@dataclass
class RobustnessResult:
    """鲁棒性评估结果"""
    dsr: float                    # Deflated Sharpe Ratio
    mc_survival_rate: float       # MC 生存率
    benchmark_relative_win_rate: float  # 基准相对胜率
    strategy_config: str          # 策略配置名
    strategy_metrics: dict        # 基准回测指标 {annual_return, max_drawdown, sharpe}
    details: dict                 # 详细数据 {mc_runs, wf_windows, ...}


def compute_dsr(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float
) -> float:
    """
    计算 Deflated Sharpe Ratio。

    Args:
        sharpe: 观测到的年化 Sharpe
        n_trials: 尝试的变体数（如 2，即 v2.3 基线 + D4 tuned 两个）
        n_obs: 周收益观测数
        skew: 周收益偏度
        kurtosis: 周收益峰度

    Returns: DSR 概率 (0~1)
    """


def run_mc_survival_test(
    config_path: str,
    n_runs: int = 100,
    perturbation: float = 0.10
) -> tuple[float, list[dict]]:
    """
    Monte Carlo 参数扰动生存率测试。

    对 mom_w, vol_w, def_alloc, step_high, step_low 等核心参数
    同时 ±10% 随机扰动，运行 N 次回测。

    Returns:
        survival_rate: Sharpe > 0 的比例
        mc_details: 每次运行的详细结果
    """


def compute_benchmark_relative_win_rate(
    config_path: str,
    n_windows: int = 9
) -> tuple[float, list[dict]]:
    """
    Walk-Forward 基准相对胜率。

    滚动窗口（1 年窗口，半年间隔），每窗口计算：
      策略 Sharpe - 等权 ETF 基准 Sharpe

    等权基准：5 ETF 各 20%，周频再平衡。

    Returns:
        win_rate: 策略Sharpe > 基准Sharpe 的窗口比例
        wf_details: 每窗口的详细结果
    """


def evaluate_robustness(
    config_path: str,
    n_mc: int = 100,
    n_wf_windows: int = 9
) -> RobustnessResult:
    """
    完整鲁棒性评估。

    1. 运行基准回测 → 获取 Sharpe, skew, kurtosis
    2. 计算 DSR
    3. 运行 MC 生存率测试
    4. 运行基准相对胜率 Walk-Forward
    5. 汇总为 RobustnessResult
    """


def generate_robustness_report(
    results: list[RobustnessResult],
    output_dir: str
) -> str:
    """生成 Markdown 鲁棒性评估报告"""
```

---

## 二、CLI 入口

```bash
# 单策略评估
python scripts/run_robustness.py \
    --config config/strategy_v2_3_cap040.yaml \
    --output output/robustness_v23_baseline/

# 双策略对比评估（本次任务目标）
python scripts/run_robustness.py \
    --configs config/strategy_v2_3_cap040.yaml,config/strategy_v2_3_cap040_D4_tuned.yaml \
    --labels "v2.3 基线","v2.3+cap040+D4 tuned" \
    --output output/robustness_comparison/ \
    --n-mc 100 \
    --n-wf 9
```

---

## 三、输出格式

### 报告结构
```
output/robustness_comparison/
├── ROBUSTNESS_COMPARISON_REPORT.md    # 主报告
├── robustness_results.json            # 结构化结果
├── v23_baseline/
│   ├── nav_history.csv
│   └── mc_details.csv
└── d4_tuned/
    ├── nav_history.csv
    └── mc_details.csv
```

### 主报告模板

```markdown
# 鲁棒性对比评估报告

## 策略对比

| 指标 | v2.3 基线 | v2.3+cap040+D4 tuned | 优胜 |
|------|:--------:|:-------------------:|:----:|
| 年化收益 | 14.11% | 15.65% | D4 tuned |
| 最大回撤 | 7.42% | 7.58% | 基线 |
| 夏普比率 | 1.102 | 1.216 | D4 tuned |

## 鲁棒性三指标

| 指标 | v2.3 基线 | v2.3+cap040+D4 tuned | 门控 |
|------|:--------:|:-------------------:|:----:|
| ① DSR | X.XX | X.XX | >0.95 绿 / >0.85 黄 |
| ② MC 生存率 | XX% | XX% | >90% 绿 / >80% 黄 |
| ③ 基准相对胜率 | XX% | XX% | >80% 绿 / >60% 黄 |

## 综合判定

| 策略 | 综合评级 | 建议 |
|------|:------:|------|
| v2.3 基线 | 🟢/🟡/🔴 | ... |
| v2.3+cap040+D4 tuned | 🟢/🟡/🔴 | ... |

## 详细数据
（MC 扰动分布图、WF 窗口明细表等）
```

---

## 四、实现要点

### DSR 计算（关键）

```python
def compute_dsr(sharpe, n_trials, n_obs, skew, kurtosis):
    """
    Implementation per Bailey & López de Prado (2014).

    E[max(SR_N)] ≈ √2·ln(N) · (1 − γ·SR̂ + (γ²−1)/4 · SR̂²)
    其中 γ ≈ 0.5772 (Euler-Mascheroni)

    SE(SR̂) ≈ √((1 + 0.5·SR̂² − γ₃·SR̂ + (γ₄−3)/4 · SR̂²) / n_obs)
    其中 γ₃ = skew, γ₄ = kurtosis
    """
    import numpy as np
    from scipy.stats import norm

    euler = 0.5772156649

    # Expected max SR under null (N independent trials)
    e_max_sr = np.sqrt(2 * np.log(n_trials)) * (
        1 - euler * sharpe + (euler**2 - 1) / 4 * sharpe**2
    )

    # Standard error of SR
    se_sr = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe +
                     (kurtosis - 3) / 4 * sharpe**2) / n_obs)

    # DSR = P[SR > E[max(SR_N)]]
    dsr = 1 - norm.cdf((sharpe - e_max_sr) / se_sr)
    return dsr
```

### MC 扰动参数范围

```python
PERTURB_PARAMS = {
    'mom_w': 0.10,      # ±10%
    'vol_w': 0.10,
    'def_alloc': 0.10,
    'step_high': 0.10,
    'step_low': 0.10,
    'rebalance_threshold': 0.50,  # 特殊：±50% 相对扰动（0.05~0.11）
}
```

### MC 必须并行

100 次回测串行太慢，使用 `multiprocessing.Pool`（现有 `backtest.py` 已有 grid_search 的多进程模式可参考）。

---

## 五、验收标准

1. `python scripts/run_robustness.py --configs ...` 无报错运行
2. 输出完整 Markdown 报告，包含三指标对比表
3. DSR 值在合理范围（0~1）
4. MC 生存率在合理范围（0~100%）
5. 基准相对胜率在合理范围（0~100%）
6. v2.3 基线结果与旧 `robustness_evaluation.py` 的 WF 数据可比对

---

*本文档由 quant-se 设计。实现时参考 `docs/ROBUSTNESS_3_METRICS.md` 了解三指标含义。*
