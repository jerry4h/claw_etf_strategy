# 鲁棒性评估 v3 — PBO + 起点敏感性

> 版本：v3.0
> 日期：2026-06-17
> 设计者：quant-se
> 在 v2 基础上新增两个指标

---

## 一、新增指标定义

### ④ PBO（Probability of Backtest Overfitting）— "回测结果是不是过拟合？"

**来源**：Bailey, Borwein, López de Prado & Zhu (2017). *Journal of Computational Finance*

**核心思想**：把回测区间切成 N 个子段，生成大量 IS/OOS 组合对。统计：在 IS 中表现最好的策略，在 OOS 中排第几？如果 IS 最好 ↔ OOS 最差，就是过拟合。

```
CSCV 算法:
1. 将全区间周收益切成 N=16 等长子段 S₁...S₁₆
2. 生成所有 C(16, 8) = 12870 个 IS/OOS 组合对
   - IS: 随机选 8 个子段拼接
   - OOS: 剩余 8 个子段拼接（互补集）
3. 每个组合对：
   - IS 上运行策略 → 得到 IS Sharpe
   - OOS 上运行策略 → 得到 OOS Sharpe
4. 对所有组合对，做 Spearman 秩相关：
   - 在每个 IS 集上，按 Sharpe 排序（rank_IS）
   - 在对应 OOS 集上，按 Sharpe 排序（rank_OOS）
5. 计算：
   - 选出 IS 中最优的策略组合（rank_IS = 1）
   - 这些组合在 OOS 中的秩 rank_OOS
   - PBO = P[rank_OOS(best_IS) < Median] 
```

**解读**：

| PBO | 含义 |
|:---:|------|
| < 0.10 | 🟢 IS 好 → OOS 也好，**没**有过拟合 |
| 0.10 ~ 0.30 | 🟡 轻微过拟合嫌疑 |
| > 0.30 | 🔴 IS 好 ≠ OOS 好，**过度拟合** |

**单策略简化**：如果只有一个策略（无参数搜索），PBO 退化为"时间鲁棒性"——在不同子区间上策略表现是否一致。实现方式：对策略在 IS 和 OOS 子段的 Sharpe 做相关性分析。

**实用简化**（适配我们的场景）：
```python
def compute_pbo_simplified(nav_series, n_splits=16):
    """
    简化版 PBO：将净值序列切成 N 个子段，计算各子段 Sharpe，
    然后做 leave-half-out 交叉验证。
    
    - 任选 8 个子段作为 IS，计算 IS Sharpe
    - 剩余 8 个子段作为 OOS，计算 OOS Sharpe
    - 重复 B=1000 次（随机抽样，不全枚举 12870）
    - 统计：IS_Sharpe 与 OOS_Sharpe 的 Spearman 秩相关 ρ
    - PBO ≈ (1 − ρ) / 2  （ρ 越低 → 过拟合越严重）
    """
```

---

### ⑤ SPS（Starting Point Sensitivity）— "如果运气最差那年开始投，有多惨？"

**问题**：回测从 2013 年开始是一个好年份。如果有人从 2015 年顶峰、2018 年贸易战前、2020 年疫情前开始投，结果怎样？

**计算**：
```
逐月滚动起点（2013-01 到 2023-01，共 ~120 个起点）：
  每个起点向后投 3 年固定期限
  记录：年化收益、Sharpe、最大回撤、终点净值
  
分析：
  - 最差起点：年化收益最低的那个
  - 收益分布：均值、标准差、偏度
  - "倒霉投资者"：下 10% 分位数
  - 负收益比例：终点净值 < 1.0 的起点占比
```

**解读**：

| 指标 | 🟢 好 | 🟡 可接受 | 🔴 差 |
|------|:-----:|:--------:|:----:|
| 最差起点年化 | > 5% | 0 ~ 5% | < 0% |
| 负收益比例 | < 10% | 10% ~ 25% | > 25% |
| 收益标准差 | < 8% | 8% ~ 15% | > 15% |

**关键洞察**：如果一个策略"年化 14% 但 2015 年开始就亏 30%"，对真实投资者的伤害远大于"年化 10% 但每年都赚"。SPS 回答的是"最坏情况下我能不能接受"。

---

## 二、最终五指标体系

| # | 指标 | 问什么 | 🟢/🟡/🔴 门控 |
|---|------|--------|:---:|
| ① | DSR | 真 alpha 还是运气？ | >0.95 / >0.85 / <0.85 |
| ② | MC 生存率 | 参数偏了还能赚钱？ | >80% / 50-80% / <50% |
| ③ | WF 基准相对胜率 | 比等权持有更强？ | >80% / 60-80% / <60% |
| ④ | PBO | 回测过拟合了吗？ | <0.10 / <0.30 / >0.30 |
| ⑤ | SPS 最差起点 | 最倒霉时还能赚吗？ | >5% / 0-5% / <0% |

### 综合判定

```
5/5 绿 → 🟢 实盘可上（高度鲁棒，任何维度找不到软肋）
4/5 绿 → 🟢 实盘可上（一个维度黄，可接受）
3/5 绿 + 0 红  → 🟡 可小资金试
≥2 红 → 🔴 需要改进
```

---

## 三、实现规范

### 新增函数（`src/robustness.py`）

```python
# === ④ PBO ===

def compute_pbo_simplified(
    config_path: str,
    n_splits: int = 16,
    n_resamples: int = 1000,
) -> tuple[float, dict]:
    """
    简化版 PBO (CSCV)。

    Args:
        config_path: 策略配置
        n_splits: 子段数（默认 16，对应 ~10 个月/段）
        n_resamples: 重采样次数（默认 1000，不全枚举 C(16,8)）

    Returns:
        pbo: PBO 概率 (0~1)
        details: {rank_corr, is_sharpes, oos_sharpes, ...}
    """


# === ⑤ 起点敏感性 ===

def compute_starting_point_sensitivity(
    config_path: str,
    horizon_years: int = 3,
    step_months: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """
    滚动起点敏感度分析。

    Args:
        config_path: 策略配置
        horizon_years: 投资期限（默认 3 年）
        step_months: 滚动步长（默认 1 月）

    Returns:
        metrics: {
            'worst_annual_return': float,
            'worst_start_date': str,
            'mean_annual_return': float,
            'std_annual_return': float,
            'negative_return_ratio': float,
            'p10_annual_return': float,
            'best_annual_return': float,
        }
        details_df: 每个起点的详细结果 DataFrame
    """
```

### RobustnessResult 新增字段

```python
@dataclass
class RobustnessResult:
    # ... 原有字段 ...
    pbo: float | None = None                    # [v3] PBO
    starting_point_sensitivity: dict | None = None  # [v3] SPS metrics
```

### CLI 新增参数

```bash
--pbo           # 启用 PBO 计算（默认关闭，n_resamples=1000）
--pbo-splits 16 # PBO 子段数
--sps           # 启用起点敏感度分析（默认关闭）
--sps-horizon 3 # 起点敏感度投资期限（年）
```

### 预估耗时

| 项目 | 计算量 | 预估耗时 |
|------|:-----:|:------:|
| PBO (单策略) | 1000 次回测 | ~5 min |
| SPS (单策略) | ~120 次回测 | ~2 min |

---

## 四、报告新增章节

### PBO 章节
```
## ④ PBO（概率过拟合）

| 策略 | PBO | IS/OOS 秩相关 | 评级 |
|------|:---:|:-----------:|:----:|
| v2.3 基线 | X.XX | X.XX | 🟢/🟡/🔴 |
| D4 tuned | X.XX | X.XX | 🟢/🟡/🔴 |

分析: ...

### CSCV 分布
- IS Sharpe vs OOS Sharpe 散点图（可选，文字描述即可）
```

### 起点敏感性章节
```
## ⑤ 起点敏感性（SPS）

| 指标 | v2.3 基线 | D4 tuned |
|------|:--------:|:--------:|
| 最差起点年化 | X.XX% (YYYY-MM) | X.XX% (YYYY-MM) |
| 均值年化 | X.XX% | X.XX% |
| 标准差 | X.XX% | X.XX% |
| 负收益比例 | X.X% | X.X% |
| P10 年化 | X.XX% | X.XX% |

分析:
- "倒霉投资者"从最差起点开始投 3 年，结果...
- D4 tuned 的最差起点是否优于基线？

### 滚动起点收益分布
- 横轴: 起点日期，纵轴: 3年年化收益
- 文字描述分布形状、尾部风险
```

---

## 五、最终五指标对比表

| 指标 | v2.3 基线 | v2.3+cap040+D4 tuned | 优胜 |
|------|:--------:|:-------------------:|:----:|
| ① DSR | X.XX 🟢 | X.XX 🟢 | |
| ② MC 生存率 | XX% 🟡 | XX% 🟢 | |
| ③ WF 相对胜率 | XX% 🟡 | XX% 🟡 | |
| **④ PBO** | X.XX 🟢 | X.XX 🟢 | |
| **⑤ SPS 最差起点** | X.XX% 🟡 | X.XX% 🟢 | |
| **综合** | X/5 绿 | X/5 绿 | |

---

*本文档由 quant-se 设计，作为鲁棒性 v3 实现规范。*