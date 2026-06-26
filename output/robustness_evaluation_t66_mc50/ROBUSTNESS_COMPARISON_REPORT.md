# 鲁棒性对比评估报告 (v3)

**评估日期**: 2026-06-25 00:38
**策略数**: 1

---

## 策略对比

| 指标 | config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml |
|------|:--------:|
| 年化收益 | 10.35% |
| 最大回撤 | 13.32% |
| 夏普比率 | 0.781 |

---

## 鲁棒性指标

### ① DSR（Deflated Sharpe Ratio）

| 策略 | DSR | 评级 | 解读 |
|------|:---:|:----:|------|
| config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml | 0.0000 | 🔴 | 很可能只是运气好选中了 |

### ② MC 生存率（参数扰动盈利概率，v2 收紧标准）

| 策略 | 生存率 | 评级 | 解读 |
|------|:-----:|:----:|------|
| config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml | 20.0% | 🔴 | 在刀刃上，参数略微偏离就可能亏损 |

### ③ 基准相对胜率（Walk-Forward 相对等权基准）

| 策略 | 胜率 | 评级 | 解读 |
|------|:---:|:----:|------|
| config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml | 50.0% | 🔴 | 大部分时间不如简单等权持有 |

---

## ④ PSS（参数稳定性评分）

| 策略 | 年化 P10/P50/P90 | DD P10/P50/P90 | Sharpe P10/P50/P90 | CV | 评级 |
|------|:---:|:---:|:---:|:--:|:--:|
| config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml | 8.7%/9.4%/10.4% | 13.1%/17.6%/20.2% | 0.62/0.67/0.77 | r:0.07 d:0.19 s:0.10 | 🔴 |

---

## 五指标综合对比

| 指标 | config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml | 优胜 |
|------|:--------:|:----:|
| ① DSR | 0.0000 🔴 | config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml |
| ② MC 生存率 | 20.0% 🔴 | config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml |
| ③ WF 相对胜率 | 50.0% 🔴 | config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml |
| **④ PSS** | 8.7%/20.2% 🔴 | config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml |
| **综合** | 0/5 绿, 4 红 | |

---

## 综合判定

**综合评级**: 🔴 需要改进（4 红）

| 策略 | 综合评级 | 建议 |
|------|:------:|------|
| config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml | 🔴 | 需要改进 |

---

## 详细数据

### config/strategy_v2_3_cap040_D4_tuned_improved_v5.yaml

**Walk-Forward 窗口明细**:

| 窗口 | 起始 | 结束 | 策略 Sharpe | 基准 Sharpe | 相对 Sharpe | 跑赢 |
|------|------|------|:----------:|:----------:|:---------:|:---:|
| 0 | 2013-10-21 | 2014-10-21 | 1.0181 | 0.3929 | 0.6252 | ✅ |
| 1 | 2025-04-20 | 2026-04-20 | 0.7304 | 1.5773 | -0.8469 | ❌ |

**MC 统计** (50 次有效运行):
- Sharpe 均值: 0.6926
- Sharpe 标准差: 0.0663
- Sharpe 范围: [0.5286, 0.8516]
- 年化收益均值: 9.48%
- 最大回撤均值: 16.72%
- 年化>10% AND DD<15% 次数: 10/50
- MC 生存率: 20.0%

**DSR 计算参数**:
- 观测 Sharpe: 0.7809
- 试验数: 10
- 观测数: 649
- 偏度: 0.3922
- 峰度: 4.5602

**PSS 参数稳定性详情**:
- 年化收益 P10/P50/P90: 8.71% / 9.36% / 10.39%
- 最大回撤 P10/P50/P90: 13.15% / 17.60% / 20.19%
- Sharpe P10/P50/P90: 0.6185 / 0.6729 / 0.7742
- CV (收益/回撤/Sharpe): 0.0716 / 0.1898 / 0.0957
- 样本数: 50

---

## 数据文件

- 结构化结果: `/home/ubuntu/claw_etf_strategy/output/robustness_evaluation_t66_mc50/robustness_results.json`
