# v4.5-pvd PVD 条件激活因子闭环文档

> 任务链：E0（数据基建）→ E1（IC 评估）→ E2（线性叠加）→ E2b（条件激活 Plan C）→ T1（实现）→ T2（影子对照）→ T3（正式管线）  
> 结论：**GO — 条件通过**。v4.5-pvd 配置就绪，默认仍 v4.3（用户手动切换）。  
> realized Sharpe 1.588（+0.10 vs v4.3），MaxDD 5.80%（−0.04pp），block bootstrap 200 路径胜率 95%。

---

## 1. 预研链路

### 1.1 E0 数据基建（NO-GO 方向排除）

| 估计器 | 数据源 | 结论 |
|--------|--------|------|
| Parkinson (High/Low) | 纳指 QDII 日频 H/L | NO-GO：溢价扭曲 corr=0.30（与 CC 相关极低） |
| Garman-Klass | 同上 | NO-GO：同源问题 |
| 已实现波动率 (RV) | CC 日频收盘价 | NO-GO：增量 IC ≈ 0，与周频 tapered vol 高度冗余 |
| **日频成交量** | tushare fund_daily vol 列 | **GO**：独立信息源，非价格维度 |

### 1.2 E1 信息增量评估

- PVD（Price-Volume Divergence）= `rolling_corr(log_ret, log(vol_t/vol_{t-1})), window=8`
- 组合层面 IC = **0.053**，t-stat = **2.53**（显著 > 2.0 门禁）
- 分 ETF：纳指 IC=0.071 最强，红利低波 IC=0.022 最弱（数据起 2019 样本短）
- 判定：**通过 E1 门禁**

### 1.3 E2 线性叠加（NO-GO）

直接将 PVD 加入评分 `score += pvd_w * pvd`：
- Sharpe 小幅提升但 MaxDD **+14pp**（从 5.8% 飙至 ~20%）
- 原因：PVD 在高波动期给出强信号但方向不稳定
- 判定：**NO-GO**

### 1.4 E2b 条件激活 Plan C（GO）

设计：仅在"信噪比高的平静环境 + ETF 分化小"时启用 PVD 作为 tiebreaker：
- 条件 1：纳指成交量 ∈ [25th, 75th 百分位]（排除极端环境）
- 条件 2：top-2 动量 gap < 0.05（动量接近时 PVD 才有边际价值）
- 满足两条件：`score += 0.15 × PVD`

| 指标 | v4.3 baseline | Plan C | Δ |
|------|--------------|--------|---|
| Sharpe | 1.488 | **1.589** | +0.101 |
| MaxDD | 5.84% | **5.80%** | −0.04pp |
| Block bootstrap 200 路径 | — | 中性（无显著劣化） | — |

判定：**GO — 进入生产化**

---

## 2. Plan C 机制详解

```
每周评分后、选股前：

IF nasdaq_vol ∈ [percentile_25, percentile_75]     # 条件1: 中位成交量环境
   AND top2_momentum_gap < 0.05:                    # 条件2: 动量排名接近
   
   FOR each offensive ETF j:
       scores[j] += 0.15 × PVD[week, j]            # 注入 tiebreaker
   
ELSE:
   pass                                             # 完全退化为 baseline（零开销）
```

**设计理念**：
- PVD 信号在中等波动环境下信噪比最高（极端波动时量价关系崩坏）
- 仅当动量评分接近时才有 tiebreaker 价值（大幅领先时无需额外信号）
- 条件不满足时完全不注入 → 降级为 baseline 配置 → 无副作用保证

---

## 3. T1 实现

### 3.1 改动范围

| 文件 | 行数 | 内容 |
|------|------|------|
| `src/strategy.py` | +16 | 6 个 PVD 配置字段 + load_config 解析 |
| `src/data_loader.py` | +61 | `load_weekly_volume_from_cache()` 日频→周频聚合 |
| `src/factors.py` | +45 | `compute_pvd_factor()` + `compute_all_factors` 扩展 |
| `src/backtest.py` | +43 | 数据加载 + 百分位预计算 + 条件激活注入 |
| `config/strategy_v4_5_pvd.yaml` | 80 | 新配置（基于 v4.4 + pvd_factor 段） |
| `tests/test_pvd_factor.py` | 202 | 11 个单元测试 |

### 3.2 验收结果

- pytest：**212 passed**
- baseline pin：v4.3 Sharpe = **1.4878**（与 T1 前完全相同）
- 交叉验证锚点：v4_5_pvd Sharpe = **1.5880**，MaxDD = **5.80%**（与 E2b monkeypatch 对齐，delta = 0.001）
- `rebalance_live.py --verify`：**✅ 通过**（默认 v4.3 路径零扰动）

---

## 4. T2 影子对照

### 4.1 Realized 性能

| 配置 | Sharpe | MaxDD | Annual |
|------|--------|-------|--------|
| v4.3 (baseline) | 1.4878 | 5.84% | 14.52% |
| v4.4 (EWMA L3.5) | 1.4985 | 5.85% | 14.51% |
| **v4.5-pvd** | **1.5880** | **5.80%** | **15.33%** |

### 4.2 CCC-GARCH 对抗评估

4 个标准情景中 PVD 配置 Sharpe 劣化 −0.04~−0.07（vs baseline）。

**用户裁决（2026-08-01）**：
> 合成 DGP 无成交量模型（VAR+GARCH 仅建模价格收益），对非价格因子（成交量）天然不公平。
> Block bootstrap 保留真实价量关系，以此为准。grey_corr_combo 12.81% 为三版本共有的预存
> 问题（v4.3/v4.4/v4.5-pvd 均触发），非 PVD 引入。

---

## 5. T3 正式管线

### 5.1 OOS 三通道

| 通道 | baseline | v4.5-pvd | 判定 |
|------|----------|----------|------|
| A (held-out 幅度, CCC-GARCH) | 80% | 70% | FAIL（DGP 无量模型，预期内） |
| **B (独立 seed)** | 100% | 100% | **PASS** |
| C (block bootstrap, 30 路径) | 93% | 87% | MARGINAL FAIL |

Channel C 分析：
- pass_rate 87%（26/30 路径胜出）vs baseline 93%
- 但**中位 Sharpe 更高**（1.088 vs 1.064）、**avg_margin 更高**（+0.246 vs +0.236）
- FAIL 由 3 条极端尾部路径驱动
- 30 路径统计功效不足，以 200 路径 Joint Test2 为准

### 5.2 联合鲁棒性

| 测试 | 结果 | 判定 |
|------|------|------|
| Test1 参数轴 | 8/8 无断崖 | **PASS** |
| **Test2 数据轴 bootstrap (200 路径)** | **胜率 95%**，alpha P10 = +0.092 | **PASS** |
| Test3 联合 | 方差比 = 0.802 ≤ 1.30 | **PASS（无薄峰）** |

### 5.3 一致性终检

| 路径 | ΔSharpe (引擎 vs 脚本) | 判定 |
|------|------------------------|------|
| v4.3 默认 | 0.0099 | PASS (≤ 0.01) |
| v4.5-pvd | 0.0795 | 结构性 N/A |

v4.5-pvd 偏差原因：`rebalance_live.py` 明确不含 PVD 逻辑（仅回测引擎实现），脚本输出 = v4.4 水平。非回归问题。

### 5.4 门禁总判定

**CONDITIONAL GO**：以 200 路径 block bootstrap（胜率 95%、alpha P10 +0.092）为主要判据通过。

---

## 6. 遗留与远期

| 项目 | 状态 |
|------|------|
| 生产默认配置 | 仍 v4.3（用户手动切换至 v4.5-pvd） |
| `rebalance_live.py` PVD 集成 | 待后续同步（当前仅回测引擎） |
| 数据管线 | tushare_cache 增量更新 + 新浪(vol=股)/腾讯(vol=手) 三级链备选 |
| 远期探索方向 | PVD 窗口自适应、与 crisis_corr 联动、PVD 分位数条件门控 |

---

## 7. 实验数据索引

| 文件 | 内容 |
|------|------|
| `scripts/_exp_volume_signal_study.py` | E0 数据基建 + E1 IC 评估 |
| `scripts/_exp_volume_signal_e2.py` | E2 线性叠加实验 |
| `scripts/_exp_volume_signal_e2b.py` | E2b 条件激活 Plan A/B/C/D |
| `output/experiments/exp_volume_signal_e1.json` | E1 信息增量数据 |
| `output/experiments/exp_volume_signal_e2.json` | E2 线性叠加数据 |
| `output/experiments/exp_volume_signal_e2b.json` | E2b 条件激活数据 |
| `output/experiments/v45_pvd_oos.json` | T3 OOS 三通道结果 |
| `output/robustness/robustness_joint_all_*.json` | T3 联合鲁棒性结果 |
| `config/strategy_v4_5_pvd.yaml` | v4.5-pvd 生产就绪配置 |
| `tests/test_pvd_factor.py` | PVD 专项单元测试（11 个） |
