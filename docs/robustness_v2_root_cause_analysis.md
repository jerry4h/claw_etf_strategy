# Robustness V2 根因分析与改进方案

> Author: quant-se (技术竞争力负责人)
> Date: 2026-06-17
> Trigger: T57 — V2 DSR=0.0144/0.0857, MC survival 100% (lenient criterion), worst delta -1.6959

---

## 1. 现状（两套实现，两套结果）

项目目前有两套鲁棒性评估实现，给出一致性极差的结果：

| 指标 | `src/robustness.py` (完整公式) | `src/robustness_v2.py` (简化公式) |
|------|:--------------------------:|:------------------------------:|
| DSR 公式 | 偏度/峰度感知的完整 B&LdP | 固定 sharpe_std=0.5 |
| MC 扰动参数 | 7 个 (含 D4) | 4 个 (不含 D4) |
| MC 生存标准 | 年化>10% AND DD<15% | Sharpe > 0 |
| n_trials 默认 | 2 (CLI) | 52 (hardcoded) |

### 结果对比

| 维度 | robustness.py (n_trials=2) | robustness_v2.py (n_trials=52) |
|------|:--------------------------:|:------------------------------:|
| DSR (基线) | ~1.0 | 0.0144 |
| DSR (D4) | ~1.0 | 0.0857 |
| MC 生存率 (基线) | 48.75% (±15%, goal-aligned) | 100% (±10%, Sharpe>0) |
| MC 生存率 (D4) | ? (未跑) | 100% (±10%, Sharpe>0) |
| WF 最差 delta | -0.584 (基线 V2) | -1.6959 (D4 tuned, 推测) |

**核心矛盾**：同一策略，一套说「高度鲁棒」(DSR=1.0)，另一套说「纯运气」(DSR=0.0144)。

---

## 2. 根因分析

### 根因 1: DSR 两套公式差异达 70×

**robustness_v2.py 简化公式**（与 robustness_v2_report.py 联动）：

```python
sharpe_std = 0.5  # 固定值，代表截面 Sharpe 标准差
e_max = sharpe_std * sqrt(2 * ln(n_trials))
se_sr = sharpe_std / sqrt(T_years)
```

当 n_trials=52, T_years=12.96 时：
- E[max(SR)] = 0.5 × √(2 × ln(52)) = 1.405
- SE(SR) = 0.5 / √12.96 = 0.139
- 基线 z-score = (1.102 − 1.405) / 0.139 = −2.18 → DSR = 0.0144
- D4 z-score = (1.216 − 1.405) / 0.139 = −1.36 → DSR = 0.0857

**robustness.py 完整公式**（正确实现）：

```python
e_max_sr = sqrt(2 * ln(n_trials)) * (1 − γ·SR̂ + (γ²−1)/4 · SR̂²)
se_sr = sqrt((1 + 0.5·SR̂² − skew·SR̂ + (kurt−3)/4 · SR̂²) / n_obs)
```

当 n_trials=52, skew=0.727, kurt=5.851 时：
- E[max(SR)] = 2.811 × (1 − 0.636 + (−0.667)/4 × 1.214) = 2.811 × 0.161 = 0.454
- SE(SR) = √(1.671 / 649) = 0.051
- 基线 z-score = (1.102 − 0.454) / 0.051 = +12.78 → DSR ≈ 1.0

**判定**：简化公式将截面估计 (sharpe_std=0.5) 误用于时序推断，忽略了策略自身收益分布的正偏度 (0.727) 对标准误的压缩效应。**完整公式才是正确的**——B&LdP 2014 原论文同时给出两个公式，截面公式用于跨策略比较（未知各策略分布），完整公式用于单策略推断（已知分布）。本项目场景是单策略推断，应使用完整公式。

**结论**：DSR 实际上 ≈ 1.0（高度显著），不是 0.0144。0.0144 是使用了错误的公式。

### 根因 2: n_trials CLI 默认值错误

`scripts/run_robustness.py` 第 76 行：

```python
parser.add_argument('--n-trials', type=int, default=2, ...)
```

默认 n_trials=2 仅覆盖最终 2 个候选策略（基线 + D4 tuned），忽略了 50 个中间变体。全搜索历史为 52。应默认为 52。

**影响**：当 n_trials=2 时，E[max(SR)] ≈ 0.83（完整公式），z-score 更大，DSR 更高但掩盖了真实搜索空间。

### 根因 3: MC 生存标准两套不一致

| 实现 | 标准 | 基线结果 (400 runs, ±15%) |
|------|------|:-------------------------:|
| robustness_v2.py | Sharpe > 0 | ~100% (过于宽松) |
| robustness.py | 年化>10% AND DD<15% | 48.75% (正确标准) |

Sharpe > 0 对 Sharpe=1.1 的策略几乎 100% 通过——就像问「博尔特还能走路吗？」。

**正确标准**（已在 robustness.py 中实现，但 robustness_v2.py 未采用）：
```python
n_survived = sum(1 for r in mc_details
                 if r['annual_return'] > 0.10 and r['max_drawdown'] < 0.15)
```

这直接对齐用户原始目标：年化 > 10% 且 DD < 15%。

**关键洞察**：策略有两个独立鲁棒性维度：
- **收益端**：极鲁棒（100% 通过 return > 10%）
- **风险端**：中度脆弱（48.75% 通过 DD < 15%）

DD 是瓶颈。参数扰动主要打击防御机制（vol 三段式），导致回撤放大。

### 根因 4: robustness_v2.py 未扰动 D4 参数

robustness_v2.py 的 `_perturb_and_clamp()` 仅扰动 4 个参数：
- mom_w, vol_w, def_alloc, stop_loss

**缺失 D4 关键参数**：
- d4_momentum_window (对 D4 tuned 策略有实质影响)
- d4_momentum_threshold (对 D4 tuned 策略有实质影响)
- step_high, step_low (vol 三段式参数，OAT 显示有敏感度)

robustness.py 已正确包含全部 7 个参数。robustness_v2.py 是旧版，不应使用。

### 根因 5: Walk-Forward 最差 delta 需要定位

V2 基线 WF 结果的最差 delta = -0.584（窗口 2016-09 至 2017-09，策略 Sharpe 0.77 vs 基准 1.36）。这是合理的——2016-2017 年 A 股结构化行情中，基准（等权 5 ETF）跑赢策略。

但 D4 tuned 的被引用最差 delta = -1.6959（远差于基线）——这说明 D4 策略在某个窗口严重跑输基准。需要：
1. 确认 D4 的哪个窗口
2. 分析该窗口的市场特征
3. 判断是 D4 参数问题还是该窗口本身是异常值

### 根因 6: OAT 揭示的敏感度模式

从 V2 OAT 数据（基线，7 参数 × 7 级别）：

| 参数 | 敏感度 | Sharpe 范围 | DD 范围 | 备注 |
|------|:------:|:----------:|:------:|------|
| mom_w | 🔴 高 | 0.90 → 1.10 | 7.4% → 17.2% | +15% 触发 DD 爆炸 |
| vol_w | 🔴 高 | 0.92 → 1.10 | 7.4% → 17.2% | −15% 触发 DD 爆炸 |
| step_high | 🟡 中 | 1.02 → 1.10 | 7.4% → 11.2% | −15% 影响较大 |
| step_low | 🟢 低 | 1.08 → 1.10 | 7.4% → 8.2% | 几乎无影响 |
| def_alloc | 🟢 低 | 1.09 → 1.11 | 7.3% → 7.9% | 线性，小范围 |
| momentum_window | — | 无变化 | 无变化 | 基线 d4_enabled=false |
| momentum_threshold | — | 无变化 | 无变化 | 基线 d4_enabled=false |

**核心模式**：mom_w ↑ 或 vol_w ↓ 时，DD 从 7.4% 跃升至 17.2%。因为权重偏移导致 top_n 选择了更波动/趋势型的 ETF，同时 vol 防御在错误强度激活。这是 MC 生存率仅 48.75% 的主要驱动因素。

---

## 3. 改进方案

### P0 修复（阻塞项，必须在下次评估前修复）

| ID | 修复 | 文件 | 变更 |
|----|------|------|------|
| F1 | 统一 DSR 为完整公式 | `src/robustness_v2.py` → 废弃或重写 | 删除简化公式，导入 robustness.py 的 compute_dsr() |
| F2 | 修复 n_trials 默认值 | `scripts/run_robustness.py` L76 | `default=2` → `default=52` |
| F3 | 统一 MC 生存标准 | `src/robustness_v2.py` L248 | `sr > 0` → `年化>10% AND DD<15%` |
| F4 | 废弃 `scripts/robustness_v2_report.py` | 删除或 redirect | 统一用 `scripts/run_robustness.py` |

### P1 增强（提升鲁棒性洞察深度）

| ID | 增强 | 说明 |
|----|------|------|
| E1 | D4 策略 MC 对比评估 | 对基线 AND D4 分别跑 400 次 MC (±15%, goal-aligned criteria)，获取正确的 MC 生存率对比 |
| E2 | WF 最差 delta 根因定位 | 找 D4 tuned 的 -1.6959 窗口，输出该窗口的市场特征（各 ETF 收益、波动率、PE 百分位） |
| E3 | MC 扰动覆盖 D4 专用参数 | 确认 `src/robustness.py` 的 MC_PARAMS 对 D4 启用策略完整覆盖（已含 7 参数，确认） |
| E4 | OAT 增加 regime 交互分析 | 按 regime（BULL/NORMAL/BEAR）分拆 OAT 结果，看哪些参数的敏感度是 regime-dependent |

### P2 战略（策略层面改进方向）

| ID | 方向 | 触发条件 | 预期效果 |
|----|------|----------|----------|
| S1 | 防御机制硬化 | MC 生存率 < 60% (DD 瓶颈) | 扩宽 vol 三段式阶梯、增加第二波动率代理 |
| S2 | D4 动量窗口去敏感化 | D4 WF 最差 delta < -1.0 | 调整 momentum_threshold 或加 regime-conditional override |
| S3 | 评分权重平滑 | OAT 显示 mom_w/vol_w 高敏感 | 探索 softmax 替代 hard top_n（Phase 5 方案） |

---

## 4. 实施路径

### Phase 1: P0 修复（本次 T57 产出）

创建 3 个 kanban task：

```
T58 (coder):   F1-F4 修复 — unify DSR formula, fix n_trials default, unify MC criterion, deprecate robustness_v2
T59 (reviewer): Review T58 — verify formula correctness, MC criterion, n_trials default
T60 (tester):  Run robustness V3 — 基线 + D4 对比, n_trials=52, MC=400, ±15%, goal-aligned
```

### Phase 2: P1 增强（T60 产出后触发）

- 根据 V3 结果决定是否启动 E1-E4

### Phase 3: P2 战略（E1-E4 产出后触发）

- 根据增强结果决定 S1-S3 优先级

---

## 5. 关键判定

| 结论 | 说明 |
|------|------|
| DSR 实际 ≈ 1.0 | 完整公式+正偏度使 DSR 接近 1.0，策略 Sharpe 是真 alpha |
| MC 生存率实际 ~48-54% | DD 是瓶颈，防御机制（vol 三段式）需硬化 |
| WF 胜率 6/9 = 66.7% | 可接受，但非优秀；最差 delta 需调查 |
| 两套实现是混乱根源 | P0 修复后统一为 robustness.py（完整公式） |
| n_trials=2 是 bug | 应默认为 52（全搜索历史） |