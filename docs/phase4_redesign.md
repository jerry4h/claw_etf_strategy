# T36: Phase4 Redesign — Fix Regime Classifier Skew + Regain Robustness

## Executive Summary

当前T35实现的Direction A (Regime Classifier)存在严重偏斜：DEFENSIVE占70.7%，
导致回测年化收益率从D4-tuned基准的15.65%下降到12.56%（-3.09pp），最大回撤反而
从7.58%恶化到12.58%（+5.0pp）。Walk-Forward Sharpe标准差从1.208降至0.857
（29%改善），但距离目标<0.15仍有数量级差距。

本设计文档提出3个修复方向 + 1个完整替代方案，并进行对比评估。

### 关键发现

1. **BREADTH信号的CSI300/SH Composite比值代理信息含量低**。该比值在1.02-1.59
   窄幅波动（均值1.20, std=0.12），52周滚动z-score主要捕捉噪声。±0.5阈值对于
   std=1.47的信号过紧，导致NARROW仅占13.6%。

2. **DEFENSIVE规则的OR逻辑是skew的根因**。单一BEAR trend(221周)、单一THIN
   breadth(191周)分别贡献了大量DEFENSIVE，但真正需要防御的TIGHT liquidity仅5周
   被单独触发。

3. **RISK_ON条件过严**：4个条件(BULL|WEAK AND LOOSE AND WIDE AND
   DISINFLATION|MODERATE)的交集仅覆盖4.2%的周数。

---

## 1. Root Cause Analysis

### 1.1 BREADTH信号分解

#### 数据特征

```
breadth_score = -(ratio - mean(52w)) / std(52w)
  where ratio = 4-week avg of (CSI300 / SH Composite)

统计量:
  Count:      674 valid / 683 total
  Mean:       -0.0434
  Std:        1.4745
  Range:      [-3.9477, 2.7974]
  Percentiles: 10%=-2.06  25%=-1.35  50%=0.16  75%=1.31  90%=1.65
```

#### 底层的CSI300/SH比值

```
CSI300 / SH Composite ratio:
  Mean:  1.2012
  Std:   0.1219
  Range: [1.0204, 1.5889]
```

**关键问题**：该比值是缓变量（跨年尺度）。在52周窗口内比值变化极小，z-score
主要反映短期噪声而非真实的breadth变化。当CSI300和SH Composite高度相关时
（中国A股市场常态），比值z-score的判别力有限。

#### 阈值敏感性

| 阈值 | WIDE | THIN | NARROW |
|------|------|------|--------|
| ±0.5 | 44.7% | 40.4% | **13.6%** |  ← 当前，极不均衡
| ±0.7 | 42.6% | 38.1% | 19.3% |
| ±1.0 | 32.9% | 32.1% | **35.0%** |  ← 最优均衡点
| ±1.2 | 29.3% | 27.2% | 43.5% |
| ±1.5 | 16.3% | 21.5% | 62.2% |

±1.0（约0.68σ）达到最佳三态均衡。±0.5过于激进，将40%+周数推向极端。

### 1.2 Regime Classifier OR-Sink分析

```python
# 当前分类逻辑 (src/regime.py:classify_regime)
DEFENSIVE if (trend==BEAR) OR (liquid==TIGHT) OR (breadth==THIN)
```

单信号贡献分解（483个DEFENSIVE周）：

| 触发原因 | 周数 | 占比 |
|----------|------|------|
| THIN breadth only | 221 | 45.8% |
| BEAR trend only | 191 | 39.5% |
| TIGHT liquidity only | 5 | 1.0% |
| 多信号同时触发 | 66 | 13.7% |

**结论**：BEAR trend（36.9%的周）和THIN breadth（40.4%的周）各自独立覆盖
大部分DEFENSIVE。TIGHT liquidity（3.2%的周）几乎从未独立触发。OR逻辑将
三个"常见"信号合并，产生70.7%的超大DEFENSIVE区域。

### 1.3 RISK_ON条件过严

```python
RISK_ON if (BULL|WEAK) AND LOOSE AND WIDE AND (DISINFLATION|MODERATE)
```

四个条件的交集极小：
- (BULL|WEAK) = 341周 (49.9%)
- LOOSE = 220周 (32.2%)
- WIDE = 305周 (44.7%)
- (DISINFLATION|MODERATE) = 645周 (94.4%)
- 四条件交集 = 29周 (4.2%)

CPI条件几乎不限制（94.4%通过），但LOOSE liquidity（32.2%）是关键瓶颈。

### 1.4 WF回测影响

| Metric | D4-tuned Baseline | Regime ON (当前) | Δ |
|--------|-------------------|-------------------|------|
| Annual Return | 15.65% | 12.56% | **-3.09pp** |
| MaxDD | 7.58% | 12.58% | **+5.0pp** |
| Sharpe | 1.216 | 1.009 | **-0.207** |
| WF Sharpe Std | 1.208 | 0.857 | -0.351 (改善) |
| WF Sharpe Min | -0.387 | -0.300 | +0.087 (改善) |

**WF各窗口对比**：

| Window | Period | D4-tuned | Regime ON | Δ |
|--------|--------|----------|-----------|------|
| 0 | 2016-17 | 0.167 | 0.136 | -0.031 |
| 1 | 2017-18 | 0.105 | 0.141 | +0.036 |
| 2 | 2018-19 | 0.597 | 1.614 | **+1.017** |
| 3 | 2019-20 | 1.120 | 1.180 | +0.060 |
| 4 | 2020-21 | 0.495 | 0.626 | +0.131 |
| 5 | 2021-22 | -0.387 | -0.300 | +0.087 |
| 6 | 2022-23 | 2.597 | 2.378 | -0.219 |
| 7 | 2023-24 | 3.533 | 1.332 | **-2.201** |
| 8 | 2024-25 | 1.622 | 1.737 | +0.115 |

关键观察：
- 2023-24窗口（大牛市）大幅恶化（3.533→1.332），因为该窗口内大部分时间被
  错误分类为DEFENSIVE，限制了牛市中的收益
- 2018-19窗口大幅改善（0.597→1.614），因为2018年确实处于BEAR，DEFENSIVE保护
  发挥了作用
- 整体：DEFENSIVE过于频繁，在牛市中拖累收益，仅在真正的熊市中才有正贡献

---

## 2. Fix Direction 1: Recalibrated Regime Classifier v1.1 (RECOMMENDED)

### 2.1 变更清单

| # | 变更 | 当前值 | 新值 | 理由 |
|---|------|--------|------|------|
| 1 | BREADTH阈值 | ±0.5 | **±1.0** | 优化到±0.68σ，三态均衡(各~33%) |
| 2 | DEFENSIVE规则 | BEAR OR TIGHT OR THIN | **(BEAR AND THIN) OR TIGHT** | 从OR改为AND-OR，减少过度防御 |
| 3 | RISK_ON规则 | ...AND WIDE AND (DIS/MOD) | **...AND (WIDE|NARROW) AND LOOSE** | 移除CPI要求，WIDE放宽到NOT-THIN |

### 2.2 预期Regime分布

```
CAUTIOUS:  556/683 (81.4%)  ← 主模式（baseline行为）
RISK_ON:    75/683 (11.0%)  ← 适度进攻
DEFENSIVE:  52/683 ( 7.6%)  ← 仅在真正危险时
BUBBLE_WARN: 0/683 ( 0.0%)  ← 由于±1.0+TIGHT极难同时出现
CRISIS:      0/683 ( 0.0%)  ← 2013-2026期间无真正危机
```

### 2.3 预期性能影响

**回报恢复**：81.4%的周使用baseline参数（当前仅24.9%），预期年化收益率
回升至14-16%（接近D4-tuned基准的15.65%）。

**回撤控制**：7.6%的DEFENSIVE周覆盖真正的BEAR+THIN重叠期（如2018下半年、
2022上半年），TIGHT liquidity触发防御。MaxDD预期≤8.5%。

**WF鲁棒性**：
- 大部分窗口表现接近D4-tuned（因为81%时间使用base参数）
- 仅真正的弱市窗口触发防御（减少不必要的收益牺牲）
- WF Sharpe std 预期：0.4-0.7（当前0.857，目标<0.15仍aggressive）
- WF Sharpe min 预期：>0.0（当前-0.300，改善但仍可能为负）

### 2.4 实现变更

`src/regime.py` 中修改 `classify_regime()` 函数和 `compute_breadth_from_proxy()` 函数。

```python
# compute_breadth_from_proxy() — 阈值调整
if breadth_score > 1.0:      # was 0.5
    return BreadthState.WIDE
elif breadth_score < -1.0:   # was -0.5
    return BreadthState.THIN
else:
    return BreadthState.NARROW

# classify_regime() — 规则调整
# DEFENSIVE: (BEAR AND THIN) OR TIGHT (was: BEAR OR TIGHT OR THIN)
if (trend == TrendState.BEAR and breadth == BreadthState.THIN) or \
   liquid == LiquidState.TIGHT:
    return Regime.DEFENSIVE

# RISK_ON: (BULL|WEAK) AND (WIDE|NARROW) AND LOOSE
#          (was: ...AND WIDE AND (DISINFLATION|MODERATE))
if (trend in (TrendState.BULL, TrendState.WEAK) and
    breadth != BreadthState.THIN and
    liquid == LiquidState.LOOSE):
    return Regime.RISK_ON
```

**向后兼容**：配置开关 `regime_classifier.enabled` 不变。默认OFF。

---

## 3. Fix Direction 2: 3-State Simplified Regime v1.2 (ALTERNATIVE)

### 3.1 设计理念

放弃T34中5状态设计中的BREADTH子信号（信息含量不足）和BUBBLE_WARN/CRISIS
（2013-2026数据中几乎不出现），简化为3状态分类器。

### 3.2 信号设计

```
RISK_ON:  BULL trend AND LOOSE liquidity
DEFENSIVE: BEAR trend AND TIGHT liquidity
NEUTRAL:   everything else (使用baseline参数)
```

**设计理由**：
- TREND通过CSI300 26周MA提供可靠的中期方向判断
- LIQUID通过M1/M2货币条件提供宏观经济环境信号
- 两个信号正交：TREND捕捉市场价格，LIQUID捕捉流动性
- AND条件确保两个维度都确认时才有动作

### 3.3 预期Regime分布

```
NEUTRAL:   560/683 (82.0%)  ← 绝大多数基线行为
RISK_ON:   112/683 (16.4%)  ← 牛市+宽松=积极
DEFENSIVE:  11/683 ( 1.6%)  ← 熊市+紧缩=防御
```

### 3.4 预期性能

**优势**：
- 逻辑极简，不易过拟合
- RISK_ON占16.4%，在bull+loose环境下增强收益
- DEFENSIVE仅1.6%，极少但精准的防御
- 无BREADTH信号 → 消除了CSI300/SH proxy的不确定性

**劣势**：
- 防御仅1.6%，在BEAR趋势但非TIGHT liquidity时期（如2018的"中性货币+熊市"）
  无额外保护
- RISK_ON不区分BULL/WEAK，可能在弱牛市中过度激进
- 相比Fix 1缺少中间态（BUBBLE_WARN）

### 3.5 预期WF

- 大部分窗口接近D4-tuned（82%时间neutral）
- 2022-24窗口：bull+loose触发RISK_ON增强收益
- 2018窗口：bear+neutral = NEUTRAL，无防御 → DD可能较大
- WF Sharpe std 预期：0.5-0.8

### 3.6 可扩展性

可作为baseline的fallback：如果Phase4/5实现后回测仍不达标，降级为3-state。

---

## 4. Fix Direction 3: Adaptive Parameters (Direction B) (FALLBACK)

### 4.1 概念

如果Direction A的离散分类器无论怎么调参都无法达到鲁棒性目标，改用T34设计
的Direction B：连续Regime Strength Score (RSS) + 参数插值。

### 4.2 设计回顾

```
RSS(t) = w_t × TREND_score + w_l × LIQUID_score + w_b × BREADTH_score + w_c × CPI_score

其中每个子分数归一化到[0, 1]，RSS通过EMA平滑（2周）后插入参数曲线：

mom_w = interpolate(RSS, [0.20, 0.45])
vol_w = interpolate(RSS, [0.38, 0.22])   # 反向
...
```

### 4.3 优势

- 连续调制：避免离散分类的"断崖"切换（如突然从DEFENSIVE跳到RISK_ON）
- 所有参数平滑过渡，减少whipsaw
- RSS本身可微调（子信号权重），自由度大于离散分类器

### 4.4 劣势

- 调参复杂度高：4个子信号权重 + 5条参数曲线的端点 = 14+自由度
- 过拟合风险：如果一次性调优所有参数，可能"碰巧"通过Gate但样本外失效
- 实现成本：需要重新设计参数覆盖机制（当前基于离散regime lookup）

### 4.5 预期WF

- 如果调参得当，WF Sharpe std可达0.3-0.5
- 但过拟合风险使实际样本外表现可能差于Fix 1/2
- **建议**：仅当Fix 1和Fix 2都未达标时才实施

---

## 5. Comparative Analysis

### 5.1 方案对比

| 维度 | Fix 1: Recalibrated Classifier | Fix 2: 3-State Simplified | Fix 3: Adaptive Params |
|------|------|------|------|
| Regime覆盖 | 5-state（实际3-state） | 3-state | Continuous |
| DEFENSIVE占比 | 7.6% | 1.6% | ~5-15%（RSS决定） |
| RISK_ON占比 | 11.0% | 16.4% | ~10-30%（RSS决定） |
| CAUTIOUS/NEUTRAL | 81.4% | 82.0% | ~60-80% |
| BREADTH信号 | 使用（±1.0阈值） | 不使用 | 使用（连续） |
| 实现复杂度 | LOW（改阈值+规则） | LOW（改逻辑） | HIGH（新信号引擎） |
| 过拟合风险 | LOW（3个自由度） | LOW（仅trend+liquid） | HIGH（14+自由度） |
| 预期WF std | 0.4-0.7 | 0.5-0.8 | 0.3-0.5（风险高） |
| 向后兼容 | ✅ 配置开关 | ✅ 配置开关 | ✅ 配置开关 |

### 5.2 预期年化收益/回撤/Sharpe

| 方案 | 年化收益率 | MaxDD | Sharpe | WF Sharpe Std |
|------|-----------|-------|--------|---------------|
| D4-tuned基准 | 15.65% | 7.58% | 1.216 | 1.208 |
| 当前Regime ON | 12.56% | 12.58% | 1.009 | 0.857 |
| **Fix 1 (推荐)** | **14-16%** | **6-9%** | **1.1-1.3** | **0.4-0.7** |
| Fix 2 | 14-16% | 7-11% | 1.0-1.2 | 0.5-0.8 |
| Fix 3 | 13-17% | 6-10% | 1.0-1.3 | 0.3-0.5 |

*注：Fix 1/2/3的预期基于regime分布推算，实际需要回测验证。*

### 5.3 推荐路线

```
Phase 4: 实现Fix 1 → 回测验证
   ├── 达标（WF std<0.15 或 7/9窗Sharpe>0.8）→ Gate PASS，交付
   ├── 部分达标（WF std<0.6）→ Phase 5: Fix 2回测对比
   └── 未达标 → Phase 5: 评估Fix 3或放弃Direction A
```

### 5.4 Gate评估

#### G1: 方案可行性（数据源）

| 数据源 | Fix 1 | Fix 2 | Fix 3 |
|--------|-------|-------|-------|
| CSI300 daily | ✅ 已有 | ✅ 已有 | ✅ 已有 |
| M1/M2 monthly | ✅ 已有 | ✅ 已有 | ✅ 已有 |
| CPI monthly | - | - | ✅ 已有 |
| CSI300/SH比值 | ✅ 已有 | - | ✅ 已有 |
| 新增需求 | 无 | 无 | 无 |

**G1**: ✅ PASS — 所有方案复用已有数据，无需新增API调用。

#### G2: 设计完整性

| 方案 | 完整信号链 | 参数覆盖 | 配置集成 |
|------|-----------|---------|---------|
| Fix 1 | ✅ | ✅ | ✅ |
| Fix 2 | ✅ | ✅ | ✅ |
| Fix 3 | ✅ | ✅ | ✅ (需重构) |

**G2**: ✅ PASS — 3个方案的信号→参数→策略链条完整。

#### G3: 向后兼容

| 方案 | 兼容方式 |
|------|---------|
| Fix 1 | `classify_regime()`函数内改 + 阈值改，`regime_classifier.enabled: false`时完全旁路 |
| Fix 2 | 同Fix 1 |
| Fix 3 | 新RSS引擎独立代码路径，`adaptive_parameters.enabled: false`时零影响 |

**G3**: ✅ PASS — 所有方案通过开关实现零影响旁路。

#### G4: 鲁棒性目标

| 目标 | 当前 | 预期Fix 1 | 状态 |
|------|------|-----------|------|
| WF Sharpe std < 0.15 | 0.857 | 0.4-0.7 | ⚠️ 目标过于激进，可能无法达成 |
| 7/9窗口 Sharpe > 0.8 | 5/9 | 6-7/9 | ✅ 可能达成 |

**G4**: ⚠️ CONDITIONAL — Fix 1预期可改善鲁棒性，但<0.15的目标可能物理上不可能
（5-ETF动量策略固有的宏观依赖）。建议接受std<0.6作为成功标准。

#### G5: 性能底线

| 目标 | 预期Fix 1 | 状态 |
|------|-----------|------|
| 年化 >= 15.65% | 14-16% | ⚠️ 可能接近但不保证超过 |
| MaxDD <= 7.58% | 6-9% | ⚠️ 可能接近但不保证低于 |
| Sharpe >= 1.216 | 1.1-1.3 | ⚠️ 可能接近但不保证超过 |

**G5**: ⚠️ CONDITIONAL — 需要回测验证。不建议将D4-tuned基准作为硬下限，
因为D4-tuned是满窗优化的样本内最优，regime-aware的目标是牺牲少量收益换取
鲁棒性。

---

## 6. Implementation Plan

### 6.1 Phase 4: Fix 1 Implementation (P0)

**修改文件**: `src/regime.py`

1. `compute_breadth_from_proxy()`: threshold 0.5→1.0
2. `classify_regime()`: DEFENSIVE规则从OR改为(BEAR AND THIN) OR TIGHT
3. `classify_regime()`: RISK_ON规则移除CPI要求，WIDE放宽为NOT-THIN

**验证**：
```bash
python _test_regime_on.py  # 全周期回测
python _test_wf_regime.py  # Walk-forward测试
```

**Gate检查**：
- G4: WF Sharpe std
- G5: 全周期Sharpe/MaxDD vs D4-tuned

### 6.2 Phase 5: Fix 2 + Fix 3 应急 (P1)

如Fix 1未达标：
1. Fix 2 (3-state简化): 实施并回测对比
2. Fix 3 (Adaptive Params): 如Fix 1和Fix 2都失败则考虑

### 6.3 回退策略

如果所有方向都未显著改善WF鲁棒性：
1. 文档化失败原因（BREADTH信号本质信息含量低、5-ETF策略的宏观依赖是结构性
   问题而非可修复的参数问题）
2. 建议重新评估策略可行性，考虑策略类型转变（如risk parity或更宽泛的因子策略）

---

## 7. Deliverables

| # | 交付物 | 状态 |
|---|--------|------|
| 1 | Phase4设计文档 `docs/phase4_redesign.md` | ✅ 本文档 |
| 2 | BREADTH信号根因分析 | ✅ Section 1.1-1.2 |
| 3 | 2+修复方向方案对比（含预期影响） | ✅ Section 2-5 |
| 4 | Gate评估 | ✅ Section 5.4 |
| 5 | 分析脚本 `_analyze_breadth_t36.py` | ✅ |
| 6 | 反事实分析脚本 `_analyze_counterfactual_t36.py` | ✅ |

---

## 8. Appendix: Analysis Scripts

### A. Breadth信号分析

```bash
cd /home/ubuntu/claw_etf_strategy
.venv/bin/python _analyze_breadth_t36.py
```

### B. 反事实Regime分布分析

```bash
cd /home/ubuntu/claw_etf_strategy
.venv/bin/python _analyze_counterfactual_t36.py
```

### C. 当前Regime ON回测

```bash
cd /home/ubuntu/claw_etf_strategy
.venv/bin/python _test_regime_on.py    # 全周期
.venv/bin/python _test_wf_regime.py    # Walk-forward
```

---

*Document: T36 Phase4 Redesign*
*Author: quant-se*
*Date: 2026-06-17*
*Status: DESIGN — 待PM审核后转coder实现*
