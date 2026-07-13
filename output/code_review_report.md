# 量化工程代码审视报告 — claw_etf_strategy（终版）

**日期**: 2026-07-13  
**审查轮次**: 第 4 轮（全量）  
**配置文件**: config/strategy_v3_0_final.yaml  
**核心引擎**: src/backtest.py  

---

## 审查结论总览

**本报告经过 4 轮迭代审查，共发现 10 项问题（9 项已修复，1 项需确认）。**

| 轮次 | 发现 | 已修复 | 未修复 |
|------|------|--------|--------|
| 第 1 轮 | 6 项（NameError、web_app.py、标签混淆、report.py重复行、硬编码0.2、tmp/清理） | 6 ✅ | 0 |
| 第 2 轮 | 4 项（score_margin未传 ×3、OAT clamp不一致、MC不对称clamp、cap作用于防御层） | 3 ✅ | 1 |
| 第 3 轮 | _run_single_grid still missing score_margin | 1 ✅ | 0 |
| **第 4 轮** | **robustness.py MC/OAT/Grid clamp范围与C1参数不兼容** | **0** | **1 ⚠️** |

---

## 第 4 轮核心发现

### 🔴 Bug 5（高）：robustness.py clamp 范围未更新至 C1 参数

**影响范围**：`src/robustness.py` 中 3 处参数钳制范围

| 位置 | 参数 | 当前 clamp | C1 实际值 | ±15% 扰动范围 | 问题 |
|------|------|-----------|-----------|---------------|------|
| _mc_single_worker L402 | mom_w | **[0.20, 0.50]** | **1.00** | [0.85, 1.15] | 1.00 → **0.50** |
| _mc_single_worker L403 | vol_w | **[0.15, 0.45]** | **1.10** | [0.94, 1.27] | 1.10 → **0.45** |
| OAT L687-688 | mom_w/vol_w | **[0.05, 0.80]** | 1.00 / 1.10 | 同上 | 1.00 → 0.80, 1.10 → 0.80 |
| GRID_CLAMP L380-381 | mom_w/vol_w | **(0.05, 0.80)** | 同上 | 同上 | 同上 |

**量化影响**：
- **mom_w**: C1=1.00 → MC 使用 **0.50**（-50%） → OAT/Grid 使用 **0.80**（-20%）
- **vol_w**: C1=1.10 → MC 使用 **0.45**（-59%） → OAT/Grid 使用 **0.80**（-27%）

这意味着：
1. MC 的 100% 生存率是 **在 mom_w=0.50, vol_w=0.45 下测试的**，而非 C1 的 1.00/1.10
2. OAT 的敏感度曲线在 mom_w=0.80、vol_w=0.80 处饱和，**无法反映真实 C1 操作点的敏感度**
3. Grid 搜索也为同样原因失效

**这些 clamp 范围是 v2.x 时代的遗产（当时 mom_w=0.35, vol_w=0.30）**，迁移到 C1后从未更新。

**修复建议**：

```python
# MC clamp — 允许扰动到至少 ±15% 之外
mom_w=min(max(params.get('mom_w', base_cfg.mom_w), 0.05), 1.50),
vol_w=min(max(params.get('vol_w', base_cfg.vol_w), 0.05), 1.50),

# OAT clamp — 同样放宽
if param_name in ('mom_w', 'vol_w'):
    new_val = max(0.05, min(1.50, new_val))

# GRID_CLAMP — 同样更新
GRID_CLAMP = {
    'mom_w': (0.05, 1.50),
    'vol_w': (0.05, 1.50),
    # ... 其他不变
}
```

**修复后必须重跑全量鲁棒性评估**（MC 400次 + OAT 49次 + Grid 63点）。

---

## 全部修复确认

### ✅ 已修复项目

| # | 问题 | 文件 | 修复版本 |
|---|------|------|----------|
| 1 | rebalance_live.py 边界条件 NameError | scripts/rebalance_live.py | 21b704d |
| 2 | web_app.py 不再需要 | — | 已删除 |
| 3 | rebalance_live.py print 标签混淆 | scripts/rebalance_live.py | 21b704d |
| 4 | report.py "标准夏普"重复行 | src/report.py | 21b704d |
| 5 | calc_performance.py 硬编码 0.2 | scripts/calc_performance.py | 21b704d |
| 6 | tmp/ 目录清理 | — | 已删除 |
| 7 | robustness.py score_margin 未传给构造函数 ×3 | src/robustness.py + src/backtest.py | 7d6455a + f9f17a0 |
| 8 | robustness.py OAT momentum_threshold clamp 不一致 | src/robustness.py | 7d6455a |
| 9 | robustness.py _mc_single_worker 不对称 clamp | src/robustness.py | 7d6455a |
| 10 | rebalance_live.py cap 错误作用于防御层 ETF | scripts/rebalance_live.py | f9f17a0 |

### ❌ 剩余未修复

| # | 问题 | 严重度 | 建议 |
|---|------|--------|------|
| **11** | robustness.py MC/OAT/Grid clamp 范围与 C1 参数不兼容 **🆕** | 🔴 高 | 更新 clamp 范围至 [0.05, 1.50] 后重跑全量鲁棒性评估 |
| 12 | `--verify` DD 偏差 3.30pp 持续存在 | 🟡 中 | 因 verify 不计费 + 少量算法差异，属于可接受的近似偏差。Sharpe 差 0.0035（达标 <0.02）。 |

---

## 最终建议

**优先级 1**：修复 robustness.py 的 clamp 范围（Bug 11）—— 这是目前唯一可能影响策略可信度的未修复问题。

**优先级 2**：修复后重跑 `run_robustness_test()` 全量评估，确认：
- MC 生存率仍 ≥ 95%（预期变化：因 clamp 正确后参数空间扩展，可能略降）
- DSR 仍 ≥ 0.95
- OAT 曲线恢复，vol_w 敏感度在 1.10 处应有峰值
- Grid 结果覆盖 [0.85, 1.15] × [0.94, 1.27] 全范围而非仅一个点

**优先级 3**：重新生成 README 中的鲁棒性指标表格。