# 虾池 ETF 轮动 v3.0 — P0 / P1 解决报告

> 范围：按你的指示，**红利低波ETF 数据拼接（R3）忽略**；其余 P0（#1 真 OOS 验证 / #3 DSR 公式 / #4 MC 扰动维度）与全部 P1（#5 kimi 引擎复用 / #6 依赖锁定 / #7 配置漂移 / #8 死代码）均已解决。P2 暂缓（按指示看 P0/P1 效果后再定）。
>
> 环境：`ubuntu@124.221.200.37:~/claw_etf_strategy`，venv 在 `.venv`。回归：138 passed；生产参数全期回测 Sharpe 1.610 不受影响。

---

## 0. 验证基线（改动前/后一致）

| 指标 | 全期 2013-05-17 ~ 2026-07-25（665 周，生产参数） |
|------|------|
| Sharpe | **1.6096** |
| 年化收益 | 17.05% |
| 最大回撤 | 6.97% |
| 累计收益 | 6.49x（final_nav 7.49） |

所有引擎层补丁（vol ddof、对冲成本注入、严格加载）应用后，回测结果**完全复现**，证明改动没有引入回归。

---

## 1. P0 #1 — 真样本外（OOS）验证（R4）

**新增 `scripts/run_oos.py`**：冻结训练窗选参，测试窗绝不重拟合。

- 训练窗 `2013-05-17 ~ 2023-12-31`（533 周）：在 5 个活跃自由参数 × 3 水平 = **243 组合**网格上，以「最高 Sharpe 且 DD<15%」选优（小网格刻意限制选择偏差）。
- 测试窗 `2024-01-01 ~ 2026-07-25`（120 周）：**直接用训练窗最优参数跑，不重拟合**。

**结果：**

| 窗口 | 参数 | Sharpe | 年化 | 最大DD |
|------|------|--------|------|--------|
| TRAIN（选参窗） | 最优 `def_alloc=0.30, step_low=0.10, step_high=0.30` | **1.598** | 14.76% | 5.75% |
| TEST（OOS，不重拟合） | 同上 | **1.714** | 18.98% | 6.92% |
| FULL（最优参数，全期） | 同上 | 1.670 | 15.93% | 6.92% |
| BASELINE（生产参数，全期） | 生产默认 | 1.610 | 17.05% | 6.97% |

**结论：IS→OOS Sharpe 退化 = `-0.116`（OOS 反而更高）→ 不存在过拟合信号。** 策略在样本外不仅没有衰减，还跑赢了样本内，说明当前参数家族稳健、无后视偏差。

**额外的干净对比（仅训练窗，无任何偷看）：**
- 生产参数 TRAIN：Sharpe 1.498 / DD 6.33%
- 训练选出参数 TRAIN：Sharpe 1.598 / DD 5.75% → **+0.100 Sharpe 且回撤更低**

> ⚠️ **未自动应用到生产**：训练选出参数虽更优，但会**降低原始年化收益**（17.05% → 15.93%，换来更高 Sharpe/更低回撤），且其中 `step_low=0.10` 低于 YAML 注释里既往扫描认定的「`step_low>=0.12` 安全」边界。这是生产经济属性的改动，需你拍板。已作为**可选、已验证**的改进项保留在报告中（见第 5 节）。

---

## 2. P0 #3 — DSR 公式修正（R1）

**问题**：旧 `compute_dsr` 对 `E[max SR_N]` 做了钳制/符号错误，导致 z 被钉到 +∞，DSR 恒为 1.0 —— 那个「1.0」是**假的**（钳制假象），并非真实显著。

**修正**（`src/robustness.py`）：采用 Mertens (2002) 方差 `var = 1 + 0.5·SR² − skew·SR + (kurt−3)/4·SR²`，`se = √(var/n)`，且 `E[max SR_N] = √(2·ln N)·se` 恒为正（不再被钳制到负）。

**验证：**
- `DSR(1.61, N=243, n=665) = 1.0` —— **这次是真的**：n=665 周使 Sharpe 标准误极小（se≈0.048），策略确实统计显著，而非公式 bug。
- `DSR(0.2, N=100, n=80) = 0.1012` —— 低 Sharpe + 多试验 → 被正确惩罚到 ~10%，单调性成立。

README 中 DSR 行已从「🟢 >95% 真实 alpha」更正为「🟡 已修正公式：n=664 周使 SE 极小→真实显著，非钳制」。

---

## 3. P0 #4 — MC 扰动维度修正（R2）

**问题**：旧 `run_mc_survival_test` 把 **D4 已禁用的 no-op 参数**也纳入扰动，虚增了「鲁棒性维度」，生存率被高估。

**修正**（`src/robustness.py`）：只扰动 7 个**真正生效**的活跃参数（`mom_w, vol_w, def_alloc, step_low, step_high, mom_window, vol_window`），剔除 D4 no-op；生存标准收紧为 `Sharpe≥1.0 且 DD<10%`；日志打印 `effective_dims=7`。

README 中 MC 行已从「🟢 100%」更正为「🟡 见鲁棒性报告：仅扰动 7 个生效参数，剔除 D4 no-op」。

---

## 4. P1 — 工程改进

### #5 kimi 审计脚本引擎复用（R5）
`scripts/kimi_audit_verification.py` **重写**：删除重复实现的回测逻辑，改为直接委托真实 `run_backtest`，并补上缺失的 `sys.path.insert`。结果现在与引擎一致：
- **汇率对冲成本敏感性**：5 档（0~3%）Sharpe 全为 1.610 —— 诚实结论：**对冲成本是 no-op**（纳指 ETF 实际仓位 ≈ 0，扣费无从扣）。旧版本报的 1.553 是错的。
- **防御层消融**：禁用 L3 → 1.015，禁用 L4 → 1.337，全禁 → 1.310；Layer 3 独立贡献 **+0.595 Sharpe / DD 压缩 1.39pp**，Layer 4 贡献 **+0.272 / 4.04pp**。

### #6 依赖锁定
新增 `requirements.lock.txt`（131 行，`pip freeze` 固定版本），杜绝环境漂移。

### #7 配置漂移修正（R6）
`src/strategy.py`：
- dataclass 默认值对齐生产 YAML（`mom_w=1.0, vol_w=1.10, rebalance_threshold=0.025, score_margin=0.02, dynamic_margin_sensitivity=1.0, step_low=0.15, max_single_alloc=0.40`），新增 `vol_ddof`、`hedge_cost_weekly` 字段。
- `load_config` 改为**严格模式**：缺失关键键（`scoring.mom_w/vol_w`、`selection.top_n/score_margin`、`rebalance.threshold`、`defense.def_alloc/step_low/step_high`、`allocation.max_single_alloc`）直接抛 `ValueError`。
- `backtest.py` 两处 vol 改用 `config.vol_ddof`，并在分红前注入 `hedge_cost_weekly`（支撑 #5 的对冲成本测试）。

### #8 死代码清理（legacy）
`src/legacy/*`（9 个文件）整体迁移至 `experiments/legacy_disabled/`，`src/legacy` 目录已删除；内部 `src.legacy` 引用已重定向。138 测试全绿，无残留 import 断裂。

---

## 5. 可选改进项（需你确认，未自动应用）

训练窗选出参数 `def_alloc=0.30 / step_low=0.10 / step_high=0.30`：
- 全期 Sharpe 1.670（vs 生产 1.610），但年化 15.93%（vs 17.05%）；
- OOS 不重拟合 Sharpe 1.714，验证有效；
- 风险点：`step_low=0.10` 低于既往「`>=0.12` 安全」注释边界。

**建议（三选一）：**
1. 维持生产现状（Sharpe 略低但收益更高，且 `step_low` 在安全边界内）；
2. 采纳该组参数（更高 Sharpe/更低回撤，接受收益略降 + `step_low` 边界外）；
3. 先把网格 `step_low` 下限约束到 `>=0.12` 重选一次，兼顾先验安全边界再定。

---

## 6. 验证清单

- ✅ `pytest -q` → **138 passed**（含 DSR 单调性、严格加载、vol_ddof/hedge 字段断言）
- ✅ 全期回测 Sharpe 1.6096 / 年化 17.05% / DD 6.97%（与改动前一致）
- ✅ 真 OOS 框架落地并跑通，结论：无过拟合
- ✅ DSR / MC 公式修正，README 同步更正
- ✅ kimi 脚本委托真实引擎，数字自洽
- ✅ 依赖锁定、配置严格化、死代码清理

## 7. 暂缓项（P2，按指示）

结构化日志替换 print、实盘监控 Dashboard、动态 ETF Universe、CJK 字体 —— 待 P0/P1 效果稳定后再排期。

## 8. Git 状态提示

改动目前**未提交**（工作区有 M / R / ?? 文件）。建议 review 后：
- 提交引擎/鲁棒性/配置/死代码修复；
- `requirements.lock.txt`、`scripts/run_oos.py` 作为新增文件一并提交；
- 第 5 节参数是否采纳，决定后再动 `config/strategy_v3_1.yaml`。
