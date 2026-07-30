# 消融实验：v4.3 四层防御结构 vs 朴素动量（任务ID 4）

**日期**：2026-07-30 ｜ **基线**：`config/strategy_v4_3.yaml`（生产）｜ **数据**：2013-05-17 ~ 最新（662 个回测周）
**回测引擎**：`src/backtest.run_backtest`（未做任何修改，全部变体经 `scripts/run_backtest.py` 同一路径可复现）
**运行方式**：`.venv/bin/python scripts/_exp_ablation_run.py`（新增只读 runner，逐变体调用 `load_config` + `run_backtest`，结果落盘 `output/experiments/ablation_results.json`）

基线复现校验：Sharpe **1.488** / 年化 **14.52%** / MaxDD **5.84%** / Calmar **2.49** / 周胜率 **60.7%** —— 与任务书给定基线逐位一致。

---

## 1. 变体实现方式（以代码实际行为为准）

精读 `src/strategy.py::load_config`、`src/backtest.py` 主循环与 `src/engine_core.py` 后确认：**全部消融均可用纯配置实现，无需 monkey-patch**。各机制的等效关闭手段与代码依据：

| 机制 | 关闭手段 | 代码依据 |
|------|----------|----------|
| Layer3 防御层 | `def_alloc=0` **且** `max_def=0` | `calculate_defense_ratio` 三段式：vol<step_low 返回 base=0；vol>step_high 返回 max_def=0；区间内线性插值 0→0 恒为 0。`step_low/step_high` 为 load_config 严格校验必需键，保留原值（数值上不再起作用） |
| Layer3.5 crisis boost | `max_boost=0` **且** `corr_threshold=1.01` | `compute_crisis_boost`：无 enabled 开关。`corr_threshold=1.01` 使 `max_pair_corr > threshold` 永假（\|corr\|≤1）；`max_boost=0` 双保险使 `min(x, 0)=0`；引擎 `if crisis_boost > 0` 分支永不进入 |
| 止损 | `stop_loss=1.0`（事实禁用） | `check_stop_loss`：`(peak-nav)/peak >= 1.0` 要求 NAV 归零，不可能发生。无 enabled 开关。注意止损触发的效果是 `def_ratio = max(def_ratio, config.max_def)`，在 max_def=0 的变体中即使触发也无效果，`stop_loss=1.0` 系双保险 |
| Layer2 inv_vol | `inv_vol_allocation.enabled: false` | 引擎 `off_weights=None` → `allocate()` 回退进攻等权路径。`enabled` 为必需键（审计 M2），显式写 false |
| Layer1 粘性 | `score_margin=0` **且** `dynamic_margin_sensitivity=0` | 引擎粘性判断 `if config.score_margin > 0 or config.dynamic_margin_sensitivity > 0 or snr_adaptive` 全假即跳过；`trend_confirm_weeks=0` 本就关闭 |
| 40% 单资产上限 | `max_single_alloc=1.0` | `allocate()` 中 `apply_cap and max_single_alloc < 1.0` 为假即不套帽。**关键交互**：该上限的溢出去向是防御层（`overflow_to_defense_only=true`），在"无防御层"变体（A1/A2/A2b/A3）中若保留 0.4，等权 0.5+0.5 的进攻仓会被强制挤出 2×10%=20% 进防御 ETF，事实上重建了一个静态防御层、污染消融。故将该上限视为防御结构的一部分，在无防御变体中一并关闭；A4/A5/A6 保持 0.4 不动 |

其他说明：
- `factors` 段（mom6 / tapered_vol14）在**所有**变体中与 v4.3 逐字段一致 → 预热窗口 `start_idx=max(14,6)` 相同 → 所有变体回测区间严格同为 662 周，指标可直接横向比较。
- `hongli_formula`/`hongli_ratio` 只影响防御层内部拆分，防御为 0 时无效，保留原值。
- `rebalance`（threshold=0.025, fee=0.5bp）在所有变体保持不变，非本次消融对象。
- A2 的口径：任务书"同 A1 但保留 vol_w=1.1 评分（即 Layer1 完整）"两种读法有歧义——"Layer1 完整"含 score_margin 粘性，"仅保留评分"不含。本实验按 **A2 = Layer1 完整（含粘性）** 执行，另补充 **A2b = 仅评分、无粘性**，把 Layer1 内部再拆成"波动率惩罚项"与"粘性"两个可加部分。

变体配置文件（均基于 v4.3 复制修改，改动点写入各文件 `strategy.note`）：

| 变体 | 配置文件 | 相对 v4.3 的改动 |
|------|----------|------------------|
| A1 | `config/experiments/ablation_a1_pure_momentum.yaml` | vol_w=0；粘性关；防御关；crisis 关；止损关；inv_vol 关；上限关 |
| A2b | `config/experiments/ablation_a2b_score_no_margin.yaml` | 同 A1 但 vol_w=1.1 |
| A2 | `config/experiments/ablation_a2_layer1_full.yaml` | 同 A2b 但恢复粘性（score_margin=0.02, dms=1.0） |
| A3 | `config/experiments/ablation_a3_no_defense.yaml` | 同 A2 但恢复 inv_vol（enabled=true） |
| A4 | `config/experiments/ablation_a4_no_crisis_boost.yaml` | 仅关 crisis boost，其余=v4.3 |
| A5 | `config/experiments/ablation_a5_no_stoploss.yaml` | 仅 stop_loss=1.0，其余=v4.3 |
| A6 | `config/experiments/ablation_a6_no_invvol.yaml` | 仅 inv_vol enabled=false，其余=v4.3 |

---

## 2. 完整对比表（全期 2013-05 ~ 最新，662 周）

| 变体 | Sharpe | 年化 | MaxDD | Calmar | 周胜率 | 年化波动 | 调仓次数 | 防御周数 | 止损触发 |
|------|--------|------|-------|--------|--------|----------|----------|----------|----------|
| **v4.3 基线** | **1.488** | **14.52%** | **5.84%** | **2.49** | **60.7%** | 7.64% | 377 | 647 | 0 |
| A1 纯动量等权 | 0.814 | 14.07% | 27.70% | 0.51 | 59.4% | 14.39% | 172 | 0 | 0 |
| A2b 动量-波动评分（无粘性） | 1.273 | 19.94% | 13.00% | 1.53 | 60.3% | 13.01% | 104 | 0 | 0 |
| A2 Layer1 完整（评分+粘性） | 1.182 | 18.70% | 13.00% | 1.44 | 59.7% | 13.13% | 59 | 0 | 0 |
| A3 Layer1+Layer2（无防御） | 1.256 | 18.80% | 13.29% | 1.41 | 59.7% | 12.35% | 267 | 0 | 0 |
| A4 完整减 crisis boost | 1.484 | 14.53% | 5.84% | 2.49 | 60.7% | 7.67% | 367 | 647 | 0 |
| A5 完整减止损 | 1.488 | 14.52% | 5.84% | 2.49 | 60.7% | 7.64% | 377 | 647 | 0 |
| A6 完整减 inv_vol | 1.396 | 14.46% | 7.07% | 2.04 | 60.6% | 8.13% | 307 | 647 | 0 |

原始数据：`output/experiments/ablation_results.json`。

---

## 3. 各层边际贡献

两种视角：**逐层叠加**（A1→A2b→A2→A3→+防御→v4.3）与**从完整版单独拆除**（v4.3 vs A4/A5/A6）。

### 3.1 叠加视角（自下而上）

| 增量 | 对比 | ΔSharpe | Δ年化 | ΔMaxDD | ΔCalmar | 解读 |
|------|------|---------|-------|--------|---------|------|
| 波动率惩罚项（vol_w=1.1） | A2b − A1 | **+0.459** | +5.87pp | **−14.70pp** | +1.02 | **单项贡献最大**。纯动量会追高波动资产；vol 惩罚同时提升收益并砍掉一半以上回撤 |
| score_margin 粘性 | A2 − A2b | −0.091 | −1.24pp | ±0 | −0.09 | **样本内为负贡献**，但调仓次数 104→59（−43%）。粘性的价值在抗噪/降换手/OOS 稳健，不体现在样本内点估计 |
| Layer2 inv_vol（无防御环境） | A3 − A2 | +0.074 | +0.10pp | +0.29pp | −0.03 | 无防御层时作用有限 |
| Layer3+3.5 防御结构 | v4.3 − A3 | **+0.232** | **−4.28pp** | **−7.45pp** | +1.08 | 防御结构的本质：**用约 4.3pp 年化收益购买 7.45pp 回撤削减**，Sharpe/Calmar 大幅改善 |

### 3.2 拆除视角（自上而下，单独关一个机制）

| 机制 | 对比 | ΔSharpe | Δ年化 | ΔMaxDD | ΔCalmar | 边际贡献结论 |
|------|------|---------|-------|--------|---------|--------------|
| crisis boost（Layer3.5） | v4.3 − A4 | +0.004 | −0.01pp | 0.00pp | 0.00 | **几乎为零**。防御周数不变（647），仅在个别周把 def_ratio 边际抬高一点。13 年样本内该层近似冗余 |
| 止损 | v4.3 − A5 | **0（逐周 NAV 完全一致）** | 0 | 0 | 0 | **严格为零**，见 3.3 |
| inv_vol（Layer2） | v4.3 − A6 | +0.092 | +0.06pp | −1.23pp | +0.45 | 在完整体系内贡献明确：回撤 7.07%→5.84%，Calmar 2.04→2.49，几乎不花收益代价 |

注意 inv_vol 的贡献存在**机制间交互**：单独看（A3−A2）只有 +0.074 Sharpe 且回撤略增，但在有防御层的完整体系内（v4.3−A6）它与 40% 上限、防御层配合，回撤削减效果被放大。消融差值不可简单线性相加。

### 3.3 止损从未触发 —— 猜想证实

三重证据：

1. **A5 与基线逐周 NAV 完全一致**：`max |NAV_diff| = 0.0`（662 周逐位相同，非近似）。
2. **直接计数**：`scripts/_exp_ablation_run.py` 对 `run_backtest` 返回的 `weekly_records[].in_stop_loss` 统计 False→True 跳变，**全部 8 个变体触发次数 = 0，止损状态周数 = 0**（`run_backtest.py` 的年度分解表"止损周数"列 14 年全为 0，口径一致）。
3. **数学蕴含**：引擎在每周开头用当前 nav/peak 判断 `(peak-nav)/peak >= 0.08`，与回撤序列同源；基线全期 MaxDD=5.84% < 8%，故止损条件在任何一周都不可能成立。**防御层把回撤压到止损阈值之下，使止损成为永不激活的"死保险丝"**——它无历史贡献，但作为尾部风险兜底（未来若出现 >8% 回撤的新形态危机）保留成本为零。

---

## 4. 结论："复杂的四层防御结构相对朴素动量到底贡献了多少"

**总账（v4.3 vs A1 纯动量）：Sharpe 0.814 → 1.488（+83%），MaxDD 27.70% → 5.84%（削减 79%），Calmar 0.51 → 2.49（×4.9），代价是年化仅从 14.07% → 14.52%（基本持平）。** 即：全部复杂度加起来，没有用来赚更多钱，而是把同样的钱赚得**平稳了近 5 倍**（年化波动 14.39% → 7.64%）。

分解到层：

1. **最有价值的不是"防御结构"，而是评分里的波动率惩罚项**（vol_w=1.1）：一项就贡献 +0.459 Sharpe、−14.7pp MaxDD，占 v4.3 相对 A1 全部 Sharpe 改善（+0.674）的约 2/3。它是隐性风控——在选基环节就回避高波动资产，比事后防御便宜得多。
2. **Layer3 防御层是第二大贡献者**（+0.232 Sharpe、−7.45pp MaxDD），且是唯一"花钱"的层：付出约 4.3pp 年化。是否值得取决于效用函数——对 Calmar/低回撤目标（v4.3 的设计目标）明确值得；对纯收益最大化则 A3（年化 18.80%、MaxDD 13.29%）更优。
3. **Layer2 inv_vol 是廉价的锦上添花**：完整体系内 +0.092 Sharpe、−1.23pp MaxDD，几乎零收益代价，保留无疑。
4. **Layer3.5 crisis boost ≈ 冗余**（ΔSharpe 0.004），**止损 = 严格零贡献**（13 年 0 次触发）。这两层在历史样本内可以整体删除而结果几乎不变/完全不变；保留理由只能是对样本外尾部情景的保险，而非历史绩效。
5. **score_margin 粘性样本内是负贡献**（−0.091 Sharpe），其真实作用是把调仓次数砍掉 43%（104→59 次/无防御口径），降低实盘摩擦与噪声敏感性；评价它应看 OOS/成本敏感性实验而非本表。

一句话回答：**四层防御结构对"赚多少"的贡献≈0，对"怎么赚"的贡献是决定性的（回撤 −79%、Sharpe +83%）；其中真正干活的是 vol 评分惩罚 + Layer3 防御层 + inv_vol 三件套，crisis boost 与止损在 13 年历史内是纯装饰性保险。**

---

## 附：复现命令

```bash
# 单变体
.venv/bin/python scripts/run_backtest.py --config config/experiments/ablation_a1_pure_momentum.yaml --output output/experiments/ablation_report_a1.md --no-charts
# 全套（含止损触发统计与 A5-基线 NAV 一致性校验）
.venv/bin/python scripts/_exp_ablation_run.py
```
