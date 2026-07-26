# 代码/策略层 Review — P0/P1 提交后的专门复查

> 基于 commit `c4bea99`（P0/P1 修复）的代码层 + 策略层专门 review。
> 之前只验证了「测试通过 + 回测数字复现」，未逐行 review 改动本身。本次逐块审了
> `src/robustness.py` / `src/strategy.py` / `src/backtest.py` / `scripts/kimi_audit_verification.py`
> / `scripts/run_oos.py` / `tests/*`，并对两处怀疑做了**实证诊断**（见下方 DIAG 1/2）。

---

## 一、严重问题（High）

### H1. Kimi 验证脚本的 L3 消融是错的，结论夸大 3.25 倍（已实证）

- **位置**：`scripts/kimi_audit_verification.py` `run_ablation(disable_layer3=True)`
- **问题**：关 L3 时设 `step_low=0, step_high=0, crisis_corr_max_boost=0`。
  但 `calculate_defense_ratio`（strategy.py:532）逻辑是：
  - `nasdaq_vol < step_low` → 返回 base
  - `nasdaq_vol > step_high` → 返回 max_def
  - 之间 → 线性插值
  由于 `nasdaq_vol ≥ 0` 永远 `> 0 = step_high`，**每周期都走 `return max_def`（95%）**——
  这不是「关闭 L3」，而是把防御**永远钉在最大值**。正确关 L3 应让防御恒等于 base
  （`step_low=+inf` 使首分支恒成立 + `crisis_corr_max_boost=0`）。
- **实证（DIAG 2）**：

  | 配置 | Sharpe | DD |
  |---|---|---|
  | FULL (L3+L4) | 1.6096 | 6.97% |
  | L3-off（脚本当前写法） | 1.0148 | 8.36% |
  | **L3-off（正确写法）** | **1.4692** | 7.59% |
  | L4-off（干净） | 1.3373 | 11.01% |

  - 脚本报 `L3 contrib = +0.595`（= FULL − 错误baseline），**标签错、夸大 3.25 倍**。
  - **真实 L3（vol/crisis 缩放）贡献 = +0.140**（= FULL − 正确baseline）。
  - L4（base DefAlloc）贡献 = +0.272，与脚本一致，**这部分是对的**。
- **影响**：之前 `P0_P1_RESOLUTION.md` 里「Layer 3 是策略风险控制核心贡献者（+0.595）」
  结论错误，需更正。
- **修复**：`run_ablation` 中 L3 关法改为
  `kw['step_low'] = float('inf'); kw['crisis_corr_max_boost'] = 0.0`
  （不要动 step_high，保持 `> step_low` 避免 0/0），并修正结论文案。

---

## 二、中等问题（Medium）

### M1. `vol_ddof` / `hedge_cost_weekly` 是死字段，引擎从不消费（已实证）

- **位置**：`StrategyConfig` 声明了 `vol_ddof`(strategy.py:193) 与 `hedge_cost_weekly`(194)，
  但 `src/backtest.py` 的 `run_backtest` **零引用**这两个字段（grep 确认）。
- **实证（DIAG 1）**：`run_backtest(cfg, vol_ddof=0)` 与 `run_backtest(cfg, vol_ddof=1)`
  **结果完全相同**（Sharpe 1.6096 / 17.05% / 6.97%）。对冲同理。
- **影响**：
  1. 这是**半成品功能**——config 有字段、文档当功能讲，引擎没接线。
  2. Kimi 脚本的 `run_with_ddof` 设了这两个字段但引擎忽略 → **ddof 测试和对冲测试是空转**，
     两次结果必然相同，得出「无影响」是**假阳性**。我之前在 resolution 报告里写的
     「ddof delta=0.000」「hedge 是 no-op（纳指仓位≈0）」均不成立——前者是功能未接线、
     后者未真正测量。
  3. **ddof=1 是否影响策略，目前根本没被验证过**（引擎不支持），这是个开放问题而非「已证无影响」。
- **修复（二选一）**：
  - (a) 真正接线：在 `backtest.py` 两处 `np.std(..., ddof=0)` 改为 `ddof=config.vol_ddof`；
    在周收益合成处加 `if config.hedge_cost_weekly>0: wret -= alloc[NASDAQ_IDX]*config.hedge_cost_weekly`
    （注意需确认 `NASDAQ_IDX` 在该作用域可用）。
  - (b) 若暂不实现，则从 dataclass 与文档删除这两个字段，避免「假功能」。
  - 推荐 (a) 并重新跑 Kimi ddof/hedge 验证，得到真实结论。

### M2. 严格模式 `_critical` 漏了 `dynamic_margin_sensitivity`

- **位置**：`src/strategy.py` `load_config` 的 `_critical` 字典（约 300 行）。
- **问题**：`dynamic_margin_sensitivity` 用 `.get('dynamic_margin_sensitivity', 0.0)` 兜底
  （dataclass 默认是 `1.0`），但**不在 `_critical` 校验列表**里。若 YAML 删掉该 key，
  会**静默变成 0.0 → 关掉动态 margin**，而 strict-mode 不报错。
  当前 YAML 有该 key（=1.0），且 `test_yaml_populates_all_known_fields` 断言 `==1.0` 能抓到，
  但那是测试兜底，**load_config 自身的严格模式未覆盖它**，与 P1 #7「杜绝静默回退」的初衷不符。
- **修复**：把 `'selection': ['top_n', 'score_margin', 'dynamic_margin_sensitivity']` 加入 `_critical`，
  兜底值从 `0.0` 改为 `1.0`（与 dataclass 默认一致）。

---

## 三、低等问题（Low / 方法论）

### L1. DSR 的 `n` 取 `n_obs - 1` 而非 `n_obs`

- **位置**：`compute_dsr` 第 158 行 `n = max(int(n_obs) - 1, 1)`。
- **问题**：B&LdP(2014) 的 SE 公式是 `sqrt(variance / n)`，`n` = 观测数（即 `n_obs`）。
  用 `n_obs-1` 是多余的有限样本修正。n=665 时影响可忽略（se 差 0.05%），但口径不标准。
- **修复**：改为 `n = max(int(n_obs), 1)`；或加注释说明为何 -1。

### L2. DSR 里 `euler` 变量定义未使用（死代码）

- `compute_dsr` 内 `euler = 0.5772156649` 从未使用，删掉。

### L3. MC 生存测试只扰动 7 个活跃参数，覆盖有缺口

- `run_mc_survival_test` 的 `active_params` 含 `mom_w/vol_w/def_alloc/step_low/step_high/
  mom_window/vol_window`。但 `score_margin / rebalance_threshold / max_single_alloc /
  dynamic_margin_sensitivity / top_n` 同样是**活跃自由参数**却没扰动 → MC 低估了参数敏感度。
- **建议**：把上述也纳入 active_params（注意边界范围），或显式说明 MC 只评估「权重/防御类」参数。

### L4. MC 生存测试用的是全期（in-sample）回测

- `_mc_single_worker` 调 `run_backtest(cfg)`（无日期范围）→ 用 config 默认全期（含测试窗）。
  严格说 MC 鲁棒性是 in-sample 的。可接受，但报告时应注明「全期 in-sample 鲁棒性」。

### L5. OOS 网格的 243-选-1 选择偏差

- `run_oos.py` 在训练窗 243 组合里取 Sharpe 最高者，该 IS best 是「243 个里的最大值」，
  本身向上偏。测试窗 OOS(1.714) ≈ IS(1.598) 仍是很强的「无过拟合」证据，但 IS 数偏乐观。
- **建议**：对 IS best 用 DSR(N=243) 做多重检验矫正，或再加一层 holdout 切分。低优先级。

### L6. `step_low == step_high` 的潜在 0/0 退化

- `calculate_defense_ratio` 当 `step_low == step_high` 且 vol 恰落在该点时，
  插值分支 `(vol-step_low)/(step_high-step_low)` = 0/0 → NaN。当前 OOS 网格
  step_low∈[0.10,0.20]、step_high∈[0.30,0.40] 不会出现；但 L3 ablation 脚本误触（已证）。
- **建议**：`load_config` 或 `calculate_defense_ratio` 入口加 `assert step_low < step_high`。

### L7. `P0_P1_RESOLUTION.md` 的 Kimi 段含已被证伪的结论

- 该文档「Kimi 审计验证」一节写「ddof delta=0.000」「hedge 是 no-op（纳指仓位≈0）」
  「L3 贡献 +0.595」，现均不成立/需更正（见 H1、M1）。文档需同步修订。

---

## 四、策略层观察（超出本次改动的存量问题）

1. **大量 disabled 功能字段**：config 有数百字段（softmax / regime / d1 / constituent_signals /
   market_state 止损 / 三层止损 等）大多默认关闭。属历史包袱，建议要么清理、要么在 README 标注
   「哪些真正激活」，避免读者误以为都生效。
2. **`vol_ddof`/`hedge_cost_weekly` 这类「看起来能用但没接线」的字段最危险**——比纯 dead code 更隐蔽，
   因为测试/文档会默认它工作。建议全仓扫一遍 config 字段 ↔ 引擎消费的对应关系（可加一个
   `test_config_consumed` 测试：断言每个非 disabled 字段在引擎里至少被读一次）。
3. **OOS/MC 都是单资产池（5 只 ETF）内的验证**，未做 universe 层面的样本外（这正是你 P2 里的
   「动态 ETF Universe」待办）。当前结论「无过拟合」仅限参数层面，不含标的筛选层面。

---

## 五、修复完成（2026-07-26，已全部修复并提交）

用户确认「全都修复」后，所有 H/M/L 发现均已在代码层修复并实证验证。

| 级别 | 项 | 修复 | 验证 |
|---|---|---|---|
| High | H1 L3 消融写法 + 结论更正 | kimi `run_ablation` 改 `step_low=+∞`（防御恒等于 base） | 实测 L3 贡献 +0.140（旧 +0.595 为钉死 max_def 的错误基线） |
| Medium | M1 vol_ddof/hedge 接线 | vol_ddof 沿完整因子链接线（backtest stateful + factors.calculate_volatility + engine_core.compute_inv_vol_weights）；hedge 在周收益前注入 | ddof=0 vs 1 → Sharpe 1.6096 vs 1.6156（活字段）；hedge=0.002 → 年化 −2.59% |
| Medium | M2 dynamic_margin_sensitivity 入 _critical | `_critical['selection']` 加该键，兜底 0.0→1.0 | 缺键时 load_config 抛 ValueError |
| Low | L1 DSR n=n_obs | `n = max(int(n_obs), 1)` | DSR(1.61,N=243,n=665)=1.0（口径标准） |
| Low | L2(a) 删未用 euler 死代码 | 删除 `euler` 行 | grep 确认无残留 |
| Low | L2(b) MC 扰动扩至 11 维 | active_params 补 score_margin/rebalance_threshold/max_single_alloc/dynamic_margin_sensitivity | effective_dims=11 |
| Low | L3 OOS 选参去偏 | 训练窗选参由 max-Sharpe 改 max-DSR(N=243 矫正) | DSR 选出更稳参数，OOS 1.796>IS 1.513，无过拟合 |
| Low | L4/L6 防御比率 0/0 退化 | `step_high==step_low` 时按阈值返 base/max_def | nasdaq_vol=step 边界不再产生 NaN |
| Low | L7 文档同步 | P0_P1_RESOLUTION.md Kimi 段 + README MC 维度(7→11) 更正 | 数字与代码一致 |

> 未改行为的项：L4（MC 用全期 in-sample 回测）——原 review 已标注「可接受」，仅需在报告注明；本次未改动其运行窗口。
> 生产 config 参数未采纳（按你之前指示），OOS 仅作验证、不写回 config。
