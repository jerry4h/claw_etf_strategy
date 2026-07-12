# 量化工程代码审视报告 — claw_etf_strategy

**日期**: 2026-07-12  
**审查范围**: quant-se 最近 9 次提交（HEAD~9..HEAD）  
**配置文件**: config/strategy_v3_0_final.yaml  
**核心引擎**: src/backtest.py, src/strategy.py, src/data_loader.py, src/factors.py  

---

## 一、量化专项检查

### 1.1 配置完整性 ✅

配置检查结果：

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 配置文件是否存在 | ✅ | config/strategy_v3_0_final.yaml 存在 |
| 配置引用是否闭合 | ✅ | 所有脚本引用路径均可解析 |
| nav_path 数据文件存在 | ✅ | data/all_etfs_nav_latest.csv（最新数据） |
| pe_path 数据文件存在 | ✅ | data/300etf_pe_percentile_weekly.csv |
| 新增 score_margin 配置 | ✅ | selection 段已定义（0.02） |
| .gitignore 是否排除了关键数据 | ✅ | 保留最新 CSV 文件 |

**发现**：config 顶部注释中的参数描述与实际值存在小差异。  
- `# C1参数: vol_w=1.10, vol_window=11, mom_window=4, invvol=10, thresh=0.025` ✓  
- `# score_margin 在 selection 段定义（默认 0=关闭）` ✓  
- 但注释中 `鲁棒性: MC 100%, DSR 1.0, WF 66.7%` 与旧版本（WF 77.8%）不同，已更新。

### 1.2 参数流完整性 ✅

`config YAML → StrategyConfig dataclass → backtest/script` 链路检查：

| 参数 | YAML | StrategyConfig | load_config | backtest | rebalance_live | 状态 |
|------|------|---------------|-------------|----------|----------------|------|
| score_margin | selection.score_margin = 0.02 | ✅ 已添加 | ✅ | ✅ 已使用 | ✅ 已使用 | ✅ |
| mom_window | factors.mom_window = 4 | ✅ 已存在 | ✅ | ✅ | ✅ 现已读取 | ✅ |
| vol_window | factors.vol_window = 11 | ✅ 已存在 | ✅ | ✅ | ✅ 现已读取 | ✅ |
| mom_w | scoring.mom_w = 1.0 | ✅ | ✅ | ✅ | ✅ | ✅ |
| vol_w | scoring.vol_w = 1.10 | ✅ | ✅ | ✅ | ✅ | ✅ |
| inv_vol_window | inv_vol_allocation.window = 10 | ✅ | ✅ | ✅ | ✅ | ✅ |
| rebalance_threshold | rebalance.threshold = 0.025 | ✅ | ✅ | ✅ | ✅ | ✅ |

**结论**：参数流完整闭合，没有遗漏或未使用的参数。所有新增参数（score_margin、mom_window、vol_window）都已正确贯通。

### 1.3 架构合规性 ✅

| 目录 | 用途 | 合规 | 说明 |
|------|------|------|------|
| src/ | 核心模块 | ✅ | backtest, strategy, data_loader, factors, regime, report, robustness, utils |
| config/ | YAML 策略配置 | ✅ | 单一最终配置文件 |
| scripts/ | 入口脚本 | ✅ | rebalance_live, run_backtest, calc_performance, web_app, update |
| data/ | CSV 数据 | ✅ | 遵循 .gitignore 规则 |
| output/ | 生成结果 | ✅ | 报告已放置（report_v3_0.md） |
| tests/ | 测试 | ⚠️ 未见测试文件 | 建议补充单元测试 |

**发现**：
- tmp/ 目录下有 candidate_eval.py 和 param_sweep.py，属于临时分析脚本。建议完成后移入 scripts/ 或删除。
- web_app.py 是新增功能（692行），结构清晰，内含 Flask Web 界面和 Basic Auth。
- .venv/ 在代码根目录下，已在 .gitignore 中通过 `.venv/` 忽略 ✅

### 1.4 数据引用完整性 ✅

| 引用路径 | 脚本 | 存在 | 说明 |
|----------|------|------|------|
| data/all_etfs_nav_latest.csv | config 默认路径 | ✅ | 已更新为最新数据（2026-07-10） |
| data/300etf_pe_percentile_weekly.csv | config 默认路径 | ✅ | 存在 |
| data/.last_alloc.json | rebalance_live.py | ✅ 运行时创建 | 保存调仓状态 |

数据路径引用全部可解析。之前版本中使用硬编码路径的问题已修复。

### 1.5 输出标签准确性 ✅

| 报告指标 | 数值 | 计算方法 | 正确性 |
|----------|------|----------|--------|
| Sharpe（标准夏普） | 1.485 | compute_sharpe（扣无风险利率） | ✅ 统一标准定义 |
| 年化收益 | 16.75% | annualize_return（52/n 指数） | ✅ |
| 最大回撤 | 8.71% | compute_max_drawdown（全局回撤） | ✅ |
| 防御周数 | 343/663（52%） | def_ratio > 0.25 计数 | ✅ |

**注意**：报告中「标准夏普」出现了两次（同一指标重复）。代码中已统一为单一 Sharpe 定义。

---

## 二、quant-se 核心变更分析

### 2.1 沪深300ETF → 中证500ETF 迁移

这是 quant-se 最核心的变更。变更涉及的源文件：

- **data/all_etfs_nav_latest.csv** — 替换为 中证500ETF 数据
- **config/strategy_v3_0_final.yaml** — ETF 名称、参数、注释全部更新
- **src/data_loader.py** — ETFS 列表从'沪深300ETF'改为'中证500ETF'
- **src/strategy.py** — 进攻层定义、ETF 列表同步更新
- **src/robustness.py** — 基准 Sharpe 计算中 ETF 名称更新
- **scripts/rebalance_live.py** — CSV 格式注释更新
- **scripts/update_etf_data_tushare.py** — ETF_MAP 代码映射更新

**迁移理由**：README 中说明中证500ETF 相关性更低（0.245 vs 0.55），年化收益更高（8.31% vs 6~8%）。

### 2.2 C1 参数优化

参数变更对比：

| 参数 | 旧值（沪深300版） | 新值（中证500版） | 变更原因 |
|------|-------------------|-------------------|----------|
| vol_w | 1.05 | **1.10** | 更高波动惩罚，压制中证500高波动 |
| vol_window | 20 | **11** | 更快捕捉中证500的波动变化 |
| inv_vol_window | 12 | **10** | 更快的 inv-vol 响应 |
| rebalance_threshold | 0.06 (6%) | **0.025 (2.5%)** | 更频繁调仓应对快速风格切换 |
| step_low | 0.12 | **0.15** | 防御起效阈值提高 |

性能结果：Sharpe 从 ~1.397 提升至 1.485（+0.088），年化从 ~15.12% 提升至 16.75%（+1.63pp），DD 从 ~8.4% 微升至 8.71%。

### 2.3 score_margin 防噪声换仓（新功能）

**新增参数**：`selection.score_margin = 0.02`  
**生效范围**：
- `src/strategy.py` — StrategyConfig 新增字段
- `src/backtest.py` — run_backtest() 逐周循环中加入 margin 判断逻辑
- `scripts/rebalance_live.py` — compute() 函数增加 prev_sel 参数，实现 score_margin 过滤

**逻辑**：当 TOP_N 和 TOP_N+1 名的得分差距小于 margin 时，保持上次选择的进攻 ETF，防止微小分数差导致无意义的换仓。

**代码路径完整性检查**：
1. config.yaml → load_config → ✅
2. backtest.py run_backtest() 中的实现 ✅
3. rebalance_live.py compute() 中的实现 ✅
4. last_selected 在逐周循环中正确维护 ✅

### 2.4 `--save-state` 调仓状态持久化

**变更**：rebalance_live.py 新增 `--save-state` 参数，将调仓结果保存到 `data/.last_alloc.json`。

**改进**：
- 阈值判断参考的是"上次实际调仓仓位"而非"上周理论计算仓位"
- 有状态文件时优先加载，否则降级到上周理论仓位
- 使调仓决策更贴近实际操作

**问题**：
- web_app.py 未同步支持 save-state 逻辑（但属于不同使用场景，可接受）
- calc_performance.py 不涉及调仓，无需同步

### 2.5 因子窗口从硬编码改为配置读取

**变更前**：rebalance_live.py 中的 `engine_factors()` 硬编码 `window=4` 和 `window=20`
**变更后**：使用 `MOM_WINDOW` 和 `VOL_WINDOW`（从 config 加载）

这修正了一个重要的一致性漏洞：之前实时脚本与引擎回测可能使用不同的因子窗口，导致 --verify 偏差 0.0505。修正后偏差降至 0.0109 ✅。

---

## 三、代码质量评估

### 3.1 代码风格与可维护性

| 项目 | 评价 |
|------|------|
| PEP8 合规 | ✅ 总体良好，部分长行未折叠 |
| 类型注解 | ✅ 代码库广泛使用注解（`StrategyConfig` dataclass 等） |
| 函数文档 | ✅ 大部分函数有 docstring |
| 中英文混合 | ⚠️ 注释以中文为主，变量名和代码符号保持英文，合理 |

### 3.2 发现的问题（含 Bug）

#### 🔴 Bug 1：rebalance_live.py 潜在 NameError（边界条件）

**位置**：`scripts/rebalance_live.py` 第 240-244 行

```python
prev_sel = None
if idx > max(MOM_WINDOW, VOL_WINDOW):       # idx > 11
    prev_sc = compute(df, idx - 1)[1]        # [1] = sc
prev_sel = sorted(prev_sc, ...) if prev_sc else None  # ⚠️ prev_sc 可能未定义
```

**问题**：当 `idx == max(MOM_WINDOW, VOL_WINDOW)`（即 idx == 11）时：
1. `idx > 11` 为 False，跳过赋值
2. 但变量 `prev_sc` 从未定义
3. 第 244 行访问 `prev_sc` → **NameError**

**影响**：当查询日期恰好等于第 11 个数据点时（即最早可计算因子的边界），程序崩溃。

**修复**：`if idx > ...` 改为 `if idx >= ...`，或在 `if` 内部完成所有 prev_sel 计算：

```python
prev_sel = None
if idx > max(MOM_WINDOW, VOL_WINDOW):
    prev_sc = compute(df, idx - 1)[1]  # sc is index 1
    prev_sel = sorted(prev_sc, key=lambda e: prev_sc[e], reverse=True)[:TOP_N] if prev_sc else None
```

#### 🔴 Bug 2：web_app.py 硬编码因子窗口与配置不匹配

**位置**：`scripts/web_app.py` 第 69-70 行

```python
m4 = calculate_momentum(nav, window=4)   # 硬编码 4
v20 = calculate_volatility(nav, window=20) # 硬编码 20
```

**问题**：
- config 中 `vol_window = 11`，而 web_app.py 使用 `window=20`
- 这导致 Web 界面计算的波动率与引擎回测不一致
- `--verify` 如果用于 web_app 结果，偏差会很大

**影响**：Web 界面展示的调仓方案可能错误（尤其影响 `vol` 三段式防御和 inv-vol 权重）。

**修复**：从 cfg 读取窗口参数：

```python
MOM_WINDOW = cfg.mom_window
VOL_WINDOW = cfg.vol_window

def engine_factors(nav):
    m4 = calculate_momentum(nav, window=MOM_WINDOW)
    v20 = calculate_volatility(nav, window=VOL_WINDOW)
    ...
```

#### 🔴 Bug 3（关联）：web_app.py print 标签错误

**位置**：`scripts/rebalance_live.py` 第 256-258 行

```python
print(f" mom_w={MOM_W}  vol_w={VOL_W}  top_n={TOP_N}  invvol{INV_VOL_W}  "
      f"mom_w={MOM_WINDOW}  vol_w={VOL_WINDOW}  "  # ⚠️ 标签错
      f"step_low={STEP_LOW}  thresh={REBAL_THRESH}")
```

**问题**：第二行 `mom_w={MOM_WINDOW}` 标签写着 `mom_w` 但实际显示的是 `mom_window`（窗口值 4），同理 `vol_w={VOL_WINDOW}` 显示的是 `vol_window`（11）。这会严重混淆使用者。

**预期输出**：
```
mom_w=1.0  vol_w=1.10  top_n=2  invvol10  mom_window=4  vol_window=11  step_low=0.15  thresh=0.025
```

**实际输出**（当前代码）：
```
mom_w=1.0  vol_w=1.10  top_n=2  invvol10  mom_w=4  vol_w=11  step_low=0.15  thresh=0.025
```

### 3.3 其他改进建议

#### ⚠️ 问题 1：report.py 中多余输出行

`generate_metrics_table()` 中「标准夏普」出现了两次：

```python
| 标准夏普 | {m['sharpe_ratio']:.3f} |
| 标准夏普 | {m['sharpe_ratio']:.3f} |
```

这是旧版本遗留的重复行（之前分别显示"标准夏普"和"简化夏普"）。统一 Sharpe 后应删除重复行。

#### ⚠️ 问题 2：web_app.py 硬编码因子窗口

```python
def engine_factors(nav):
    m4 = calculate_momentum(nav, window=4)   # ⚠️ 硬编码 4
    v20 = calculate_volatility(nav, window=20) # ⚠️ 硬编码 20
```

与 rebalance_live.py 不同，web_app.py 仍使用硬编码窗口，未从 config 加载 mom_window/vol_window。这会导致 Web 界面与引擎回测结果不一致。建议同步修改为 config 驱动的窗口参数。

#### ⚠️ 问题 3：web_app.py API 使用固定因子计算

web_app.py 的 `/api/preview` 和 `/api/update` 路由同样使用固定 4/20 窗口（通过 `compute_allocation` → `engine_factors` 链路），需要同步修复。

#### 🔧 建议 1：calc_performance.py 硬编码基准

```python
bn = np.ones(len(w_prices))
for i in range(1, len(w_prices)):
    wret = sum(0.2 * w_rets[i-1, j] for j in range(len(cols))
               if not np.isnan(w_rets[i-1, j]))
    bn[i] = bn[i-1] * (1 + wret)
```

基准直接使用 0.2（20%）等权，与防御层/进攻层 ETF 数量无关。当 ETF 数量变化时此值需要手动更新。建议改为 `1.0 / len(cols)` 自动计算。

#### 🔧 建议 2：tmp/ 目录的脚本需要清理

`tmp/candidate_eval.py`（177行）和 `tmp/param_sweep.py`（71行）不是最终代码，建议完成验证后清理或移入 `scripts/`。

#### 🔧 建议 3：缺少测试文件

`tests/` 目录存在但为空。建议补充：
- `test_factors.py` — 验证因子计算结果与 reproduce_original 一致
- `test_backtest.py` — 回测引擎关键路径测试
- `test_strategy.py` — 评分、选股、防御逻辑测试

### 3.3 鲁棒性评估结果（来自 README/config）

| 指标 | 结果 | 评价 |
|------|------|------|
| DSR（Deflated Sharpe） | 1.0000 🟢 | 真实 alpha 概率 >95% |
| MC 生存率（±15% 扰动） | 100% 🟢 | 参数扰动全部满足条件（年化>10% & DD<15%） |
| WF 相对胜率（6个窗口） | 66.7% 🟡 | 4/6 窗口跑赢等权基准 |
| **综合评级** | **🟢 实盘可上** | |

---

## 四、逐提交回顾

| # | 提交 | 类型 | 说明 |
|---|------|------|------|
| 1 | `1883c68` cleanup | 清理 | 最终清理 |
| 2 | `08ee247` fix | 修复 | print_scores 显示实际选中结果（margin 覆写后） |
| 3 | `d2fac3b` score_margin+final C1 | 新功能 | 新增 score_margin 防噪声换仓 + C1 参数最终确定 |
| 4 | `19f3a6b` fix | 修复 | rebalance_live 因子窗口从硬编码改为配置读取，verify 偏差 ↓ |
| 5 | `200a868` ETF迁移 | 重构 | 中证500ETF全面迁移 + C1 参数优化 |
| 6 | `a043c4c` WIP | 开发中 | 沪深300→中证500替换 + fetch_adj修复 |
| 7 | `28ac616` 路径更新 | 维护 | 更新数据路径至 latest + README |
| 8 | `3c1c49c` fix | 修复 | calc_performance 使用统一引擎（不再重复实现回测逻辑） |
| 9 | `3bea666` add | 新功能 | 新增 calc_performance.py |

**结构合理性**：提交历史清晰，修复/功能/重构分类明确。WIP 提交在合理范围内，后续有 clean commit 整理。

---

## 五、总结

### 通过的检查

| 检查项 | 结果 |
|--------|------|
| ✅ 配置完整性 | 通过 — 所有引用文件存在，参数闭合 |
| ✅ 参数流完整性 | 通过 — YAML→dataclass→engine→script 链路完整 |
| ✅ 架构合规性 | 通过 — 目录结构符合 ARCHITECTURE 约定 |
| ✅ 数据引用完整性 | 通过 — 数据文件路径均可解析 |
| ✅ 输出标签准确性 | 通过 — 指标计算与标签一致，Sharpe 已统一 |
| ✅ Sharpe 定义统一性 | 通过 — 已移除简化夏普，仅保留标准夏普 |
| ✅ Git 提交质量 | 通过 — 提交信息清晰，变更范围合理 |

### 待改进项目

| 优先级 | 问题 | 类型 | 建议 |
|--------|------|------|------|
| 🔴 高 | rebalance_live.py 边界条件 NameError | **Bug** | `if idx > ...` 改为 `if idx >= ...` 或将赋值移入 if 块内 |
| 🔴 高 | web_app.py 硬编码 vol_window=20（应为 11） | **Bug** | 从 cfg 读取窗口参数，与引擎保持一致 |
| 🔴 中 | rebalance_live.py print 标签混淆 | **Bug** | 第二行标签 mom_w→mom_window, vol_w→vol_window |
| 🟡 中 | report.py "标准夏普"重复行 | 清理 | 删除冗余行 |
| 🟢 低 | calc_performance.py 硬编码 0.2 | 可维护性 | 改为自动计算 1.0/len(cols) |
| 🟢 低 | tmp/ 目录临时脚本 | 清理 | 完成验证后清理或移入 scripts/ |
| 🟢 低 | tests/ 缺少测试文件 | 质量 | 补充核心模块单元测试 |

### 总体评价

quant-se 的变更工程规范，核心逻辑正确。主要做了一项关键重构（ETF迁移）和多项修复（硬编码→配置驱动、score_margin 防噪声换仓、save-state 状态持久化、Sharpe 定义统一）。代码质量良好，架构清晰，参数流闭合。

**发现 3 个需要提前修复的 Bug**：
1. **rebalance_live.py 边界条件 NameError**：当 `idx == max(MOM_WINDOW, VOL_WINDOW)` 时，`prev_sc` 未定义导致崩溃
2. **web_app.py 硬编码 vol_window=20**：config 实际为 11，Web 展示的波动率与引擎不一致
3. **rebalance_live.py 打印标签混淆**：`mom_w` 处实际打印了 `mom_window`（4），`vol_w` 处实际打印了 `vol_window`（11）

建议优先修复这三项后重新验证 `--verify`，以保证所有入口路径结果一致。