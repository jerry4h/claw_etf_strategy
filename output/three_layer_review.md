# 三层深度审查报告 — 虾池ETF轮动 v3.0

> 审查日期: 2026-07-22
> 审查范围: 全仓代码（src/, scripts/, tests/, config/, docs/）
> 审查维度: 机构级 / 策略级 / 代码级
> 当前状态: 39/39 测试通过, Sharpe 1.522, 年化 15.97%, DD 6.38%

---

## 一、机构级审查（Institutional Level）

### I-P0: 无自动化交易执行与对账系统

**现状**: `rebalance_live.py` 仅输出调仓建议，需人工在券商APP手动执行。`--save-state` 依赖人工确认后才保存。无任何 broker API 集成。

**风险**:
- 人工执行存在操作错误风险（买错标的、数量错误、遗漏执行）
- 无自动对账机制验证实际持仓与目标仓位的一致性
- 执行延迟可能导致信号过期
- 无法满足机构级的 best execution 要求

**建议**:
- 短期：增加 post-trade 对账脚本，手动输入实际成交价和数量，自动计算偏离度
- 中期：接入券商API（如 QMT/Ptrade），实现半自动执行
- 长期：构建完整 OMS（订单管理系统），含执行算法和 slippage 监控

### I-P0: 无实时风控监控系统

**现状**: 策略仅在每周一调仓时计算一次信号。无日内回撤监控、无仓位偏离告警、无黑天鹅事件熔断。

**风险**:
- 周内极端行情（如2020-03 COVID单周跌15%+）无法触发应急防御
- ETF 暂停交易、涨跌停、大规模折溢价无法实时响应
- 服务器宕机无人知晓，可能错过关键调仓窗口

**建议**:
- 增加日频 NAV 监控脚本，设置回撤告警阈值（如日内DD > 3% → 推送通知）
- 利用定时任务功能配置每周一自动运行调仓脚本
- 配置服务器存活检测（cron 每小时 ping，失败推送）

### I-P1: 单点故障 — 单服务器无容灾

**现状**: 策略代码和数据全部在单一 qcloud 服务器 (124.221.200.37) 上运行。无备份服务器、无热切换、无异地灾备。

**风险**:
- 服务器硬件故障、云服务商宕机、网络中断 → 策略完全停摆
- 数据文件损坏（如 symlink target 丢失）→ 无法回测或调仓
- 无系统化备份策略，仅依赖 git push 和手动 data backup

**建议**:
- 关键状态文件 (.last_alloc.json, .last_backup_target) 同步到云存储
- 建立二级服务器或本地备用环境，每周自动同步代码+数据
- 编写灾难恢复 runbook，明确各故障场景的恢复步骤和 RTO

### I-P1: 无交易审计日志

**现状**: 调仓决策仅通过 print() 输出到终端。状态文件 (.last_alloc.json) 只存最新一次仓位，无历史决策记录。

**风险**:
- 无法追溯某次调仓的决策依据（评分、vol、def_ratio 等）
- 无法满足机构级的合规审计要求
- 事后复盘缺乏完整证据链

**建议**:
- 每次调仓生成结构化日志（JSON/CSV），包含：日期、评分明细、防御比例、目标仓位、实际仓位、偏离度
- 日志持久化到 `logs/rebalance_YYYYMMDD.json`
- 保留至少 2 年的决策历史

### I-P1: 数据管道单一依赖

**现状**: 所有 ETF 数据来自 Tushare 单一数据源，单一 API token。数据更新脚本依赖 Tushare 服务可用性。

**风险**:
- Tushare 服务中断或 API 变更 → 数据更新失败
- Tushare 数据错误（除权除息处理错误、价格异常）→ 回测结果失真
- 无替代数据源验证数据准确性

**建议**:
- 增加数据交叉验证：每周数据更新后，用第二个数据源（如 akshare/eastmoney）抽样验证
- 配置 Tushare API 调用失败的重试机制和告警
- 数据更新后自动运行回测 sanity check（Sharpe 偏离 < 5%）

### I-P2: 无容量分析

**现状**: 策略交易5只ETF，未分析各ETF的流动性、市场冲击成本、容量上限。

**风险**:
- 纳指ETF(513100)日均成交额约 3-5 亿，大资金调仓可能产生显著滑点
- 黄金ETF(518880)流动性更差，大额交易冲击更大
- QDII 额度限制可能导致纳指ETF暂停申赎

**建议**:
- 补充各 ETF 的日均成交额、买卖价差、最佳执行窗口分析
- 估算不同 AUM 规模下的预期滑点，确定策略容量上限
- 对纳指ETF增加溢价率实时监控（目前仅有文字提醒）

### I-P2: 密钥管理不足

**现状**: TUSHARE_TOKEN 存在 `.env` 纯文本文件中。虽然在 .gitignore 中，但服务器上无加密保护。

**建议**:
- 使用云平台 KMS 或环境变量注入方式管理密钥
- 定期轮换 API token
- 限制 token 权限范围（仅读取）

---

## 二、策略级审查（Strategy Level）

### S-P0: Walk-Forward 胜率仅 55.6%，样本外有效性存疑

**现状**: 9个滚动窗口中仅5个跑赢等权基准。虽然 DSR=1.0 和 MC=100% 表现优秀，但 WF 胜率刚过半数。

**深层问题**:
- WF 的 4/9 失败窗口集中在哪些时期？是系统性失败还是随机波动？
- 等权基准(每周再均衡)本身是一个高换手策略，可能不是最公平的比较基准
- 参数（vol_w=1.10, vol_window=11）在全区间最优，但可能存在 regime-dependent overfitting

**建议**:
- 分析 4/9 失败窗口的时间分布和失败幅度
- 增加买入持有等权（非每周再均衡）作为第二基准
- 尝试 Regime-Conditional WF：按市场状态分组评估 WF 胜率
- 考虑参数衰减分析：如果只用 2013-2019 数据选参，2020-2026 表现如何？

### S-P0: 全区间参数优化无 holdout period

**现状**: 2013-2026 全部数据用于参数选择和性能报告。标注为"C1最优参数"。无独立验证集。

**风险**:
- 参数可能在特定历史 regime 上过拟合
- DSR=1.0 的 30 试次矫正可能低估了实际的多重测试次数（超参扫描了多少组合？）
- 没有真正的 out-of-sample 验证

**建议**:
- 划分 train (2013-2020) / test (2020-2026) 两段
- 在 train 上选参，在 test 上报告性能
- 如果 test 性能显著下降，说明存在过拟合

### S-P1: ETF Universe 变更未充分验证

**现状**: 2026-07 将沪深300ETF替换为中证500ETF。虽然报告了 Sharpe +0.088 改善，但：
- 替换时机恰好在替换前表现最好时，可能存在 data-snooping
- 中证500ETF 的历史数据质量（前复权因子-0.1490）可能影响回测可比性
- 未进行 universe 变更的消融实验

**建议**:
- 同时报告新旧 universe 在相同时期的回测结果
- 验证中证500ETF 前复权因子的正确性（与交易所原始数据对比）
- 考虑动态 universe 选择机制而非硬编码

### S-P1: 汇率风险敞口未量化

**现状**: 纳指ETF(513100)持有美股资产，存在 USD/CNY 汇率风险。`kimi_audit_verification.py` 的汇率对冲成本测试有 bug（见代码级 C-P0），无法得出有效结论。

**风险**:
- USD/CNY 年化波动约 3-5%，可能侵蚀纳指ETF的收益贡献
- 汇率贬值时纳指ETF收益放大（汇率+资产双重收益），升值时反向
- 未剥离汇率因子，无法判断策略 alpha 的纯度

**建议**:
- 修复 FX hedge test bug（直接从 w_rets 扣除对冲成本，而非调整价格）
- 分解纳指ETF收益为：标普500收益 + 汇率变动 + QDII溢价/折价
- 评估汇率对冲后策略的真实 alpha

### S-P1: 危机期相关性收敛风险

**现状**: 5只ETF在正常时期相关性低，但压力测试仅覆盖6个窗口。未分析极端情况下相关性收敛的影响。

**风险**:
- 2008金融危机、2020-03 COVID 等极端时期，纳指与黄金可能同时下跌
- 防御层(红利低波+国债)在流动性危机中也可能被抛售
- inv-vol 加权在低波时期可能过度集中于单一资产

**建议**:
- 扩展压力测试至 2008 金融危机（如有数据）
- 计算滚动相关性矩阵，识别相关性收敛期
- 在 inv-vol 加权中加入相关性调整（如 risk parity 变体）

### S-P2: 动量窗口过短，噪声敏感

**现状**: mom_window=4 周。虽然鲁棒性测试显示 3-6 安全，但4周动量对短期噪声敏感，可能导致频繁换仓。

**现状缓解**: score_margin=0.02 在一定程度上抑制了噪声换仓。

**建议**:
- 分析历史换仓频率和 score_margin 触发次数
- 考虑多周期动量复合（如 4w+8w+12w 加权）降低单一窗口噪声

### S-P2: 无利率敏感度分析

**现状**: 策略持有国债ETF(511010)作为防御资产。利率变动对国债价格有直接影响，但未分析利率变化对策略的影响。

**建议**:
- 测试不同利率环境下策略表现
- 分析国债ETF在加息周期中的防御效果是否衰减

### S-P3: DefAlloc 常量硬编码风险

**现状**: `hl_ratio = clip(0.80 - 2.67 * vol_hongli, 0, 0.80)` 中的 0.80 和 2.67 是 T=0.30 的派生常量，在 backtest.py、rebalance_live.py、kimi_audit_verification.py 三处硬编码。

**建议**: 提取为 config 参数 `hl_ratio_base=0.80` 和 `hl_ratio_slope=2.67`，从 T=0.30 计算

---

## 三、代码级审查（Code Level）

### C-P0: 汇率对冲测试 bug — 价格调整不传播

**文件**: `scripts/kimi_audit_verification.py` 第 175-185 行

**问题**: FX hedge 测试通过修改 `prices_adj` 来扣除对冲成本：
```python
prices_adj[i, nasdaq_idx_col] = prices_adj[i-1, nasdaq_idx_col] * (
    prices_adj[i, nasdaq_idx_col] / prices_adj[i-1, nasdaq_idx_col] - weekly_hedge
)
```
但后续 `run_with_ddof()` 内部调用 `np.diff(prices, axis=0) / prices[:-1]` 计算收益率，由于前后价格都被调整，比率变化极小。实测差异仅 ~2e-6，而预期差异为 ~-1.9e-4。

**根因**: `np.diff` 计算的是 `(price[i+1] - price[i]) / price[i]`，对两个被同步调整的价格求比率几乎抵消了对冲成本。

**修复**: 直接在 `w_rets` 上扣除对冲成本，而非调整价格：
```python
# 在 run_with_ddof 内部增加 hedge_cost 参数
wret = sum(alloc[j] * (w_rets[i, j] - (weekly_hedge if j == nasdaq_idx else 0)) ...)
```

### C-P0: 策略逻辑三处重复实现

**文件**: `src/backtest.py`, `scripts/rebalance_live.py`, `scripts/kimi_audit_verification.py`

**问题**: 三个文件各自独立实现了完整的策略逻辑（评分-选股-防御-分配-调仓）。虽然 `rebalance_live.py` 有 `--verify` 模式验证一致性，但任何策略逻辑修改需要同步三处，极易遗漏。

**当前缓解**: `test_consistency.py::test_full_verify_sharpe_gap` 验证 Sharpe 差距 < 0.05。

**建议**:
- 将核心策略逻辑提取为 `src/engine.py` 模块，提供 `compute_weekly_decision(nav, i, config, prev_state) -> (alloc, metadata)` 接口
- backtest.py、rebalance_live.py、kimi_audit_verification.py 均调用此接口
- 消除逻辑重复，确保单一真相源

### C-P1: strategy.py 1179行巨型文件，80%为死代码

**文件**: `src/strategy.py` (1179行)

**问题**: `StrategyConfig` 包含 80+ 个字段，其中绝大部分属于已关闭功能（D1/D4/D5/T32/T35/Phase A-2/stateful stop loss/softmax allocation 等）。这些功能在最终配置中全部 `enabled: false`，但代码仍占据大量行数，增加维护负担和理解难度。

**影响**:
- 新开发者难以快速理解策略核心逻辑
- 修改参数时容易误碰已关闭功能的开关
- 代码审查复杂度大幅增加

**建议**:
- 将已废弃功能移至 `src/legacy/` 目录，保留 git 历史
- 精简 `StrategyConfig` 至仅保留 v3.0 实际使用的参数（约 20 个字段）
- 精简 `load_config` 至仅解析 YAML 中实际使用的段

### C-P1: backtest.py run_backtest() 300+行巨型循环

**文件**: `src/backtest.py`, `run_backtest()` 函数

**问题**: 核心回测循环从第 155 行到第 400+ 行，包含多个 if/elif 分支处理不同止损模式（原始/tiered/ptiered/stateful），每个分支内部又有嵌套逻辑。

**影响**:
- 难以单独测试某个止损模式的正确性
- 修改一处逻辑可能意外影响其他分支
- 循环内变量（in_stop_loss, stop_loss_level, recovery_ctr, previous_def 等）状态管理复杂

**建议**:
- 将止损逻辑提取为独立的 `StopLossManager` 类
- 将仓位分配提取为 `AllocationBuilder` 类
- 主循环精简为信号生成 - 止损检查 - 仓位构建 - 收益计算的线性流程

### C-P1: 无 CI/CD 流水线

**现状**: 仓库无 `.github/workflows/`，无任何 CI 配置。39个测试必须手动运行。

**风险**:
- 提交前容易遗忘运行测试
- 代码修改后无自动回归验证
- 无法保证 push 到 main 的代码质量

**建议**:
- 添加 `.github/workflows/test.yml`，在 push/PR 时自动运行 pytest
- 添加代码覆盖率报告（pytest-cov）
- 添加 lint 检查（ruff/flake8）

### C-P1: 状态文件写入非原子操作

**文件**: `scripts/rebalance_live.py`, `save_state()` 函数

**问题**: `STATE_FILE.write_text(json.dumps(alloc))` 直接写入目标文件。如果进程在写入中途崩溃，文件将损坏。

**建议**: 写入临时文件后原子重命名：
```python
import tempfile, os
tmp = tempfile.NamedTemporaryFile(dir=STATE_FILE.parent, delete=False, suffix='.tmp')
tmp.write(json.dumps(alloc, ensure_ascii=False, indent=2).encode())
tmp.close()
os.replace(tmp.name, STATE_FILE)
```

### C-P1: 测试覆盖不足

**现状**: 39个测试覆盖 factors/strategy/consistency/no_lookahead，但缺失：
- `src/robustness.py`（DSR/MC/WF 计算）— 零测试
- `src/report.py` — 零测试
- `src/data_loader.py`（load_nav_data, resample_weekly）— 零测试
- `scripts/rebalance_live.py` — 零测试
- `scripts/update_etf_data_tushare.py` — 零测试
- `scripts/cost_sensitivity.py`, `scripts/stress_test.py` — 零测试

**风险**:
- robustness.py 包含 DSR 等复杂数学计算，无测试验证正确性
- data_loader.py 的数据清洗逻辑（NaN 处理、ffill、截断）是回测准确性的基础，无测试保护

**建议**:
- 优先为 robustness.py 的 DSR 计算添加单元测试（与论文公式对比）
- 为 data_loader.py 的 NaN 处理和周频重采样添加测试
- 为 rebalance_live.py 的 compute() 函数添加与引擎的一致性测试

### C-P2: data_loader.py 列名匹配脆弱

**文件**: `src/data_loader.py`, `load_nav_data()` 函数

**问题**: 列名检测逻辑 `if first_col.isdigit() or (first_col.isascii() and len(first_col) <= 3)` 用于判断是否为旧格式，但这个启发式可能误判（如列名为 "001" 的 ETF 代码）。

**建议**: 使用显式的版本标记或列名列表检测

### C-P2: PE 分位数计算低效

**文件**: `src/factors.py`, `calculate_pe_percentile()` 函数

**问题**: 使用 Python for 循环逐行计算滚动百分位数，时间复杂度 O(n^2)。虽然 PE 当前未使用（config 中 pe_path 存在但回测不依赖），但如果未来启用会严重影响性能。

**建议**: 使用 pandas 的 rolling + rank 向量化实现

### C-P2: ETF 名称硬编码

**问题**: '纳指ETF', '红利低波ETF' 等字符串在 `data_loader.py`, `strategy.py`, `backtest.py`, `rebalance_live.py`, `kimi_audit_verification.py` 中硬编码。

**建议**: 统一在 `data_loader.py` 的 `ETFS` 列表中定义，其他文件通过 import 引用。大部分文件已这样做，但 backtest.py 的 `classify_etfs()` 使用关键词匹配，逻辑分散。

### C-P3: StrategyConfig 默认值与 YAML 不一致

**文件**: `src/strategy.py`

**问题**: `StrategyConfig` 的默认 `version="2.3"`，但 YAML 配置为 `version: 3.0`。虽然 `load_config()` 正确从 YAML 读取，但默认值有误导性。

**建议**: 更新默认值为 `"3.0"`

### C-P3: 无日志框架

**问题**: 所有脚本使用 `print()` 输出，无结构化日志、无日志级别、无日志文件。

**建议**: 引入 Python `logging` 模块，配置 DEBUG/INFO/WARNING 级别

---

## 四、问题优先级汇总

| 优先级 | 编号 | 问题 | 层级 |
|--------|------|------|------|
| **P0** | I-1 | 无自动化交易执行与对账 | 机构 |
| **P0** | I-2 | 无实时风控监控 | 机构 |
| **P0** | S-1 | WF胜率仅55.6%，样本外有效性存疑 | 策略 |
| **P0** | S-2 | 全区间参数优化无holdout | 策略 |
| **P0** | C-1 | FX hedge测试bug | 代码 |
| **P0** | C-2 | 策略逻辑三处重复实现 | 代码 |
| **P1** | I-3 | 单服务器无容灾 | 机构 |
| **P1** | I-4 | 无交易审计日志 | 机构 |
| **P1** | I-5 | 数据管道单一依赖 | 机构 |
| **P1** | S-3 | ETF Universe变更未充分验证 | 策略 |
| **P1** | S-4 | 汇率风险敞口未量化 | 策略 |
| **P1** | S-5 | 危机期相关性收敛风险 | 策略 |
| **P1** | C-3 | strategy.py 1179行死代码 | 代码 |
| **P1** | C-4 | run_backtest()巨型循环 | 代码 |
| **P1** | C-5 | 无CI/CD流水线 | 代码 |
| **P1** | C-6 | 状态文件写入非原子 | 代码 |
| **P1** | C-7 | 测试覆盖不足 | 代码 |
| **P2** | I-6 | 无容量分析 | 机构 |
| **P2** | I-7 | 密钥管理不足 | 机构 |
| **P2** | S-6 | 动量窗口过短 | 策略 |
| **P2** | S-7 | 无利率敏感度分析 | 策略 |
| **P2** | C-8 | 列名匹配脆弱 | 代码 |
| **P2** | C-9 | PE百分位计算低效 | 代码 |
| **P2** | C-10 | ETF名称硬编码 | 代码 |
| **P3** | S-8 | DefAlloc常量硬编码 | 策略 |
| **P3** | C-11 | 默认版本不一致 | 代码 |
| **P3** | C-12 | 无日志框架 | 代码 |

---

## 五、建议优先修复路线

### 第一阶段（立即修复，1-2天）
1. **C-1**: 修复 FX hedge 测试 bug（直接从 w_rets 扣除对冲成本）
2. **C-6**: 状态文件原子写入
3. **C-11**: 更新 StrategyConfig 默认版本
4. **I-2 部分**: 配置定时任务每周一自动运行调仓脚本

### 第二阶段（短期改进，1-2周）
5. **S-2**: 划分 train/test，在 test 上验证参数稳健性
6. **S-1**: 分析 WF 失败窗口的分布和原因
7. **S-4**: 修复后重新运行汇率对冲成本敏感度测试
8. **C-7**: 为 robustness.py 和 data_loader.py 添加单元测试
9. **C-5**: 添加 GitHub Actions CI

### 第三阶段（中期演进，1-2月）
10. **C-2**: 提取策略逻辑为 src/engine.py，消除三处重复
11. **C-3**: 清理 strategy.py 死代码
12. **C-4**: 重构 run_backtest() 为可组合组件
13. **I-4**: 建立调仓审计日志系统
14. **I-1 部分**: 增加 post-trade 对账脚本

### 第四阶段（长期建设，3-6月）
15. **I-1**: 接入券商API，半自动执行
16. **I-3**: 建立容灾备份机制
17. **I-5**: 增加数据交叉验证
18. **S-3**: 动态ETF Universe机制
19. **S-5**: 相关性收敛分析与risk parity变体
