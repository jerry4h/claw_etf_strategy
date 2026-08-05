# 虾池 ETF 策略项目行动准则与原则

> 本文件是项目级约束规则，所有代码修改、因子开发、回测实验、数据管线变更都必须遵守。

---

## 一、数据准则

1. **前复权价 × 原始量/额口径自洽**：log_return 用前复权 NAV，amount/volume 用不复权原始值——配对做相关性时代表"收益 vs 量能变化"，无口径错配。绝不用不复权价格算收益、绝不用复权值算成交额比率。

2. **成交额(amount)优于成交量(vol)**：vol（手）受拆分/分红影响产生虚假脉冲（如 512890 拆分当周 +81%），amount（千元）天然免疫。涉及量的因子一律用 amount。

3. **红利低波(512890)数据拼接感知**：日频 OHLC/amount 仅 2019-01-18 起可用，之前由 H20269 指数反推仅有周频 close。任何新因子的数据聚合必须处理此分界（pre-2019 段设为 NaN 或退化值），且聚合后 weekly close 必须与生产 NAV（`data/all_etfs_nav_latest.csv`）逐行一致（容差 ≤0.001）。参考 `docs/etf_data_build.md` §4。

4. **tushare 无 ETF 周频接口**：`fund_weekly` 不存在、`pro.weekly` 仅股票、`pro_bar(asset='FD', freq='W')` 仅日频——周频数据维持从日频手算聚合（周五锚点、H/L 周内极值、amount 周内累计）。日频是万能底仓（哨兵溢价、PVD amount、周内极值均依赖）。

5. **数据敏感性分级**：
   - 原始份额/行情数据（`data/national_team/`）不入 git（已 gitignore）
   - 聚合产出（JSON/MD 报告、events）可入库
   - `tushare_cache` 为离线复现工件，git 直接提交

---

## 二、回测与信号准则

6. **动态门限必须无前视（expanding only）**：任何基于历史分布的阈值（如 vol 百分位 p25/p75、异常检测的 99 分位）禁止用全样本预计算——必须用 expanding window 仅含截至当前周数据。违反将产生系统性乐观偏差（实测 v4.5-pvd 约 19% 的 Sharpe 增量曾来自前视泄漏）。

7. **回测与实盘共用实现**：门限计算、因子函数、条件激活逻辑抽为公共函数（如 `engine_core.compute_pvd_vol_gates`），backtest 与 rebalance_live 复用同一实现，杜绝口径分叉。禁止两端各写一遍相同逻辑。

8. **verify 循环必须与引擎同口径扣费**：`--verify` 的 NAV 模拟必须扣 `turnover × fee_rate`，否则系统性高估脚本 Sharpe。首周建仓费处理需与引擎 `last_alloc=zeros` 一致。

9. **合成 DGP 对非价格因子不公平**：CCC-GARCH 仅建模价格收益，不生成成交量/份额——含量因子策略的鲁棒性以 block bootstrap（保留真实价量关系）为主判据，合成对抗通道仅供参考。

10. **信号有效性必须经 gate 验证再集成**：预研（monkeypatch，scripts/_exp_*）→ go/no-go 门禁（IC/bootstrap/对抗）→ 配置化集成（src/ 改动）。跳过任何一步都可能导致无效信号进入生产。

---

## 三、策略集成准则

11. **辅助因子用 tiebreaker 模式而非主驱动**：新因子（如 PVD、份额信号）以极低权重（≤0.15）在"策略犹豫时"条件激活打破平局，不改变主动量排名逻辑。无条件线性叠加在窄截面（5 ETF）中灾难性（实测 MaxDD +14pp）。

12. **条件激活至少双重门控**：至少两重条件（如 vol 区间 + score gap），确保非触发环境完全退化为 baseline（零风险 no-op）。Bootstrap 中性是该设计的直接验证。

13. **硬门控优于软门控（当前策略规模下）**：软化边界理论降低决策翻转率，但实测代价过大（Sharpe −0.036 + 引入新过拟合面 τ），硬门控的"扰动下均值"仍优于软门控的"无扰动值"。p75 上边界的硬切断是危机隔离设计，不可软化。

14. **配置开关隔离（零扰动原则）**：每个新功能必须有 YAML `enabled: false` 默认关闭，确保 v4_3/v4_4 生产路径零扰动。`TestBaselineUnchanged`（Sharpe pin 精确匹配）是 CI 硬约束，任何改动后必须通过。

---

## 四、工程准则

15. **数据源三级链容灾**：tushare 主用 → 新浪日K 备选（amount 需 `close×volume` 近似，误差 ±1.3%）→ baostock 补充（原生 amount 但仅 2026+ 历史）→ 腾讯兜底。任何单一源失效时自动降级而非中断。

16. **weekly_refresh 幂等 + 失败零污染**：
    - 重复执行安全（数据已最新则跳过）
    - 任何步骤失败不产生部分提交（git add 仅在全链路成功后执行）
    - 退出码语义：0=成功、2=数据源失败、3=并发锁、4=数据质量异常、5=看板失败、6=push 失败
    - Token 永不出现在日志/提交中（掩码 `***TOKEN***`）

17. **实验脚本 `_exp_` 前缀**：不入 CI、不被生产代码引用、monkeypatch 不改 src/。实验结论通过 gate 后才提升为生产代码。

18. **`run_backtest` 有副作用**：会覆写 `output/report_v3_1.md`——实验后需 `git checkout output/report_v3_1.md` 恢复。

19. **提交前必做 token 扫描**：`git diff --cached` 中 grep 56 位 token 特征串 / `df5c` 前缀 / `TUSHARE_TOKEN=实值` 必须零命中。`.env` 必须在 `.gitignore` 中。

---

## 五、已知限制与踩坑教训

20. **QDII ETF 盘中 High/Low 被溢价/折价严重污染**：513100 的 Parkinson 波动率 corr 仅 0.30、噪声 4×——盘中价格不可用于该类 ETF 的波动率信号。日频收盘价不受此影响但"更精确 ≠ 更快"（RV 不领先 CC-tapered vol）。

21. **灰区保护与 bond_bear 零劣化物理冲突**：阈值降到 ≤0.50 修复灰区但 bond_bear DGP 自然相关中位≈0.40 恰落灰区——36 格实验证实纯标量/门控机制无解。定向 boost（仅施加进攻端，需改 Layer 3.5）是未来方向。

22. **主力份额信号对策略无可操作增量**：IC=0.015 不显著、事件后反而跑输、公告对齐 recall 仅 33%。但作为市场观察工具有独立价值（板块 rotation、逆势异动）。

23. **新因子引入后必须联合参数校验**：PVD 引入后 mom_w 最优从 1.0 微调至 1.1——3×3 网格 + bootstrap 是最低成本确认手段，不可省略。

---

## 执行约束

- 本文件中的规则优先级高于一般编码习惯
- 违反第 6/7/8/14 条的代码在 code review 中为 **Critical（必须修复）**
- 违反第 1/2/3/10/11 条为 **Warning（应修复）**
- 第 20-23 条为参考信息，指导方向选择而非强制约束
