# v4.0 对抗鲁棒性评估框架 — 方法学与设计边界

> **一句话**：把"策略在合成对抗环境下能不能守住 DD 且 Sharpe 打赢等权"当作与 realized 回测同等地位的**常规评估维度**，用**多目标约束优化**替代对历史单条路径的收益单目标最大化，并用**三通道 OOS 验证**杜绝对对抗测试自身的过拟合。

## 1. 为什么需要这套框架

`v4_1` 基线 realized 年化 17.05%、MaxDD 6.97%、Sharpe 1.61 —— 在真实历史 665 周上碾压等权（11.84% / MaxDD 20.20% / Sharpe 0.92）。看似强健，但存在一个**结构性风险**：

这些数字都在**一条特定的历史路径**上得到。2013-2026 恰好经历中国 A 股 4 次大牛熊、防御资产（红利低波 + 国债）恰好也享受了一轮结构性慢牛。策略"防御仓吃收益"的性质在这条路径上**免费**——一旦换到另一条同分布路径（比如债券熊 + A股波动放大 + 相关性坍缩同时发生），"免费"就变成"付费"。

优化历史单目标是**在自己给自己出的考题上刷高分**——这就是过拟合历史 regime 的核心。v4.0 的目标是把"另一条路径下也不崩"变成一个**可优化、可门禁**的量化指标，与 realized 收益共同约束。

## 2. 多目标约束优化定义

```
max  realized_annual_return
s.t. realized_maxdd     ≤ D_max        (D_max 用户可配, 默认 12%)
     realized_annual    > realized_ew_annual         (realized 收益打赢等权)
     adv_worst_maxdd    ≤ D_max                      (全情景对抗 DD 也守住)
     for m in {vol_defense, defense_asset, dispersion, composite}:
         median_strat_sharpe(m) > median_ew_sharpe(m)  (硬机制门禁, Sharpe 口径)
     # selection 为软门禁, 仅记录, 不阻断
```

**为什么用 Sharpe 口径而非原始收益作为对抗鲁棒指标**：防御型策略的本质就是"用一部分收益换低波动"，要求它在合成 stress 情景下**原始收益**也打赢等权，等于同时要求它是激进策略——数学上把优化器逼向纯等权/纯进攻，与低回撤目标自相矛盾。Sharpe 口径正确捕捉"风险调整后不劣化"的意图。

**为什么 selection 是软门禁**：`offense_cooldown`（μoff×0.8）刻画进攻资产收益退化，这需要**换 universe** 才能解决（把弱进攻资产从池里踢掉），不是任何 Layer1 打分权重的调整可以覆盖的——这是策略的"天花板"而非"超参"。硬约束会让优化器无解。

## 3. 五个机制的分维门禁

对抗压力情景不是一团糊糊的整体，不同扰动由**不同策略机制**承担鲁棒性：

| 情景 (扰动) | 主控机制 | 门禁 |
|---|---|---|
| `vol_stress` (σ×1.2) | vol_defense (Layer3 波动择时) | 硬 |
| `offense_cooldown` (μoff×0.8) | selection (Layer1 打分/换 universe) | 软 |
| `bond_bear` (μdef×0.5) | defense_asset (Layer3 防御标的选择) | 硬 |
| `decorrelation` (c×0.77) | dispersion (inv-vol 轮动) | 硬 |
| `stagflation` (σ×1.2+μoff×0.8) | composite (多机制承压) | 硬 |

单一 `pass_rate` 标量会把不同性质的失败糊在一起。分机制门禁能定位**哪个机制拖后腿**，是节点 3 优化器精准出手的前提。

## 4. Seed 数与统计可靠性

CCC-GARCH DGP 结构固定，但每条合成路径的 Student-t innovation 序列是伪随机的。同一压力情景在不同 seed 下会给出显著不同的策略/等权 Sharpe——这不是 bug，是自然的采样噪声。

**关键教训**：`seeds=(11,)` 上 v4_1 pass_rate 100%（幸运路径），`seeds=(11,22,33,44,55,66,77)` 上真实 pass_rate 40%。**1-seed 会给出幻觉，7-seed 中位数才稳定**。所有门禁判定用 7 seed 中位数；筛选阶段（节点 3 Stage A）用 3 seed + margin slack 加速，Stage B 用 7 seed 严格。

## 5. 参数搜索空间：从 12 降到 6 主控

原始超参空间约 15 个连续参数 + 若干窗口/开关 ≈ 10^12 组合规模。节点 2 用 Morris 敏感度扫描（12 候选 × 4 轨迹 × 3 seed = 52 次真实 evaluate_full）筛出真正影响输出的：

**保留的 6 主控**（μ* 排序，出现在 ≥5/7 输出的 Top-6）：
- `max_def_extra`（=max_def-def_alloc，防御深度） — 7/7
- `def_alloc`（Layer3 基础防御配比） — 7/7
- `top_n`（进攻资产数） — 6/7
- `step_low`（防御一档触发阈） — 5/7，realized_annual 排第 1
- `step_delta_high`（防御两档间距） — 5/7
- `vol_window`（波动率因子窗口） — 4/7

**剔除的（Morris μ* 常年≈0）**：`vol_w`、`mom_w`、`mom_window`、`crisis_corr_threshold`、`crisis_corr_max_boost`、`stop_loss`。

**反直觉发现**：Layer1 打分权重（`vol_w`、`mom_w`）和 Layer3.5 相关性放大对**对抗鲁棒性** μ* 接近 0。这颠覆"vol_w 通过让选股偏向低波资产驱动 Sharpe"的直觉——**策略鲁棒性完全由 Layer3 防御深度/触发点决定，与 Layer1 打分权重正交**。

## 6. 节点 3 优化器与 v4_2 参数

6 维 LHS N=200 + 双阶段（Stage A 3-seed 粗筛 + margin slack -0.05；Stage B 7-seed 严格约束）。共 200 次评估约 1 小时。

**v4.2 相对 v4.1 的参数变化**（**"轻&快防御"取代"重&深防御"**）：

| 参数 | v4_1 | v4_2 | 方向 |
|---|---|---|---|
| `def_alloc` | 0.25 | 0.145 | 基础防御更低 |
| `step_low` | 0.15 | 0.095 | 触发更早 |
| `step_high` | 0.35 | 0.193 | 档间距更紧 |
| `max_def` | 0.95 | 0.811 | 峰值防御更低 |
| `vol_window` | 11 | 10 | vol 信号更快 |
| `top_n` | 2 | 2 | 不变 |

**为什么"轻&快"胜过"重&深"**：深防御（`max_def=0.95`）在合成的短 σ 冲击后仓位卡在 95% 防御，**错过回弹阶段**；轻&快在同 DD 约束下**保留更多 upside 参与**，同时更快转防御让 vol 冲击起始阶段损失变小。这个结论无法凭直觉得出，是节点 2 敏感度分析 + 节点 3 联合优化的**涌现结果**。

**v4_2 vs v4_1 全面指标（7-seed 严格）**：
- realized 年化 17.05% → 15.84%（**-1.21pp，换鲁棒的代价**）
- realized MaxDD 6.97% → 6.75%（+0.22pp 更好）
- realized Sharpe 1.610 → **1.635**（**+0.025 反涨**——波动降幅 > 收益降幅）
- 全情景 worst_DD 12.19% → **11.60%**（从>12% 跌回<12%）
- 5 机制 Sharpe 胜率全 100%（含 soft selection 意外过）
- verdict: FAIL → **PASS**

## 7. OOS 三通道验证 —— 判定过拟合的关键

节点 3 的严格 PASS 是在**训练集**（5 情景 × seeds 11-77）上得到的。若不做 OOS 验证，我们无法排除"节点 3 只是拟合了训练集的 5 情景 × 7 seeds"。节点 4 用**三条互相独立**的通道，每条从一个正交方向测试过拟合假设：

| 通道 | 独立性 | 测试对象 |
|---|---|---|
| **A. held-out 幅度** | 同 DGP，训练未见的扰动幅度 | σ×0.9/1.4, μoff×0.6/1.0, μdef×0.3/0.7, c×0.5/1.3, 复合 —— 10 情景 |
| **B. 独立 seed 集** | 同 DGP + 训练情景 + 完全独立 seeds 100-116（17 个） | seed 层面的过拟合签名 |
| **C. block bootstrap** | **完全独立 DGP 族**（非参数, 从真实周收益 block=8 重采样） | 跳出 CCC-GARCH 参数假设 |

**判定语义**（**这是本框架里最容易踩坑的地方**）：

判定必须分开两件事：

- **core 检查（直接测试过拟合假设）**：v4_2 相对 v4_1 在这条独立通道上有无劣化？教科书上过拟合的签名就是"在最独立的测试上崩最狠"，所以是**相对**问题不是**绝对**问题。
- **envelope 记录（策略族设计上界，独立记录不参与判定）**：v4_2 的 worst_DD 是否在绝对阈值内？这是**架构层面**问题（策略族本身能不能承受某种极端扰动），跟过拟合无关。

历史教训：早期版本的 verdict 函数把两者混在一起（"worst_DD ≤ 13% 才 PASS"），导致 v4_2 在 σ×1.4 这种极端幅度下自动判 FAIL，掩盖了"其实 v4_2 相对 v4_1 全线更好"的事实。修正后 core=相对不劣化, envelope=绝对上界, 分开报。

**实测结论**（v4.1 历史基线 → v4.2 生产，见 `output/adversarial/oos_validation.json`）：

| 通道 | pass_rate | worst_DD | avg_margin | core | envelope |
|---|---|---|---|---|---|
| A. held-out 幅度 | 50% → 70% | 16.99% → 17.47% | +0.003 → +0.063 | **PASS** | OUT |
| B. 独立 seed 集 | 100% → 100% | 12.09% → 11.63% | +0.182 → +0.232 | **PASS** | IN |
| C. block bootstrap | 87% → 90% | 18.89% → **13.97%**（-4.92pp）| +0.222 → +0.279 | **PASS** | OUT |

**结论 = TRUE_ROBUST**（三通道 core PASS，过拟合假设被反驳）。

**最强证据是通道 C**：block bootstrap 是**最独立**的验证（跳出 CCC-GARCH 参数族），如果 v4_2 是过拟合 CCC-GARCH 的表面胜利，通道 C 应该**塌陷最狠**。实际相反——通道 C 上 v4_2 收敛得**最漂亮**（DD -4.92pp）。这直接反驳"节点 3 只是拟合 CCC-GARCH"的可能。

## 8. 对抗空间的已知局限（架构层问题，非超参层）

**多 seed 只解决"同一 DGP 内的采样噪声"，不解决"DGP 本身选对了没"。** 本框架的对抗空间有以下不可回避的边界，需要在使用/汇报时**明确交代**：

1. **CCC-GARCH 不能产生 regime switching**：真实市场有明确的牛熊切换（宏观 regime jumps），CCC-GARCH 是**平稳 VAR + 局部 GARCH 波动聚集**的近似，不能内生地跳 regime。这类冲击（例如 2015 年 A 股熔断、2020 疫情 gap-down）在本框架的对抗测试里被结构性低估。
2. **CCC 而非 DCC**：本框架假设**条件相关阵不变**（Constant Conditional Correlation），但真实市场里危机期相关性会拉高（DCC 效应）。Layer 3.5 的相关性放大机制在合成情景下没被激活，跟这个假设有关。
3. **Student-t innovation 是对称的**：真实收益尾部**不对称**（大跌尾比大涨尾胖），本框架用对称 t 逼近，会低估左尾风险。
4. **Block bootstrap 只覆盖真实数据支撑**：通道 C 虽然独立于 CCC-GARCH，但它的分布仍然锚定在 2013-2026 这 13 年的经验分布上。**从未出现过的极端事件（如 2013 年之前的更狠熊市）不在支撑集里**。
5. **σ×1.4 是策略族上界不是本框架 bug**：通道 A 的极端场景（σ×1.4、μoff×0.6、quad_shock 四轴同时崩）下 worst_DD 17%+，v4_2 与 v4_1 都超 13%。这是"仓位≤100%、不加杠杆、依赖 vol 信号防御"这类策略族的**架构上界**——需要在 universe 层（加入短端国债 ETF/黄金等硬防御）或架构层（引入尾部对冲）解决，超参优化无能为力。

## 9. 参考实现与运行入口

- 对抗评估内核: `scripts/adversarial_robustness.py`（`robustness_score` / `STRESS_SCENARIOS` / `SCENARIO_MECHANISM`）
- 统一评估: `scripts/evaluate.py --config <yaml> --dmax 0.12 [--save-baseline] [--vs-baseline]`
- 维度约简: `scripts/dim_reduction.py --r 4 --seeds 11,22,33`
- 约束优化: `scripts/optimize.py --n 200 --k 15`
- OOS 验证: `scripts/oos_validation.py`
- 基线快照: `output/adversarial/baseline_metrics.json`（当前 = v4.2 生产）
- 生产 config: `config/strategy_v4_2.yaml`

## 10. 未来工作方向（这些 v4.0 框架无法覆盖）

1. **对抗空间扩展**：引入 regime-switching DGP（Markov switching + 双状态 GARCH），把 regime jumps 也纳入压力测试。
2. **Universe 层优化**：把资产池本身作为可优化维度（引入短端国债、黄金、境外指数 ETF 等硬防御标的），解决 σ×1.4 类"策略族上界"问题。
3. **尾部对冲结构**：Layer 4（可选）加入 put-protection 或 tail hedge sleeve，绕开纯多头策略的架构约束。
4. **在线适应**：目前 v4_2 参数固定；探索基于**近期实际扰动幅度**（rolling σ、rolling correlation）在线调整 def_alloc/step_low 的机制。

## 11. v4.3 案例：tapered-vol 与"过拟合对抗测试"的实证教训

v4.3 是本框架**最有价值的一次实证**——它把"优化目标错配会过拟合对抗测试"从抽象警告
变成了可复现的实测证据，并给出了修复方法（OOS 入环门）。

### 11.1 设计动机与跳变量化

rolling `vol_window` 有固有缺陷：最老一周滚出窗口时波动率阶跃跳变。tapered vol
（窗口内最老若干周线性降权，`calculate_volatility_tapered`）平滑了这个边界。实测纳指
vol 周环比跳变：均值 -27.3%、p95 **-42.2%**、max -13.0%，而 vol 绝对水平仅 +1.9%。
**设计论点成立：taper 确实消除跳变且不改变量级。**

### 11.2 教训一：max-年化目标过拟合对抗测试（可复现）

第一次 v4.3 用框架默认的 **max realized 年化** 目标 + 7 维 taper 搜索。结果：
- Stage B 只有 **1/20** 通过 7-seed 严格约束（危险信号：约束勉强可满足→唯一通过者是极端角点）。
- 该 config in-sample 对抗 PASS，realized 年化 15.47%，但 **realized Sharpe 仅 1.31**。
- **独立 OOS 三通道验证暴露过拟合**：block bootstrap（最独立 DGP 族）通过率从 v4.2 的
  90% **崩到 63%**，worst_DD 13.97%→17.57%；独立 seed 通道 margin 从 +0.232→+0.088。
- **过拟合签名教科书级**：越独立的测试劣化越狠（通道 A 同 DGP 还行，通道 C 独立 DGP 崩）。

根因：max-年化目标在 DD≤12% 约束边界上找最高收益角点，该角点是"threading 训练集
6 情景×7 seed 那根针"的极端配置，不泛化。

### 11.3 教训二：OOS 入环门（Stage C）修复

改进 `optimize.py`：
1. **目标改 max realized Sharpe**（realized 部分确定性、不依赖 seed，排序精确无噪声）。
2. **新增 Stage C OOS 泛化门**：Stage B 在训练 seed(11-77) 严格 PASS 的候选，必须**再在
   独立 seed(100-106) 上仍 PASS**（对抗 DD≤D_max & 硬机制 margin>0）才入围。这是把
   train/validation 拆分直接做进选择循环。

结果：Stage B 5/25 PASS，**Stage C 拒掉其中 4 个**（它们在独立 seed 上 composite margin
转负），只留 1 个真泛化的。最终 v4.3：Sharpe 1.488 / 年化 14.52% / MaxDD 5.84% /
Calmar 2.49；独立 OOS 三通道通过率**全 ≥ v4.2**（不再过拟合）。

**方法学结论**：当"in-sample 对抗 PASS 候选"很稀少（如 1/20）时，唯一通过者极可能过拟合；
健康的优化应有若干通过候选 + 一道独立 OOS 门筛掉不泛化的。这道门应成为框架**标准步骤**。

### 11.4 v4.2 与 v4.3 互不支配，双保留

| 指标 | v4.2 (rolling10) | v4.3 (taper14+7) |
|---|---|---|
| realized Sharpe | 1.635 | 1.488 |
| realized MaxDD | 6.75% | 5.84% |
| Calmar | 2.35 | 2.49 |
| vol 跳变 | 有 | 消除 -27~42% |
| OOS 三通道通过率 | 基线 | 全 ≥ v4.2 |

v4.3 用 realized Sharpe(-0.147) 换 无跳变 + 更低回撤 + 更高 Calmar，且不过拟合。**经 11.5 节
消融实验确认 taper 是因子级真实优势后，v4.3 已提升为默认生产配置**；前代 v4.2 降为已验证
替代配置（回退用 `rebalance_live.py --config config/strategy_v4_2.yaml`）。tapered vol 的
`vol_taper_window`/`vol_taper_len` 已纳入优化搜索空间（`optimize.py --space taper`）。

### 11.5 消融实验：隔离"因子 vs 方法"（rolling 用同一新方法重跑）

v4.2 用旧方法（max-年化，无 OOS 门），v4.3 用新方法（max-Sharpe + OOS 门）——直接对比会
混淆"因子变化"与"方法变化"。做控制变量：rolling 也用**同一套新方法**重跑
（`optimize.py --space rolling --objective sharpe --oos-seeds 100-106`, N=300）。

结果（同方法下）：
- **rolling**: Stage B 3/25 训练 seed PASS，**Stage C OOS 泛化门 0/3 通过**——3 个候选在独立
  seed 上机制 margin 全部强负（composite -0.153/-0.041/-0.296），无一泛化。
- **taper (v4.3)**: Stage C 1/5 通过，幸存者 OOS margin 为正。

**结论（隔离出的纯因子效应）**：在完全相同的激进 max-Sharpe 优化 + 同一道 OOS 门下，
rolling 找不到任何可泛化的鲁棒配置，唯 taper 能。机制：rolling vol 跳变 → 防御触发绑定
seed 特定跳变 → 换 seed 失效；taper 平滑 → 触发在结构性 vol 水平 → 跨 seed 泛化。

**方法与因子各司其职**：OOS 门（方法）负责揭穿过拟合（两种因子下都拒掉了过拟合候选，
质控不可缺）；taper（因子）负责在激进优化下仍留有可泛化鲁棒余量。反直觉副产品：v4.2
（rolling + max-年化）本身不过拟合，是因为 max-年化恰好落在温和 config；一旦改 max-Sharpe
逼近角点，rolling 立即过拟合——即"追 Sharpe 反而让 rolling 过拟合"。这坐实了 v4.3 的
taper 因子优势是真实的、非方法假象，构成把 v4.3 提升为生产的决定性依据。
