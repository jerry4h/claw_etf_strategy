# v4.6 定向 boost 分级应用闭环文档

> 任务链：B1 预研（7 变体矩阵）→ E2 补门禁（PE/R²）→ E3 集成 → 完整对抗验证管线 → 生产切换
> 结论：**GO — v4.6 切换为生产默认**（2026-08-11）。v4.5-pvd 降为前代已验证配置与影子对照。
> realized Sharpe 1.6094 / MaxDD 5.76%；grey_corr_combo 中位 MaxDD 12.79%→11.80%（回到 12% 红线内）。

---

## 1. 动机

v4.5 灰区中止文档（`docs/v4_5_grey_corr_abort.md`）确认：36 格实验穷尽标量/门控方案后，
"定向 boost（仅对进攻端降权、不推高防御绝对水平）"是唯一未被否决的方向。机制诊断：
bond_bear DGP 下进攻对相关中位 ≈0.40 恰落灰区，降阈值触发后经现状应用点（def += b）
把权重推向下跌中的债券 → MaxDD 恶化，构成灰区-bond_bear 物理冲突。

## 2. 预研（B1，`scripts/_exp_directed_boost_study.py`，`output/experiments/exp_directed_boost.md`）

触发器 T0（生产 thr0.60）/ T1（M-C thr0.45 slope0.75）× 应用点 C（现状）/ V1（比例式
def+=b(1−def)）/ V2（现金缓冲）/ V3（混合分级：corr>0.60 用 C，否则 V1），基座 v4.4：

- REF（T1+C）复现 M-C 失败模式：bond_bear 12.25% 破线 —— 机制诊断成立
- **T1+V3 唯一全门禁 PASS**：灰区缺口 12.80%→11.82%，bond_bear 10.93% 不破线，
  corr_regime_shift 7.85% 与基线逐位一致（显性危机满额保护无损），realized +0.006
- 关键机制结论：**分级应用而非一刀切**——纯定向（T0+V1/V2）在显性危机情景反而恶化 +1.3pp

## 3. E2 补门禁（PE / R² 两个 E1 GO 项）

| 方向 | 结果 | 依据 |
|---|---|---|
| PE 防御调制（pe_pct(t-1)>0.9 → def+δ） | **E2 PASS**：D05 ΔSharpe +0.0118 / D10 +0.0226，MaxDD 均改善，bootstrap 200 不劣 | `exp_pe_defense_e2.md` |
| R² 动量替换（(exp(slope×6)−1)×R² 替 mom6） | **E2 NO-GO**：OLS ΔSharpe −0.069 / MaxDD +1.52pp；WLS 更差 | `exp_r2_momentum_e2.md` |

## 4. E3 集成（T1）

- `src/strategy.py`：directed_boost 段（enabled/threshold 0.45/slope 0.75/corr_split 0.60）+
  pe_defense 段 + EWMA 前置校验（directed 依赖 crisis_corr_ewma_enabled）
- `src/engine_core.py`：`compute_crisis_boost_directed(w_rets, i, off_idx, config)` →
  (boost, corr_level)，EWMA(hl=8) 口径，窗口 [i-window,i) 无前视
- `src/backtest.py`：分级应用（corr_level>split → def+=b；否则 def+=b×(1−def)）+
  PE 调制注入（Layer3.5/M3 之后、止损之前；5 年滚动分位 + shift(1) + ffill asof 对齐）
- `scripts/rebalance_live.py`：镜像同步（共享 engine_core 函数；PE 加载失败降级警告）
- `config/strategy_v4_6.yaml`：directed_boost enabled=true；**pe_defense enabled=false（见 §5 裁出）**
- 测试：`tests/test_v46_directed_boost.py`（配置/分级边界/pin/支配断言）；字段完整性守卫同步

## 5. 完整对抗验证管线与 PE 裁出裁决

| # | 验证 | 结果 | 判定 |
|---|---|---|---|
| 1 | evaluate --corr-scenarios（7 seeds） | **verdict PASS**，worst MaxDD 11.80%≤12%，全机制门禁 pass_rate=1.0 | ✅ |
| 2 | 对抗改善断言 vs v4.5 | grey_corr_combo 12.79%→11.80%（−0.99pp）；bond_bear −0.04pp；corr_regime_shift/combo −0.22/−0.28pp | ✅ |
| 3 | block bootstrap 200（robustness_joint t2） | 胜率 96.0%，alpha P10 +0.091 | ✅ |
| 4 | OOS 三通道（v4.5→v4.6） | A core PASS（envelope 超界为两配置共有现象）；B 全 PASS；C marginal（pass_rate 73% vs 77%，avg_margin/worst_dd 不劣）——按 T2 裁决口径（bootstrap 主判据 + A3 交叉证据）接受 | ✅（带注记） |
| 5 | 联合鲁棒性 t1/t3 | t1 8 参无断崖全 PASS；t3 相对 alpha 胜率 96.5% / P10 +0.048 PASS（绝对判据 68.5% 为合成路径已知现象，对照口径一致） | ✅ |
| 6 | A3 量价联合 DGP 复核 | grey 情景 ΔSharpe −0.004（≥−0.03），MaxDD 14.14% vs 15.10% 改善 → 无结构性损害 | ✅ |
| 7 | rebalance_live --verify | Δ=0.0066 ≤0.02 | ✅ |

**PE 防御调制裁出**（管线中发现）：含 PE 的完整 v4.6 在 bond_bear（Sharpe 0.836 < 等权）与
stagflation（0.772 < 等权）触发 defense_asset/composite 硬门禁 FAIL；消融确认 PE 调制为罪魁
（db_only 配置 verdict PASS）。根因：历史高 PE 日历窗（2019+，占 24.7% 周）与合成压力路径
重叠时，+0.10 防御把权重推向压力中的防御资产。裁决：**PE 裁出 v4.6**，代码保留默认关，
留待重新设计（候选方向：危机状态条件交互、更低 δ、进攻端调制替代防御调制）。

**最终 v4.6 = directed boost only**：realized Sharpe 1.6094（vs v4.5 1.6028，+0.0066，
风险导向门禁：≥v4.5−0.01 ✓）、MaxDD 5.76%（−0.04pp ✓）、年化 15.46%。

## 6. 生产切换（阶段四）

- 10 个生产入口默认 config 切至 strategy_v4_6.yaml（run_backtest/rebalance_live/
  benchmark_compare/calc_performance/run_walkforward/evaluate/oos_validation/
  robustness_joint/weekly_refresh/gen_dashboard）
- 默认路径防回退断言更新（test_pvd_factor / test_v46_directed_boost）
- v4.5-pvd 影子对照：周度调仓并行输出 v4.5 决策，4-8 周后复盘
- baseline_metrics.json 更新为 v4.6 口径

## 7. 遗留与远期

| 项目 | 状态 |
|---|---|
| PE 防御调制重新设计 | 留档（E2 价值真实、E3 对抗失败；候选方向见 §5） |
| R² 动量 | NO-GO 归档（E1 IC 优势未转化为组合净收益） |
| A3 联合 DGP 入生产框架 | 预研可用（结构校验 19/20）；若未来量因子立项，迁入 data_manifold.py 注册 evaluate 通道 |
| v4.5 影子对照复盘 | 4-8 周后 |

## 8. 实验数据索引

| 文件 | 内容 |
|---|---|
| `scripts/_exp_directed_boost_study.py` / `output/experiments/exp_directed_boost.{md,json}` | B1 预研 7 变体矩阵 |
| `scripts/_exp_pe_defense_e2.py` / `exp_pe_defense_e2.{md,json}` | PE E2 门禁（PASS→E3 裁出） |
| `scripts/_exp_r2_momentum_e2.py` / `exp_r2_momentum_e2.{md,json}` | R² 替换 E2（NO-GO） |
| `scripts/_exp_v46_joint_dgp_check.py` / `exp_v46_joint_dgp_check.{md,json}` | A3 联合 DGP 复核 |
| `output/eval_v46_corr.json.log` / `eval_v45_corr.json.log` | evaluate --corr-scenarios 双配置原始输出 |
| `output/robustness/robustness_joint_t{1,2,3}_2026081*.json` | 联合鲁棒性 v4.6 结果 |
| `output/adversarial/oos_validation.json` | OOS 三通道（v4.5→v4.6） |
