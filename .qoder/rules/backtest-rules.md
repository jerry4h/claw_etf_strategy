---
description: "回测引擎与信号验证规则，适用于回测逻辑修改、因子集成、verify 脚本变更。Always Apply。"
globs: "src/backtest.py,src/engine_core.py,src/factors.py,src/strategy.py,scripts/rebalance_live.py,scripts/evaluate.py,scripts/oos_validation.py,scripts/robustness_joint.py"
---

# 回测与信号准则

1. **动态门限必须无前视（expanding only）**：任何基于历史分布的阈值禁止全样本预计算——必须用 expanding window。违反产生系统性乐观偏差（实测约 19% Sharpe 增量曾来自前视泄漏）。

2. **回测与实盘共用实现**：门限/因子/条件逻辑抽为公共函数（如 engine_core），backtest 与 rebalance_live 复用。禁止两端各写一遍。

3. **verify 同口径扣费**：--verify 的 NAV 模拟必须扣 turnover×fee_rate，首周建仓费与引擎 last_alloc=zeros 一致。

4. **合成 DGP 对非价格因子不公平**：含量因子以 block bootstrap 为主判据，CCC-GARCH 仅供参考。

5. **信号必须经 gate 验证再集成**：预研(monkeypatch) → 门禁(IC/bootstrap) → 配置化集成。不可跳步。

# 策略集成准则

6. **辅助因子 tiebreaker 而非主驱动**：极低权重（≤0.15）条件激活打破平局，无条件叠加在窄截面中灾难性。

7. **至少双重条件门控**：非触发环境完全退化为 baseline。Bootstrap 中性验证。

8. **硬门控优于软门控**：软化代价超收益（Sharpe −0.036 + τ 过拟合面）。p75 硬切断是危机隔离，不可软化。

9. **配置开关隔离**：每个新功能 enabled=false 默认关闭。TestBaselineUnchanged（Sharpe pin）是 CI 硬约束。

10. **新因子后联合参数校验**：3×3 网格 + bootstrap 是最低成本确认手段。
