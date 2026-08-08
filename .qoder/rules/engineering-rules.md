---
description: "工程纪律与安全规则，适用于所有文件。Always Apply。"
globs: "*"
---

# 工程纪律

1. **Token/密钥安全**：提交前 `git diff --cached` grep 确认无 56 位 token / df5c 前缀 / TUSHARE_TOKEN=实值。`.env` 必须在 `.gitignore` 中。日志中 token 掩码为 `***TOKEN***`。

2. **实验脚本 `_exp_` 前缀**：不入 CI、不被生产代码 import。结论经 gate 后才提升为 src/ 生产代码。

3. **`run_backtest` 有副作用**：覆写 `output/report_v3_1.md`。实验后需 `git checkout output/report_v3_1.md` 恢复。

4. **weekly_refresh 幂等 + 零污染**：重复安全；任何步骤失败不产生部分提交；退出码 0-6 语义化。

5. **commit 粒度**：一个 commit 做一件完整的事（功能/修复/实验产物），不混合无关改动。message 用中文简述 + 英文类型前缀（feat/fix/chore/docs）。

6. **不 amend/force push main**：新 commit 追加，禁止改写历史。

# 代码审查分级

- **Critical（必须修复）**：违反前视偏差(expanding)、回测/实盘口径分叉、verify 不扣费、配置开关缺失
- **Warning（应修复）**：用 vol 而非 amount、数据拼接未处理、信号未经 gate 验证
- **Suggestion（考虑）**：文案不一致、实验脚本遗留问题、文档更新
