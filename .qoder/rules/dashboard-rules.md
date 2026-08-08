---
description: "看板与前端规则，适用于 gen_dashboard.py、index.html、CSS/JS 修改。Always Apply。"
globs: "scripts/gen_dashboard.py,scripts/check_dashboard.py,scripts/test_dashboard_responsive.py,index.html,dashboard/*"
---

# 看板与前端准则

1. **上线前 14 项门禁必须通过**：`python scripts/check_dashboard.py` exit 0 才允许 commit。weekly_refresh 已自动集成此校验。

2. **所有布局 grid 用 auto-fit**：`repeat(auto-fit, minmax(280px, 1fr))`——绝不使用固定列数 `repeat(2,1fr)` 或 `repeat(3,1fr)`，手机端会溢出。

3. **minmax 最小值 ≤ 300px**：确保 375px（iPhone SE）屏幕安全。当前统一用 280px。

4. **viewport meta 必须存在**：`<meta name="viewport" content="width=device-width, initial-scale=1">`——缺失时手机浏览器默认 980px 宽度，所有媒体查询失效。

5. **容器防溢出**：`.panel { max-width:100%; overflow:hidden; }`；body/html `overflow-x:hidden`。

6. **图表高度不可坍塌**：`.chart-stack { min-height:480px; }` + 子项各有 `min-height`。Chart.js 配置 `maintainAspectRatio:false`。

7. **tooltip 禁用动画**：所有图表统一 `interaction:{mode:'nearest', axis:'x', intersect:false}` + `tooltip.animation.duration=0`，消除密集数据点上的抖动。

8. **修改工作流**：编辑模板 → `gen_dashboard.py --preview` 本地预览 → `check_dashboard.py` 验证 → commit+push。禁止跳过预览直接推送。

9. **CDN 降级**：Chart.js 加载失败时显示文字提示而非白屏。`typeof Chart !== 'undefined'` 检查。
