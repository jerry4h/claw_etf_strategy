---
description: "数据口径与完整性规则，适用于任何涉及数据加载、因子计算、数据管线变更的代码。Always Apply。"
globs: "src/data_loader.py,src/factors.py,src/national_team.py,scripts/fetch_*,scripts/update_*,scripts/weekly_refresh.py"
---

# 数据准则

1. **前复权价 × 原始量/额口径自洽**：log_return 用前复权 NAV，amount/volume 用不复权原始值。绝不用不复权价格算收益、绝不用复权值算成交额比率。

2. **成交额(amount)优于成交量(vol)**：vol（手）受拆分/分红影响产生虚假脉冲（如 512890 拆分当周 +81%），amount（千元）天然免疫。涉及量的因子一律用 amount。

3. **红利低波(512890)数据拼接感知**：日频 OHLC/amount 仅 2019-01-18 起可用，之前由 H20269 指数反推仅有周频 close。新因子聚合必须处理此分界（pre-2019 设 NaN 或退化），聚合后 weekly close 与生产 NAV 逐行一致（容差 ≤0.001）。

4. **tushare 无 ETF 周频接口**：周频从日频手算聚合（周五锚点、H/L 极值、amount 累计）。日频是万能底仓。

5. **数据敏感性分级**：原始份额不入 git（data/national_team/ gitignore）；聚合产出可入库；tushare_cache 为离线复现工件直接提交。

6. **数据源三级链容灾**：tushare 主用 → 新浪日K → baostock/腾讯。单源失效自动降级不中断。
