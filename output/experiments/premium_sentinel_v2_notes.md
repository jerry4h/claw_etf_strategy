# 溢价哨兵 v2 实现说明 (任务28: R1/R2 回落防线代码化 + 数据可持续性)

日期: 2026-07-31 | 代码: `scripts/premium_sentinel.py` | 测试: `tests/test_premium_sentinel.py`
规则出处: E4 实证 (`output/experiments/premium_e4_collapse.md`) / SOP §6.2 溢价回落防线

## 1. 数据源验证结论 (tushare token 过期后的公开源方案)

| 数据 | 渠道 | 验证结果 |
|---|---|---|
| close (场内市价) 日频历史 | 东财 push2his kline | ⚠️ 本机稳定被拒 (RemoteDisconnected ×5, https/http 均失败, 疑 IP 级限流)——保留为第一优先但不可依赖 |
| 〃 | **新浪日K** (`quotes.sina.cn/.../getKLineData`, scale=240) | ✅ 主用。与 tushare 缓存 close 逐日核验 **10/10 一致**, 且含最新交易日增量 |
| 〃 | 腾讯日K (`web.ifzq.gtimg.cn/.../fqkline/get`, day 字段) | ✅ 备选, 数值一致 |
| nav (单位净值) 全历史 | **东财 pingzhongdata** (`fund.eastmoney.com/pingzhongdata/{code}.js`, Data_netWorthTrend) | ✅ 513100 全历史 3183 行 (2013-04-25 起), 含 T-1 净值 |
| 份额日频 (R2) | **上交所公开查询** (`query.sse.com.cn/commonQuery.do`, sqlId=`COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L`, 需 Referer) | ✅ TOT_VOL (万份) 与 tushare fund_share 缓存核验一致 (946511.06 @07-30)。**仅沪市**; 深市 (15xxxx) 无公开日频源, 纯缓存+降级提示 |
| 份额当日快照 | 东财 push2 quote 字段 f84 | ✅ 可作旁证 (无历史) |

结论: **R1/R2 数据在 token 过期后均可持续** —— close/nav 走"缓存打底+公开源增量"，份额走
"缓存打底+上交所增量"。全部实现于哨兵内部，联网仅在 `--premium-check`/`--collapse-check`
显式路径; 份额增量逐日 `sleep(0.6s)` 且最多补 6 日 (礼貌限频); 全程不使用/不打印 tushare token
(tushare 分支仅在 token 尚有效时作为最新溢价的第一优先, 与 v1 相同)。

## 2. 接口用法

```bash
# 调仓主命令: 哨兵 advise + 回落防线一并输出 (默认路径无 --premium-check 时零网络零变化)
.venv/bin/python scripts/rebalance_live.py --premium-check
# 单独跑回落防线
.venv/bin/python scripts/premium_sentinel.py --collapse-check
```

Python 层 (供测试/复用, 均可离线):
- `fetch_premium_history(code, online)` → 缓存 CSV (`data/experiments/tushare_cache/premium_{tag}.csv`) 打底 + 公开源增量 append-only (增量口径同缓存: close/最近 nav asof≤7 天)
- `collapse_metrics(rows)` → p5 (5日均值 min_periods=3) / dd20 (近20个有效 p5 点含当日的峰值−当日, 口径同 `_exp_premium_e4.py`) / R1 主口径 X=2pp 与备选 X=1.5pp 判定与 gap
- `fetch_share_history` / `share_metrics` → 份额 5 日扩张 (单日比率>1.5或<0.5 视拆分折算) / R2≥+5%
- `collapse_check` / `collapse_advise` / `collapse_report` → 汇总判定→文本; **任何失败只降级不抛异常**

阈值修订: `advise()` 的 THRESHOLD_LOW 2.0%→**1.5%**, 对齐 E3 纪律"溢价>1.5% 新增走场外"
(SOP §3); 被证伪的 p*≈2.1% 框架引用已全部删除。

## 3. 当前实跑输出 (2026-07-31)

```
-- 溢价回落防线 (E4/SOP §6.2: R1 峰值回撤 + R2 份额预警; 仅提示不自动切换) --
  数据: 缓存3210行@2026-07-30, 增量1日(源:sina); 末日 2026-07-31 溢价 11.02%
  p5=11.21%  20日峰值p5=11.21% (2026-07-31)  dd20=0.00pp
  ✅ R1(X=2pp) 未触发: 距触发还差 2.00pp 回撤
     备选口径 X=1.5pp: 未触发, 差 1.50pp
  ✅ R2 未触发: 份额5日变动 +0.00% (<+5%) [缓存2863行@2026-07-30, SSE增量1日]
```

dd20=0.00pp / 份额 5 日 0.00%, 与 E4 报告"当前状态: 两信号均未触发 (溢价仍在峰值区)"吻合。

## 4. 铁律验证

- **默认路径零变化**: `rebalance_live.py --verify` 输出改前后 `cmp` **逐字节一致** (392 字节,
  exit=0 ✅); 哨兵仍为 `--premium-check` 分支内惰性导入, 模块导入期零网络 (测试锁定)。
- **失败不中断**: `collapse_report` 双重兜底; R1/R2 各自独立降级 (无缓存+离线 → "R1 降级…
  人工核查 / R2 降级…份额数据不可用"), 均有离线测试覆盖。
- **测试**: 全量 pytest **200 passed** (基线 186 + 本任务新增 14: p5/dd20 数值、R1 触发/
  未触发/备选口径区/p5≤1% 地板、数据不足、R2 触发/拆分折算/降级、离线不联网守卫、report 兜底)。
