# 周内波动(High/Low)探索立项与中止复盘

> 任务 #43（预研）→ #44（E0 数据基建）→ #45（E1 信息增量评估）→ #46（E2 回测对比）  
> 结论：**E1 门禁 NO-GO + E2 分资产方案 NO-GO，课题中止**。  
> 生产维持 CC-tapered vol (window=14, taper=7) 不变。

---

## 1. 课题背景

v4.3/v4.4 策略使用 close-to-close tapered 波动率因子，仅消费周收盘价信息。
理论上 Parkinson 估计器利用周内 High/Low 可获得 ~5× 统计效率提升。

本探索评估：日频 OHLC → 周频聚合 → Parkinson/GK 估计器能否为策略 4 个波动率消费层
（L1 评分、L2 倒数加权、L3 防御、DefAlloc 红利分配）带来可操作增量。

### 1.1 立项前判定 GO 的依据

| 条件 | 状态 |
|------|------|
| 数据可得性 | ✓ tushare_cache 5 只 ETF 日频 OHLC 完整 |
| 理论基础 | ✓ Parkinson 无偏、效率约为 CC 的 5.2 倍 |
| 策略消费点 | ✓ 4 层独立使用 vol factor（可分层消融） |
| 成本 | ✓ 纯离线实验，不改生产代码 |

---

## 2. E0 数据基础设施（任务 #44）

**脚本**: `scripts/_exp_hl_vol_study.py`

### 主要成果

- 日频 OHLC → 周频前复权聚合（per-week NAV anchoring 消除累积舍入误差）
- 一致性校验全 PASS（max_diff=0.000000，5 只 ETF × 677 周）
- 红利低波(512890) 退化正确处理：2019-01-18 前无真实 OHLC → H=L=C=NAV → Parkinson=0
- 真实 OHLC 覆盖：纳指/中证500/国债 677 周完整，黄金 666 周，红利低波 385 周（2019+）
- Parkinson / GK / CC-tapered 三个估计器输出 shape (677, 5) 一致

---

## 3. E1 离线信息增量评估（任务 #45）— NO-GO

**脚本**: `scripts/_exp_hl_vol_e1.py`  
**产物**: `output/experiments/exp_hl_vol_e1.{md,json}`

### 关键数据

| ETF | corr(P,CC) | 噪声比 P/CC | 判定 |
|-----|-----------|-------------|------|
| 纳指ETF (513100, QDII) | 0.30 | 4.00 | ✗ 完全失效 |
| 中证500ETF (510500) | 0.93 | 0.50 | ✓ 优质 |
| 黄金ETF (518880) | 0.86 | 0.66 | ✓ 优质 |
| 红利低波ETF (512890) | 0.86 | 0.87 | ✓ (2019+ 段) |
| 国债ETF (511010) | 0.43 | 2.83 | ✗ 不理想 |

### NO-GO 根因

1. **QDII 溢价扭曲**：纳指 ETF 盘中 High/Low 受场内溢价/折价极端扰动（溢价尖刺 → ln(H/L)
   暴增 → Parkinson 噪声放大 4 倍、与 CC-vol 相关仅 0.30）。GK 估计器也无法挽救（corr=0.42）。

2. **国债特殊性**：国债 ETF 日内波动极低（年化 vol ~2-3%），High/Low 变化被 tick 最小精度
   截断，Parkinson 信号被量化噪声主导（corr=0.43，噪声比 2.83）。

3. **极端事件领先率不足**：3 只门控 ETF × 3 次危机 = 9 事件，P 领先≥1 周仅 4 次（44.4%），
   未达 66.7% 门禁。

4. **策略权重结构**：纳指占进攻权重 ~33% 且 vol 驱动 L3 防御核心决策（nasdaq_vol 三段式），
   即便境内 ETF 优质，纳指列不可替换→整体方案不可行。

### 门禁 #1 三条判据

| 条件 | 要求 | 实际 | 判定 |
|------|------|------|------|
| corr ∈ [0.60, 0.95] | 全部门控 ETF | 纳指=0.30 | FAIL |
| P 领先 ≥ 2/3 事件 | ≥66.7% | 44.4% | FAIL |
| noise(P) < noise(CC) | 全部 P/CC < 1 | 纳指=4.00 | FAIL |

---

## 4. E2 分资产方案回测（任务 #46）— NO-GO

**脚本**: `scripts/_exp_hl_vol_e2.py`  
**产物**: `output/experiments/exp_hl_vol_e2.{md,json}`

### 方案设计

既然纳指/国债 Parkinson 不可用，尝试"分资产 Mixed"方案：
- 纳指 + 国债：保持 CC-tapered vol
- 中证500 + 黄金 + 红利低波：替换为 Parkinson vol

### 关键结果

| 实验 | Sharpe | MaxDD | ΔSharpe vs 基线 |
|------|--------|-------|-----------------|
| Baseline (全 CC) | 1.488 | 5.84% | — |
| **Mixed** | 1.107 | 20.18% | **-0.381** |
| Full-P (全 Parkinson) | 1.066 | 25.55% | -0.422 |
| Mixed-L1 (仅评分) | 1.420 | 6.18% | -0.067 |
| Mixed-L3 (仅 M3) | 1.374 | 6.78% | -0.114 |
| Mixed-Def (仅红利) | 1.148 | 22.26% | -0.340 |

对抗 7-seed 中位：Baseline Sharpe 1.323 → Mixed 1.263 (deficit -0.060)

### NO-GO 根因

1. **红利低波 pre-2019 退化致 DefAlloc 崩溃**：2013-2019 段 Parkinson=0 → `compute_dynamic_hongli(0)` 
   返回最大 hongli_ratio → 防御分配严重失灵 → MaxDD 从 5.84% 暴涨至 22.26%

2. **即便排除 DefAlloc，其余层也无正向收益**：Mixed-L1（评分层）ΔSharpe=-0.067，
   Mixed-L3（M3 防御）ΔSharpe=-0.114

3. **CC-tapered vol 在当前架构下已是更优信号**：策略对 vol 的消费是非线性的（三段式防御、
   动态红利比、分值差 margin），Parkinson 与 CC 的微小量级差异在非线性映射后被放大为
   决策偏差而非精度提升

### 门禁 #2 四条判据

| 条件 | 要求 | 实际 | 判定 |
|------|------|------|------|
| Sharpe 改善 (Mixed) | ≥+0.02 | -0.381 | FAIL |
| Sharpe 改善 (最优层) | ≥+0.02 | -0.067 | FAIL |
| MaxDD 恶化 | ≤+0.3pp | +14.34pp | FAIL |
| 对抗中位 Sharpe | ≥基线 | -0.060 | FAIL |

---

## 5. 理论预期 vs 实际偏差

| 理论预期 | 实际 | 偏差原因 |
|----------|------|----------|
| Parkinson 效率 ~5× CC | corr 仅 0.30-0.93 | (a) QDII 溢价使 H/L 不代表真实价格边界 |
| 周内信息补充日间信息 | 噪声反而增大 | (b) 国债 tick 量化+纳指溢价 = 非价格信号 |
| 分资产可绕过失效 ETF | 仍劣化 | (c) 数据覆盖缺口(红利低波 pre-2019) + 策略非线性依赖 |
| 替换 vol 列即可注入增量 | 无正向效果 | CC-tapered vol 已通过 taper 权重实现信息平滑 |

**核心结论**：Parkinson 理论效率 5× 在实践中被三重因素消解——
(a) QDII 溢价噪声使场内 H/L 不等于 NAV 真实极值  
(b) 红利低波/国债数据覆盖或精度不足  
(c) 策略对 vol 的非线性消费将微小偏差放大为决策错误

---

## 6. 结论与处置

| 项目 | 处置 |
|------|------|
| 生产配置 | 维持 v4.3/v4.4 CC-tapered vol (window=14, taper=7)，不变 |
| 实验脚本 | 保留：`scripts/_exp_hl_vol_{study,e1,e2}.py` |
| 实验产物 | 保留：`output/experiments/exp_hl_vol_e{1,2}.{md,json}` |
| 预研文档 | 保留：`output/research_task43_intraweek_vol_exploration.md` |
| tushare_cache | 保留：`data/experiments/tushare_cache/`（5 只 ETF 日频 OHLC） |

---

## 7. 远期方向（留档不执行）

1. **纳指 → 境内宽基替换**：若未来将纳指 ETF 替换为沪深 300 等无 QDII 溢价的境内宽基，
   消除 H/L 溢价扭曲后，Parkinson 可重新评估（预期 corr >0.85）

2. **红利低波全历史 OHLC**：若获得 512890 在 2019 前的真实日频 OHLC（如从 Wind/Choice），
   消除合成退化段，仅对 DefAlloc 层可局部再验证

3. **INAR 日频 OHLC 到周频的替代聚合**：当前 per-week factor anchoring 已解决精度问题，
   但若引入真日内数据（如 1min K 线），可考虑 realized vol 等更高阶估计器

4. **架构改造**：若未来策略 vol 消费从"三段式硬阈值"改为"连续函数映射"，
   Parkinson 的平滑性优势可能重新显现（当前硬阈值放大微小偏差）

**本版不做。**仅作为远期可选课题存档。

---

## 8. 实验数据索引

| 文件 | 内容 |
|------|------|
| `scripts/_exp_hl_vol_study.py` | E0: 数据预处理与 Parkinson/GK 估计器实现 |
| `scripts/_exp_hl_vol_e1.py` | E1: 离线信息增量评估（6 项分析 + 门禁 #1） |
| `scripts/_exp_hl_vol_e2.py` | E2: 策略回测 A/B 对比（7 实验组 + 7-seed 对抗 + 门禁 #2） |
| `output/experiments/exp_hl_vol_e1.json` | E1 结构化数据 |
| `output/experiments/exp_hl_vol_e1.md` | E1 分析报告 |
| `output/experiments/exp_hl_vol_e2.json` | E2 结构化数据 |
| `output/experiments/exp_hl_vol_e2.md` | E2 对比报告 |
| `output/research_task43_intraweek_vol_exploration.md` | 任务 #43 预研报告 |
| `output/TASK43_SUMMARY.txt` | 任务 #43 摘要 |
