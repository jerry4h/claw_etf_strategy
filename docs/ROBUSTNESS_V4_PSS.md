# 鲁棒性评估体系 v4 — 方法论修正版

> 版本：v4.0（方法论修正）
> 日期：2026-06-17
> 设计者：quant-se
> 来源：v3 报告审计 + PBO 方法论错配 + 用户对 MC 悬崖和参数范围的讨论

---

## 一、v3 方法论问题诊断

### 问题 1：PBO 方法论错配

PBO（Bailey et al. 2017）的 CSCV 算法设计用于**多策略参数搜索场景**——你跑了 N 个变体，IS 最好的在 OOS 是否还能好。

对单固定参数策略，CSCV 把 649 周切成 16 段做时间 IS/OOS 对照。结果是：
- IS 包含 2016-2017 蓝筹牛市（策略弱项）→ OOS 包含 2022 熊市（策略强项）→ 秩相关 → -0.99
- **这不叫参数过拟合，这叫策略在不同牛熊市中自然地表现不同。**

**结论：对单策略鲁棒性评估，PBO 是方法论错配。移除，用 PSS（参数稳定性评分）替代。**

---

### 问题 2：MC 悬崖的实用性

MC 显示 ±15% 扰动后 52% 的情况 DD 突破 15%。但悬崖集中在两个特定方向：
- `mom_w` +5% → DD 跳
- `vol_w` -5% → DD 跳

用户指出："IS 和 OOS 的超参不一定需要完全相同，我只需要让超参数在一定波动范围内就可以了。MC 的用处就是说明，这个超参数的选择范围大概是多大。"

这意味着：**MC 不应只看通过率，而应看分布——参数扰动后年化收益和 DD 的分布范围。**

---

### 问题 3：momentum_window=8 是否过拟合？

用户问："如果选取 7 会怎么样？"

OAT 已经覆盖了——momentum_window 的 7 级测试中，-15% 就是 window≈7。答案在 OAT 数据里。但当前报告没有呈现"从最佳点偏离后的完整表现分布"。

---

## 二、v4 改进：PSS 替代 PBO

### PSS（Parameter Stability Score）— 参数稳定性评分

**定义**：对 400 次 MC 扰动结果，统计核心指标的分布。

### 计算（零额外计算量，直接用 MC 已有数据）

```python
def compute_pss(mc_details: list[dict]) -> dict:
    returns = [r['annual_return'] for r in mc_details]
    dds = [r['max_drawdown'] for r in mc_details]
    sharpes = [r['sharpe_ratio'] for r in mc_details]

    return {
        # 核心：三指标分布
        'return_p10':  np.percentile(returns, 10),
        'return_p50':  np.median(returns),
        'return_p90':  np.percentile(returns, 90),
        'dd_p10':      np.percentile(dds, 10),
        'dd_p50':      np.median(dds),
        'dd_p90':      np.percentile(dds, 90),
        'sharpe_p10':  np.percentile(sharpes, 10),
        'sharpe_p50':  np.median(sharpes),
        'sharpe_p90':  np.percentile(sharpes, 90),

        # 离散度
        'return_cv':   np.std(returns) / np.mean(returns),
        'dd_cv':       np.std(dds) / np.mean(dds),
        'sharpe_cv':   np.std(sharpes) / np.mean(sharpes),

        # 计数
        'n_total':     len(mc_details),
    }
```

### 门控

| 评级 | 条件 | 含义 |
|:---:|------|------|
| 🟢 | P10年化 > 10% AND P90回撤 < 15% | **两个尾部都在安全区** — 参数扰动后 90% 的情况符合目标 |
| 🟡 | P50年化 > 10% AND P50回撤 < 15% | **中位数安全但尾部漂移** — 多数情况 OK，部分扰动超标 |
| 🔴 | P50不满足 | 参数扰动后大部分情况不达标 |

### 相比 PBO 的优势

| 维度 | PBO (v3) | PSS (v4) |
|------|----------|----------|
| 回答的问题 | "时间切分下 IS/OOS 一致吗？"（策略间对比） | "参数扰动后收益分布如何？"（策略内稳定性） |
| 适用场景 | 多策略搜索 | **单策略参数鲁棒性** ✅ |
| 方法论匹配 | ❌ 错配（测的是牛熊切换，不是参数稳定） | ✅ 直接测参数扰动后的表现 |
| 假阳性风险 | 高——任何牛熊不对称的策略都判死 | 低——只看参数扰动后的实际表现 |
| 额外计算量 | ~1000 次回测 | **零**（直接复用 MC 400 次数据） |
| 直观性 | 难解释（秩相关 → PBO 概率） | **直觉清晰**（P90 DD = 15.5% → "10% 的情况 DD 超 15%") |

---

## 三、v4 五指标体系

| # | 指标 | 问什么 | 🟢/🟡/🔴 | 计算方法 |
|---|------|--------|:---:|---------|
| ① | **DSR** | 真 alpha 还是运气？ | >0.95 / >0.85 / <0.85 | B&LdP 2014, n_trials=52 |
| ② | **PSS** | 参数扰动后表现分布？（替代 PBO） | P10年化>10% + P90DD<15% / 中位数通过 / 中位数不通过 | MC 400 次数据统计 |
| ③ | **MC 生存率** | 参数扰动后双约束通过比例？ | >80% / 50-80% / <50% | 年化>10% AND DD<15% |
| ④ | **WF 基准相对胜率** | 各窗口跑赢等权基准？ | >80% / 60-80% / <60% | 9窗 Walk-Forward |
| ⑤ | **SPS** | 最倒霉起点能赚吗？ | 最差起点年化>5% / 0-5% / <0% | 114起点×3年 |

**PSS 和 MC 的关系**：PSS = 分布概要（"参数飘了，90% 的情况 DD 不超过多少？"），MC = 通过率（"参数飘了，多少比例还满足目标？"）。两者互补：PSS 告诉你范围，MC 告诉你概率。

### 综合判定

```
5/5 绿 → 🟢 实盘可上（高度鲁棒）
4/5 绿 → 🟢 实盘可上（一个黄可接受）
3/5 绿 + 0 红 → 🟡 可小资金试
≥2 红 → 🔴 需改进
```

---

## 四、OAT 保留：回答"每个参数偏离后怎么样"

OAT（7 参数 × 7 级别 = 49 次/策略）直接回答用户的问题：

| 用户问题 | OAT 回答 |
|---------|---------|
| "momentum_window 选 7 会怎样？" | OAT momentum_window -15% → window≈7 → 结果在看 |
| "momentum_threshold 从 -0.075 改到 -0.065？" | OAT -15% → -0.064 → 悬崖区 |
| "vol_w 降到 0.27 呢？" | OAT -10% → 0.27 → DD 跳到啥 |

OAT 的 7 级设计正好覆盖了用户关心的 ±15% 范围。报告模板保留 OAT 章节不变。

---

## 五、实现改动清单

### `src/robustness.py`

#### 改动 1：新增 `compute_pss()` 函数
位置：紧跟 `run_mc_survival_test()` 之后。纯函数，输入 mc_details，输出 dict。

#### 改动 2：移除 `compute_pbo_simplified()` 和所有 PBO 引用
`RobustnessResult.pbo` 字段改为 `RobustnessResult.pss: dict | None = None`

#### 改动 3：`evaluate_robustness()` 调整
- 移除 `--pbo` flag 和相关逻辑
- 添加 PSS 计算（从已有 MC 数据，零额外回测）

#### 改动 4：`_traffic_light()` 新增 'pss' 种类
```python
elif kind == 'pss':
    return_p10 = value.get('return_p10', 0)
    dd_p90 = value.get('dd_p90', 1)
    if return_p10 > 0.10 and dd_p90 < 0.15:
        return '🟢'
    elif value.get('return_p50', 0) > 0.10 and value.get('dd_p50', 1) < 0.15:
        return '🟡'
    else:
        return '🔴'
```

#### 改动 5：`generate_robustness_report()` 调整
- 移除 PBO 章节
- 新增 PSS 章节：五数概括（P10/P50/P90）+ 离散度（CV）
- 五指标综合表：PBO → PSS

### `scripts/run_robustness.py`
- 移除 `--pbo` 参数及相关引用
- 默认运行 PSS（零额外配置——MC 跑完自动算）

---

## 六、验收

```bash
python scripts/run_robustness.py \
    --configs config/strategy_v2_3_cap040.yaml,config/strategy_v2_3_cap040_D4_tuned.yaml \
    --labels "v2.3 基线","v2.3+cap040+D4 tuned" \
    --output output/robustness_v4/ \
    --n-mc 400 \
    --perturbation 0.15 \
    --oat \
    --sps \
    --n-wf 9
```

输出 `output/robustness_v4/ROBUSTNESS_COMPARISON_REPORT.md`：
- 五指标：DSR / PSS / MC / WF / SPS（无 PBO）
- PSS 章节含 P10/P50/P90 分布 + CV
- OAT 敏感度
- SPS 起点敏感性
- 综合判定

---

*本文档替代 `docs/ROBUSTNESS_V3_PBO_SPS.md` 中的 PBO 部分。SPS 保持不变。*
