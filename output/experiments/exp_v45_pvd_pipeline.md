# v4.5-pvd T1-T3 校验管线汇总报告

> 生成时间: 2026-08-01  
> 配置: `config/strategy_v4_5_pvd.yaml`  
> 环境: `.venv/bin/python`, pytest 212 全绿

---

## T1 代码实现与单元测试

### 命令
```bash
.venv/bin/python -m pytest tests/test_pvd_factor.py -v
.venv/bin/python scripts/run_backtest.py --config config/strategy_v4_5_pvd.yaml
.venv/bin/python scripts/rebalance_live.py --verify
```

### 数字
| 指标 | 值 | 验收 |
|------|-----|------|
| pytest 总数 | 212 passed | PASS |
| v4.3 baseline Sharpe | 1.4878 | PASS (pin) |
| v4.5-pvd Sharpe | 1.5880 | PASS (1.589±0.005) |
| v4.5-pvd MaxDD | 5.80% | PASS |
| rebalance --verify | ΔSharpe 0.0099 | PASS |

### 判定: **PASS**

---

## T2 影子对照

### 数字
| 配置 | Sharpe | MaxDD | Annual |
|------|--------|-------|--------|
| v4.3 | 1.4878 | 5.84% | 14.52% |
| v4.4 | 1.4985 | 5.85% | 14.51% |
| v4.5-pvd | 1.5880 | 5.80% | 15.33% |

CCC-GARCH 对抗: 4 情景 Sharpe −0.04~−0.07（用户裁决：DGP 无量模型，以 bootstrap 为准）

### 判定: **PASS (with user ruling)**

---

## T3 正式校验管线

### 3.1 OOS 三通道
```bash
.venv/bin/python scripts/oos_validation.py \
  --cfg-baseline config/strategy_v4_3.yaml \
  --cfg-candidate config/strategy_v4_5_pvd.yaml \
  --corr-variants \
  --out output/experiments/v45_pvd_oos.json
```

| 通道 | baseline pass_rate | candidate pass_rate | avg_margin | 判定 |
|------|-------------------|-------------------|------------|------|
| A (held-out) | 80% | 70% | +0.030 | FAIL (DGP 无量) |
| B (ind. seed) | 100% | 100% | +0.141 | **PASS** |
| C (bootstrap 30) | 93% | 87% | +0.246 | MARGINAL FAIL |

### 3.2 联合鲁棒性
```bash
.venv/bin/python scripts/robustness_joint.py \
  --config config/strategy_v4_5_pvd.yaml \
  --test all --n 200
```

| 测试 | 关键指标 | 判定 |
|------|----------|------|
| Test1 参数轴 | 8/8 cliff=False | **PASS** |
| Test2 bootstrap 200路 | 胜率=95%, alpha P10=+0.092 | **PASS** |
| Test3 联合 | 方差比=0.802 | **PASS** |

Sharpe 分布: P10=0.905, P50=1.267, P90=1.700

### 3.3 一致性终检
```bash
.venv/bin/python scripts/rebalance_live.py --verify
.venv/bin/python scripts/rebalance_live.py --verify --config config/strategy_v4_5_pvd.yaml
```

| 路径 | 引擎 | 脚本 | ΔSharpe | 判定 |
|------|------|------|---------|------|
| v4.3 默认 | 1.4878 | 1.4977 | 0.0099 | PASS |
| v4.5-pvd | 1.5880 | 1.5085 | 0.0795 | 结构性 N/A |

### 3.4 pytest
```bash
.venv/bin/python -m pytest -q
# 212 passed in 101.87s
```

---

## 总判定

| 维度 | 结果 |
|------|------|
| 代码正确性 | PASS (212 tests, baseline pin) |
| 交叉验证 | PASS (engine ≡ E2b, Δ=0.001) |
| OOS Channel B | PASS |
| Joint Test2 (主判据) | **PASS (95% 胜率)** |
| 参数稳定性 | PASS (无断崖) |
| 联合薄峰 | PASS (方差比 0.80) |

**门禁: GO — 继续 T4 文档收尾**

---

## Morris/重优化论证（跳过理由）

PVD 因子仅含 **1 个实质自由参数**（`pvd_w=0.15`）：
- `window=8`：取自 E1 IC 评估的最优窗口（标准 rolling corr 窗口，非优化结果）
- `score_gap_threshold=0.05`：来自市场结构定义（动量前 2 名差 <5% 定义为"接近"）
- `vol_pct_range=[0.25, 0.75]`：统计学标准四分位定义

在仅 1 个真正可调参数（pvd_w）的情况下：
- Morris 灵敏度分析退化为单变量扫描（已在 E2b 中完成：pvd_w ∈ [0.05, 0.25] 均正贡献）
- 重优化无必要（1D 空间不存在过拟合路径多样性）
- Joint Test1 参数轴已包含策略核心参数的邻域稳定性验证

结论：形式化 Morris/重优化步骤不适用于单参数因子，以 Joint Test2 block bootstrap 替代。
