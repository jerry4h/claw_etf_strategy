---
name: strategy-research
description: Conduct quantitative strategy research for the ETF rotation system. Follows the project's proven workflow - monkeypatch experiment → E1 signal evaluation (IC/IR) → E2 backtest A/B comparison → go/no-go gate → optional formal integration. Use when the user asks to explore a new factor, test a hypothesis, evaluate a signal, or do strategy research.
---

# Strategy Research Workflow

## When to Use

- User wants to explore a new factor or signal
- Evaluating whether data/feature has predictive value
- Any "能不能做"/"是否有收益"/"试试看" type questions about strategy enhancements

## Proven Research Pipeline

```
E0 (Data)  →  E1 (Signal Quality)  →  E2 (Backtest A/B)  →  Gate  →  E3 (Integration)
```

### E0: Data Infrastructure
- Confirm data availability (tushare, cache, alternative sources)
- Validate data quality (NaN, jumps, coverage)
- Red利低波 pre-2019 awareness (Rule 3)
- Amount over vol (Rule 2)

### E1: Signal Quality Evaluation
- rank_IC (Spearman rank correlation with forward returns)
- IR = mean(IC) / std(IC)
- Orthogonality: corr with existing factors (momentum, volatility)
- Noise ratio: std(Δsignal/signal)
- **Gate**: |IC| ≥ 0.03, |t-stat| ≥ 1.5, orthogonal to existing factors

### E2: Strategy Backtest (Monkeypatch)
- Script: `scripts/_exp_{name}.py` (never modify src/ during research)
- Baseline vs experimental group
- Block bootstrap robustness (200 paths, preserve real relationships)
- **Gate**: ΔSharpe ≥ +0.01, ΔMaxDD ≤ +0.3pp, bootstrap neutral

### E3: Formal Integration (only after GO)
- Config switch isolation (enabled: false by default)
- Tiebreaker mode (weight ≤ 0.15, conditional activation)
- TestBaselineUnchanged pin must pass
- Full OOS + joint robustness pipeline

## Critical Rules (always apply)

1. **No lookahead bias** (Rule 6): Dynamic thresholds must use expanding window only
2. **Amount not vol** (Rule 2): Volume data uses amount (千元) to avoid split artifacts
3. **Monkeypatch first** (Rule 17): `_exp_` prefix scripts, don't touch src/ until gate passes
4. **Block bootstrap for non-price factors** (Rule 9): CCC-GARCH DGP is unfair to volume/share signals
5. **Double shift for delayed data** (Rule 6): Fund share T+1 → shift(1) in backtest

## Historical Lessons

- PVD (volume signal): E1 GO (IC=0.053) → E2 NO-GO (linear) → E2b GO (conditional tiebreaker)
- Parkinson vol: FAIL (QDII premium distorts H/L, corr=0.30)
- Realized vol: FAIL (estimates same target as CC-vol, "更精确≠更快")
- National team share: NO-GO for strategy (IC=0.015 insignificant), useful as observation tool
- Grey zone M-C: 36-grid structural impossibility (bond_bear vs grey protection conflict)

## Output Conventions

- Script: `scripts/_exp_{topic}_study.py`
- Report: `output/experiments/exp_{topic}.md` + `.json`
- Never commit experiment data to git during research phase
