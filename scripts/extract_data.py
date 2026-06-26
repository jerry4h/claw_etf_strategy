"""Extract detailed data for all 6 configs: metrics, year-by-year, market state, drawdown."""
import sys, json, os
sys.path.insert(0, '/home/ubuntu/claw_etf_strategy')

from src.strategy import load_config
from src.backtest import run_backtest
import pandas as pd
import numpy as np

configs = [
    ("v2.6 all-on", "strategy_v2_6"),
    ("v2.3 baseline", "strategy_v2_3"),
    ("ablation A (Fix#1)", "strategy_v2_6_ablation_a"),
    ("ablation B (Fix#2)", "strategy_v2_6_ablation_b"),
    ("ablation C (Fix#3)", "strategy_v2_6_ablation_c"),
    ("v2.5 P1 all-on", "strategy_v2_5_p1_all_on"),
]

results = {}

for label, config_file in configs:
    config = load_config(f'config/{config_file}.yaml')
    result = run_backtest(config)
    
    m = result.metrics
    df = result.nav_series.copy()
    
    # Core metrics
    core = {}
    for k in ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio',
              'simple_sharpe', 'calmar_ratio', 'annual_volatility', 'win_rate',
              'total_weeks', 'defensive_weeks']:
        if k in m:
            core[k] = round(float(m[k]), 6)
    
    # Year-by-year
    df['year'] = df.index.year
    yearly = {}
    for yr, grp in df.groupby('year'):
        yr_ret = (1 + grp['weekly_return']).prod() - 1
        avg_def = grp['def_ratio'].mean()
        in_sl = grp['in_stop_loss'].sum()
        ms_counts = grp['market_state'].value_counts().to_dict()
        yearly[str(yr)] = {
            'return': round(float(yr_ret), 6),
            'avg_def': round(float(avg_def), 4),
            'stop_loss_weeks': int(in_sl),
            'market_states': {str(k): int(v) for k, v in ms_counts.items()}
        }
    
    # Max DD event
    dd = df['drawdown']
    max_dd_val = float(dd.max())
    max_dd_date = str(dd.idxmax().date())
    
    # Market state distribution
    ms_global = {str(k): int(v) for k, v in df['market_state'].value_counts().to_dict().items()}
    
    # Drawdown event context
    max_dd_idx = dd.idxmax()
    event = {
        'date': max_dd_date,
        'dd': max_dd_val,
        'nav': float(df.loc[max_dd_idx, 'nav']),
        'peak': float(df.loc[max_dd_idx, 'peak']),
        'def_ratio': float(df.loc[max_dd_idx, 'def_ratio']),
        'market_state': str(df.loc[max_dd_idx, 'market_state']),
    }
    
    results[label] = {
        'config': config_file,
        'metrics': core,
        'yearly': yearly,
        'max_dd_event': event,
        'market_state_distribution': ms_global,
    }
    print(f"{label}: annual={core['annual_return']*100:.2f}%, dd={core['max_drawdown']*100:.2f}%, sharpe={core['sharpe_ratio']:.3f}")

# Save
out_path = '/home/ubuntu/claw_etf_strategy/output/all_ablation_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
