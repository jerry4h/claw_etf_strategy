#!/usr/bin/env python3
"""Test dynamic hongli_ratio for 红利低波 defense allocation."""
import sys, os
sys.path.insert(0, '/home/ubuntu/claw_etf_strategy')

import subprocess, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path('/home/ubuntu/claw_etf_strategy')
config_path = str(ROOT / 'config' / 'strategy_v3_0_final.yaml')

# First, get the baseline (current fixed 50/50)
r = subprocess.run([
    ROOT / '.venv' / 'bin' / 'python3', '-c',
    'import os;os.environ["MPLBACKEND"]="Agg";'
    'import sys;sys.path.insert(0,"/home/ubuntu/claw_etf_strategy");'
    'from src.backtest import run_backtest;from src.strategy import load_config;from pathlib import Path;'
    'cfg=load_config(str(Path("/home/ubuntu/claw_etf_strategy/config/strategy_v3_0_final.yaml")));'
    'r=run_backtest(cfg);m=r.metrics;'
    'print(f"BASELINE: Sharpe={m[\'sharpe_ratio\']:.4f} AnnRet={m[\'annual_return\']*100:.2f}% DD={m[\'max_drawdown\']*100:.2f}%")'
], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
print(r.stdout.strip())
if r.stderr: print("ERR:", r.stderr[:200])

# Now modify backtest.py to use dynamic hongli_ratio
# Read the file and add the dynamic ratio computation before the defense allocation
bp = ROOT / 'src' / 'backtest.py'
text = bp.read_text()

# Find the defense allocation section and add dynamic ratio computation before it
old = """        # 防御层分配：第一个防御ETF得 hongli_ratio，其余平分 (1-hongli_ratio)
        if def_idx:
            alloc[def_idx[0]] = def_ratio * config.hongli_ratio"""

new = """        # 动态hongli_ratio：红利低波根据自身momentum+vol分配防御配额
        # score = mom4 * MOM_W - vol11 * VOL_W (同进攻层评分公式)
        if def_idx and len(def_idx) >= 2:
            hl_mom = mom_values[i, def_idx[0]]
            hl_vol = vol_values[i, def_idx[0]]
            if not np.isnan(hl_mom) and not np.isnan(hl_vol):
                hl_score = d1_mom_w * hl_mom - d1_vol_w * hl_vol
                # score=-0.20→ratio=0.0, score=+0.05→ratio=0.70
                eff_hl_ratio = max(0.0, min(0.70, (hl_score + 0.20) / 0.25 * 0.70))
            else:
                eff_hl_ratio = config.hongli_ratio
        else:
            eff_hl_ratio = config.hongli_ratio
        # 防御层分配
        if def_idx:
            alloc[def_idx[0]] = def_ratio * eff_hl_ratio"""

if old in text:
    text = text.replace(old, new)
    bp.write_text(text)
    print("\nbacktest.py: dynamic hongli_ratio added")
else:
    print("\nbacktest.py: CAN'T FIND defense allocation section!")

# Also update rebalance_live.py
rp = ROOT / 'scripts' / 'rebalance_live.py'
rt = rp.read_text()

old_r = """    alloc = {e: def_r / len(DEFENSIVE) for e in DEFENSIVE}"""

new_r = """    # 动态hongli_ratio
    if len(DEFENSIVE) >= 2:
        hl_mom = m4['红利低波ETF'].iloc[i] if hasattr(m4, 'iloc') else m4['红利低波ETF'][i]
        hl_vol = v20['红利低波ETF'].iloc[i] if hasattr(v20, 'iloc') else v20['红利低波ETF'][i]
        if not np.isnan(hl_mom) and not np.isnan(hl_vol):
            hl_score = MOM_W * hl_mom - VOL_W * hl_vol
            eff_hl_ratio = max(0.0, min(0.70, (hl_score + 0.20) / 0.25 * 0.70))
        else:
            eff_hl_ratio = HONGLI_RATIO
    else:
        eff_hl_ratio = HONGLI_RATIO
    alloc = {DEFENSIVE[0]: def_r * eff_hl_ratio} if len(DEFENSIVE) > 0 else {}
    if len(DEFENSIVE) > 1:
        alloc[DEFENSIVE[1]] = def_r * (1 - eff_hl_ratio)"""

if old_r in rt:
    rt = rt.replace(old_r, new_r)
    rp.write_text(rt)
    print("\nrebalance_live.py: dynamic hongli_ratio added")
else:
    print("\nrebalance_live.py: CAN'T FIND defense allocation!")

# Run backtest with dynamic hongli_ratio
print("\nRunning backtest with dynamic hongli_ratio...")
r = subprocess.run([
    ROOT / '.venv' / 'bin' / 'python3', '-c',
    'import os;os.environ["MPLBACKEND"]="Agg";'
    'import sys;sys.path.insert(0,"/home/ubuntu/claw_etf_strategy");'
    'from src.backtest import run_backtest;from src.strategy import load_config;from pathlib import Path;'
    'cfg=load_config(str(Path("/home/ubuntu/claw_etf_strategy/config/strategy_v3_0_final.yaml")));'
    'r=run_backtest(cfg);m=r.metrics;'
    'print(f"DYNAMIC_HL: Sharpe={m[\'sharpe_ratio\']:.4f} AnnRet={m[\'annual_return\']*100:.2f}% DD={m[\'max_drawdown\']*100:.2f}%")'
], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
print(r.stdout.strip())
if r.stderr: print("ERR:", r.stderr[:200])

# Run rebalance_live --verify
print("\nRunning rebalance_live --verify...")
r = subprocess.run([
    ROOT / '.venv' / 'bin' / 'python3',
    str(ROOT / 'scripts' / 'rebalance_live.py'), '--verify'
], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
for line in r.stdout.split("\n"):
    if any(x in line for x in ["Sharpe", "通过", "年化", "DD"]):
        if "指标" not in line:
            print("  ", line.strip())