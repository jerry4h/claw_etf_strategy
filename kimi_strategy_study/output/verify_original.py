import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest import _run_single_backtest

r = _run_single_backtest((0.15, 0.2, 0.0, 2, 0.40))
print(f"mom=0.15,vol=0.2,def=0.40: annual={r['annual_return']*100:.2f}% dd={r['max_drawdown']*100:.2f}%")

r2 = _run_single_backtest((0.30, 0.4, 0.0, 2, 0.30))
print(f"mom=0.30,vol=0.4,def=0.30: annual={r2['annual_return']*100:.2f}% dd={r2['max_drawdown']*100:.2f}%")
