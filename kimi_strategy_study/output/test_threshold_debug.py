import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import run_backtest

# 先关闭波动率调整，隔离问题
params = dict(mom_w=0.15, vol_w=0.2, val_w=0.0, top_n=2, defensive_allocation=0.40,
              enable_vol_adj=False)

for threshold in [0.0, 0.02, 0.05]:
    r = run_backtest(min_trade_threshold=threshold, stop_loss_offensive_n=2, **params)

    # 统计调仓
    trade_count = 0
    offensive_switches = 0
    OFFENSIVE = ["纳指ETF", "沪深300ETF", "黄金ETF"]

    for i in range(1, len(r)):
        prev = r.iloc[i-1]
        curr = r.iloc[i]

        has_trade = False
        for etf in ["纳指ETF", "红利低波ETF", "沪深300ETF", "黄金ETF", "国债ETF"]:
            if abs(curr[f"weight_{etf}"] - prev[f"weight_{etf}"]) > 0.0001:
                has_trade = True

        if has_trade:
            trade_count += 1

        for etf in OFFENSIVE:
            pw = prev[f"weight_{etf}"]
            cw = curr[f"weight_{etf}"]
            if (pw == 0 and cw > 0) or (pw > 0 and cw == 0):
                offensive_switches += 1

    final_val = r.iloc[-1]["portfolio_value"]
    annual = (final_val) ** (52 / len(r)) - 1
    max_dd = ((r["peak_value"] - r["portfolio_value"]) / r["peak_value"]).max()

    print(f"阈值={threshold:.0%} (关闭波动率调整):")
    print(f"  调仓周数: {trade_count}/{len(r)} ({trade_count/len(r)*100:.1f}%)")
    print(f"  进攻层切换: {offensive_switches}")
    print(f"  年化: {annual*100:.2f}%  回撤: {-max_dd*100:.2f}%")
