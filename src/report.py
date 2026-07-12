"""绩效报告 & 可视化 — 从 BacktestResult 生成指标表和图表。"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.backtest import BacktestResult

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def _fig_to_base64(fig: plt.Figure) -> str:
    """将 matplotlib figure 转为 base64 PNG 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def generate_metrics_table(result: BacktestResult) -> str:
    """生成核心指标 Markdown 表格"""
    m = result.metrics
    return f"""## 核心指标

| 指标 | 数值 |
|------|------|
| 累计收益 | {m['total_return']*100:.1f}% |
| 年化收益 | {m['annual_return']*100:.2f}% |
| 最大回撤 | {m['max_drawdown']*100:.2f}% |
| 标准夏普 | {m['sharpe_ratio']:.3f} |
| 卡尔马比率 | {m['calmar_ratio']:.2f} |
| 年化波动率 | {m['annual_volatility']*100:.2f}% |
| 周胜率 | {m['win_rate']*100:.1f}% |
| 平均周收益 | {m['avg_weekly_return']*100:.2f}% |
| 回测周数 | {m['total_weeks']} |
| 防御周数 | {m['defensive_weeks']} |
| 调仓次数 | {m.get('rebalance_count', 0)} |
"""


def generate_annual_breakdown(result: BacktestResult) -> str:
    """生成年度收益分解 Markdown 表格"""
    df = result.nav_series.copy()
    df['year'] = df.index.year

    lines = ['## 年度收益分解\n', '| 年份 | 收益率 | 平均防御比例 | 止损周数 |', '|------|--------|-------------|---------|']

    for year, group in df.groupby('year'):
        yr_ret = (1 + group['weekly_return']).prod() - 1
        avg_def = group['def_ratio'].mean()
        stop_weeks = group['in_stop_loss'].sum()
        lines.append(f'| {year} | {yr_ret*100:+.1f}% | {avg_def*100:.0f}% | {int(stop_weeks)} |')

    return '\n'.join(lines)


def generate_drawdown_analysis(result: BacktestResult) -> str:
    """生成回撤分析 Markdown 表格"""
    df = result.nav_series.copy()

    # 找出每次回撤事件
    dd = df['drawdown']
    max_dd = dd.max()
    max_dd_date = dd.idxmax()

    lines = [
        '## 回撤分析\n',
        f'- 最大回撤: {max_dd*100:.2f}%（发生于 {max_dd_date.date()}）',
        f'- 当前回撤: {dd.iloc[-1]*100:.2f}%',
    ]
    return '\n'.join(lines)


def generate_contribution_analysis(result: BacktestResult) -> str:
    """生成 ETF 贡献分析：各 ETF 持有周数、平均权重"""
    from src.data_loader import ETFS
    df = result.nav_series.copy()

    lines = ['## ETF 持仓分析\n', '| ETF | 平均权重 | 持有周数（>0）|', '|-----|---------|------------|']

    for etf in ETFS:
        col = f'weight_{etf}'
        if col in df.columns:
            avg_w = df[col].mean()
            weeks_held = (df[col] > 0.001).sum()
            lines.append(f'| {etf} | {avg_w*100:.1f}% | {weeks_held} |')

    return '\n'.join(lines)


# === 图表（返回 base64 PNG） ===

def chart_nav_curve(result: BacktestResult) -> str:
    """净值曲线，返回 base64 PNG"""
    df = result.nav_series

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['nav'], color='#2196F3', linewidth=1.2, label='Strategy NAV')
    ax.fill_between(df.index, 1.0, df['nav'], alpha=0.1, color='#2196F3')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title('虾池ETF轮动策略 v3.0 — 净值曲线', fontsize=14)
    ax.set_ylabel('Net Asset Value')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    return _fig_to_base64(fig)


def chart_drawdown(result: BacktestResult) -> str:
    """回撤曲线，返回 base64 PNG"""
    df = result.nav_series

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df.index, 0, df['drawdown'] * 100, color='#F44336', alpha=0.3)
    ax.plot(df.index, df['drawdown'] * 100, color='#F44336', linewidth=0.8)
    ax.set_title('Drawdown', fontsize=14)
    ax.set_ylabel('Drawdown (%)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    return _fig_to_base64(fig)


def chart_annual_returns(result: BacktestResult) -> str:
    """年度收益柱状图，返回 base64 PNG"""
    df = result.nav_series.copy()
    df['year'] = df.index.year

    annual = df.groupby('year')['weekly_return'].apply(lambda x: (1 + x).prod() - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in annual.values]
    ax.bar(annual.index.astype(str), annual.values, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Annual Returns (%)', fontsize=14)
    ax.set_ylabel('Return (%)')
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    return _fig_to_base64(fig)


def chart_monthly_heatmap(result: BacktestResult) -> str:
    """月度收益热力图，返回 base64 PNG"""
    df = result.nav_series.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month

    monthly = df.groupby(['year', 'month'])['weekly_return'].apply(lambda x: (1 + x).prod() - 1) * 100
    monthly = monthly.unstack()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(monthly.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index.astype(str))
    ax.set_title('Monthly Returns Heatmap (%)', fontsize=14)
    plt.colorbar(im, ax=ax)

    return _fig_to_base64(fig)


def chart_risk_dashboard(result: BacktestResult) -> str:
    """风险仪表盘（回撤 + 波动率 + 防御比例），返回 base64 PNG"""
    df = result.nav_series

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 回撤
    axes[0].fill_between(df.index, 0, df['drawdown'] * 100, color='#F44336', alpha=0.3)
    axes[0].set_ylabel('DD (%)')
    axes[0].grid(True, alpha=0.3)

    # 纳指波动率
    axes[1].plot(df.index, df.get('nasdaq_vol', pd.Series(0, index=df.index)) * 100, color='#FF9800', linewidth=0.8)
    axes[1].axhline(y=20, color='gray', linestyle='--', linewidth=0.5, label='step_low=20%')
    axes[1].axhline(y=35, color='gray', linestyle='--', linewidth=0.5, label='step_high=35%')
    axes[1].set_ylabel('Nasdaq Vol (%)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 防御比例
    axes[2].fill_between(df.index, 0, df['def_ratio'] * 100, color='#2196F3', alpha=0.3)
    axes[2].set_ylabel('Defense (%)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Risk Dashboard', fontsize=14)

    return _fig_to_base64(fig)


def generate_full_report(
    result: BacktestResult,
    output_path: str | Path,
    include_charts: bool = True
) -> str:
    """
    生成完整 Markdown 报告，写入文件。

    Args:
        result: 回测结果
        output_path: 输出文件路径
        include_charts: 是否包含图表

    Returns:
        报告的 Markdown 文本
    """
    sections = [
        f'# 虾池ETF轮动策略 — 回测报告\n',
        f'策略: {result.config.name} v{result.config.version}\n',
        f'回测区间: {result.nav_series.index[0].date()} ~ {result.nav_series.index[-1].date()}\n',
        generate_metrics_table(result),
        generate_annual_breakdown(result),
        generate_drawdown_analysis(result),
        generate_contribution_analysis(result),
    ]

    if include_charts:
        charts = [
            ('净值曲线', chart_nav_curve(result)),
            ('回撤曲线', chart_drawdown(result)),
            ('年度收益', chart_annual_returns(result)),
            ('月度热力图', chart_monthly_heatmap(result)),
            ('风险仪表盘', chart_risk_dashboard(result)),
        ]
        for title, b64 in charts:
            sections.append(f'## {title}\n')
            sections.append(f'![{title}](data:image/png;base64,{b64})\n')

    report = '\n'.join(sections)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    return report
