"""统一回测引擎 — 所有回测（单次、网格搜索、消融实验）均走此引擎。"""

from __future__ import annotations

import itertools
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import (
    ETFS, OFFENSIVE_IDX, DEFENSIVE_IDX,
    load_nav_data, load_pe_percentile, resample_weekly
)
from src.factors import compute_all_factors
from src.strategy import (
    StrategyConfig, load_config,
    score_offensive, select_top, calculate_defense_ratio,
    allocate, check_rebalance, check_stop_loss
)
from src.utils import (
    annualize_return, compute_max_drawdown, compute_sharpe,
    compute_simple_sharpe, compute_calmar, compute_annual_volatility
)


@dataclass
class BacktestResult:
    """回测结果"""
    nav_series: pd.DataFrame      # 逐周净值、峰值、回撤、仓位
    metrics: dict                 # 年化、回撤、夏普、胜率等
    config: StrategyConfig        # 使用的配置（可复现）
    weekly_records: list[dict] = field(default_factory=list)  # 逐周明细


def run_backtest(
    config: StrategyConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    data_path: str | Path | None = None,
    pe_path: str | Path | None = None
) -> BacktestResult:
    """
    统一回测引擎。

    流程：
    1. data_loader.load_nav_data() + load_pe_percentile()
    2. data_loader.resample_weekly()（若需要）
    3. factors.compute_all_factors()
    4. 逐周循环：评分 → 选股 → 防御 → 止损 → 分配 → 调仓 → 扣费
    5. 汇总绩效指标

    Args:
        config: 策略配置
        start_date: 回测起始日期（覆盖 config）
        end_date: 回测结束日期（覆盖 config）
        data_path: NAV 数据路径（覆盖 config）
        pe_path: PE 数据路径（覆盖 config）

    Returns:
        BacktestResult
    """
    # === 1. 数据加载 ===
    _nav_path = data_path or config.nav_path
    _pe_path = pe_path or config.pe_path

    # 解析相对路径
    project_root = Path(__file__).resolve().parent.parent
    if not Path(_nav_path).is_absolute():
        _nav_path = project_root / _nav_path
    if _pe_path and not Path(_pe_path).is_absolute():
        _pe_path = project_root / _pe_path

    nav_df = load_nav_data(_nav_path)
    weekly_nav = resample_weekly(nav_df, anchor=config.anchor)

    # PE 数据
    pe_df = None
    if _pe_path and Path(_pe_path).exists():
        pe_df = load_pe_percentile(_pe_path)

    # === 2. 日期过滤 ===
    start = start_date or config.start_date
    end = end_date or config.end_date
    if start:
        weekly_nav = weekly_nav[weekly_nav.index >= pd.to_datetime(start)]
    if end:
        weekly_nav = weekly_nav[weekly_nav.index <= pd.to_datetime(end)]

    # === 3. 因子计算 ===
    config_dict = {
        'factors': {
            'mom_window': config.mom_window,
            'vol_window': config.vol_window,
            'pe_window_years': config.pe_window_years,
        }
    }
    factors = compute_all_factors(weekly_nav, pe_df, config_dict)
    momentum = factors['momentum']
    volatility = factors['volatility']

    # === 4. 转为 numpy 加速回测 ===
    w_prices = weekly_nav.values
    w_index = weekly_nav.index
    n_weeks = len(w_index)
    w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]

    # 因子转为 numpy（对齐回测逻辑）
    mom_values = momentum.values
    vol_values = volatility.values

    # ETF 索引
    NASDAQ_IDX = 0  # 纳指用于 vol 三段式防御

    # === 5. 逐周回测 ===
    start_idx = config.vol_window  # 需要 vol_window 周预热
    nav = 1.0
    peak = 1.0
    last_alloc = np.zeros(5)
    max_dd = 0.0

    # 止损状态
    in_stop_loss = False
    stop_loss_weeks = 0

    weekly_records = []

    for i in range(start_idx, n_weeks - 1):
        date = w_index[i]

        # --- 评分 ---
        scores_vec = np.full(5, -np.inf)
        for j in OFFENSIVE_IDX:
            mom_val = mom_values[i, j]
            vol_val = vol_values[i, j]
            if not np.isnan(mom_val) and not np.isnan(vol_val):
                scores_vec[j] = config.mom_w * mom_val - config.vol_w * vol_val

        # --- 选 top_n ---
        off_scores = [(scores_vec[j], j) for j in OFFENSIVE_IDX if not np.isnan(scores_vec[j])]
        off_scores.sort(key=lambda x: x[0], reverse=True)
        selected_off = [j for _, j in off_scores[:config.top_n]]

        # --- vol 三段式防御 ---
        nasdaq_vol = vol_values[i, NASDAQ_IDX]
        def_ratio = calculate_defense_ratio(nasdaq_vol, config)

        # --- 止损兜底 ---
        if not in_stop_loss and check_stop_loss(nav, peak, config.stop_loss):
            in_stop_loss = True
            stop_loss_weeks = 0

        if in_stop_loss:
            def_ratio = max(def_ratio, 0.95)
            stop_loss_weeks += 1
            if stop_loss_weeks >= config.recovery_weeks:
                in_stop_loss = False

        # --- 构建仓位 ---
        alloc = np.zeros(5)
        alloc[DEFENSIVE_IDX[0]] = def_ratio * config.hongli_ratio
        alloc[DEFENSIVE_IDX[1]] = def_ratio * (1 - config.hongli_ratio)
        if selected_off:
            for j in selected_off:
                alloc[j] = (1 - def_ratio) / len(selected_off)
        else:
            # 极端情况：无进攻层 ETF 可选 → 全额防御
            alloc[DEFENSIVE_IDX[0]] = config.hongli_ratio
            alloc[DEFENSIVE_IDX[1]] = 1 - config.hongli_ratio

        # --- 调仓阈值检查 ---
        if i > start_idx:
            max_change = np.max(np.abs(alloc - last_alloc))
            if max_change < config.rebalance_threshold:
                alloc = last_alloc.copy()

        # --- 调仓手续费 ---
        turnover = np.sum(np.abs(alloc - last_alloc))
        fee_cost = turnover * config.fee_rate

        # --- 周收益 ---
        wret = sum(
            alloc[j] * w_rets[i, j]
            for j in range(5)
            if not np.isnan(w_rets[i, j])
        )
        nav *= (1 + wret - fee_cost)
        peak = max(peak, nav)

        dd = (peak - nav) / peak
        if dd > max_dd:
            max_dd = dd

        # --- 记录 ---
        record = {
            'date': w_index[i + 1],  # 下一周才知道收益
            'nav': nav,
            'peak': peak,
            'drawdown': dd,
            'weekly_return': wret - fee_cost,
            'def_ratio': def_ratio,
            'in_stop_loss': in_stop_loss,
            'nasdaq_vol': nasdaq_vol,
            'turnover': turnover,
            'fee_cost': fee_cost,
        }
        # 记录仓位
        for k, etf in enumerate(ETFS):
            record[f'weight_{etf}'] = alloc[k]

        weekly_records.append(record)
        last_alloc = alloc.copy()

    # === 6. 构建 nav_series ===
    nav_df_result = pd.DataFrame(weekly_records)
    nav_df_result['date'] = pd.to_datetime(nav_df_result['date'])
    nav_df_result = nav_df_result.set_index('date')

    # === 7. 计算指标 ===
    metrics = compute_metrics(
        nav_df_result, config.risk_free_rate
    )

    return BacktestResult(
        nav_series=nav_df_result,
        metrics=metrics,
        config=config,
        weekly_records=weekly_records
    )


def compute_metrics(
    nav_series: pd.DataFrame,
    risk_free_rate: float = 0.025
) -> dict:
    """
    从净值序列计算核心指标。

    Args:
        nav_series: 回测结果 DataFrame（含 nav, weekly_return 等列）
        risk_free_rate: 年化无风险利率

    Returns:
        {
            "total_return": 4.40,
            "annual_return": 0.1406,
            "max_drawdown": 0.0821,
            "sharpe_ratio": 1.102,
            "simple_sharpe": 1.60,
            "calmar_ratio": 1.71,
            "win_rate": 0.583,
            "avg_weekly_return": 0.0027,
            "std_weekly_return": 0.018,
            "annual_volatility": 0.13,
            "total_weeks": 674,
            "defensive_weeks": 135,
            "rebalance_count": 455,
        }
    """
    n_weeks = len(nav_series)
    final_nav = nav_series['nav'].iloc[-1]
    total_return = final_nav - 1.0

    weekly_returns = nav_series['weekly_return']

    annual_ret = annualize_return(total_return, n_weeks)
    max_dd = nav_series['drawdown'].max()
    sharpe = compute_sharpe(weekly_returns, risk_free_rate)
    simple_sharpe = compute_simple_sharpe(weekly_returns)
    calmar = compute_calmar(annual_ret, max_dd) if max_dd > 0 else float('inf')
    annual_vol = compute_annual_volatility(weekly_returns)
    win_rate = (weekly_returns > 0).mean()
    avg_wret = weekly_returns.mean()
    std_wret = weekly_returns.std()

    # 防御周数
    def_weeks = int(nav_series['def_ratio'].gt(0.25).sum())

    # 调仓次数
    if 'turnover' in nav_series.columns:
        rebalance_count = int((nav_series['turnover'] > 0).sum())
    else:
        rebalance_count = 0

    return {
        'total_return': total_return,
        'annual_return': annual_ret,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'simple_sharpe': simple_sharpe,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'avg_weekly_return': avg_wret,
        'std_weekly_return': std_wret,
        'annual_volatility': annual_vol,
        'total_weeks': n_weeks,
        'defensive_weeks': def_weeks,
        'rebalance_count': rebalance_count,
        'final_nav': final_nav,
    }


def _run_single_grid(params: tuple, base_config: StrategyConfig) -> dict | None:
    """单组参数回测（供多进程使用）"""
    param_dict, = params
    # 复制配置并覆盖参数
    cfg = StrategyConfig(
        name=base_config.name,
        version=base_config.version,
        mom_w=param_dict.get('mom_w', base_config.mom_w),
        vol_w=param_dict.get('vol_w', base_config.vol_w),
        top_n=param_dict.get('top_n', base_config.top_n),
        mom_window=base_config.mom_window,
        vol_window=base_config.vol_window,
        pe_window_years=base_config.pe_window_years,
        def_alloc=param_dict.get('def_alloc', base_config.def_alloc),
        step_low=param_dict.get('step_low', base_config.step_low),
        step_high=param_dict.get('step_high', base_config.step_high),
        max_def=param_dict.get('max_def', base_config.max_def),
        hongli_ratio=base_config.hongli_ratio,
        rebalance_threshold=param_dict.get('rebalance_threshold', base_config.rebalance_threshold),
        fee_rate=base_config.fee_rate,
        anchor=base_config.anchor,
        stop_loss=base_config.stop_loss,
        recovery_weeks=base_config.recovery_weeks,
        nav_path=base_config.nav_path,
        pe_path=base_config.pe_path,
        start_date=base_config.start_date,
        end_date=base_config.end_date,
        risk_free_rate=base_config.risk_free_rate,
    )

    result = run_backtest(cfg)
    if result.nav_series.empty:
        return None

    return {
        **param_dict,
        'annual_return': result.metrics['annual_return'],
        'max_drawdown': result.metrics['max_drawdown'],
        'sharpe_ratio': result.metrics['sharpe_ratio'],
        'calmar_ratio': result.metrics['calmar_ratio'],
        'win_rate': result.metrics['win_rate'],
        'total_weeks': result.metrics['total_weeks'],
    }


def grid_search(
    param_space: dict[str, list],
    base_config_path: str | Path,
    n_jobs: int = -1,
    filters: dict | None = None
) -> pd.DataFrame:
    """
    网格搜索。每个参数组合调用一次 run_backtest()。
    使用 multiprocessing 并行。

    Args:
        param_space: {"mom_w": [0.30, 0.35, 0.40], "step_high": [0.35, 0.40, 0.45]}
        base_config_path: 基准配置文件路径
        n_jobs: 并行进程数（-1 = CPU 数的一半）
        filters: 结果过滤器 {"annual_return_gt": 0.10, "max_drawdown_lt": 0.15}

    Returns:
        DataFrame, 每行一组参数 + 绩效指标
    """
    base_config = load_config(base_config_path)

    # 生成参数组合
    keys = list(param_space.keys())
    values = list(param_space.values())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    n_workers = n_jobs if n_jobs > 0 else max(1, int(multiprocessing.cpu_count() * 0.5))
    print(f"网格搜索: {len(combinations)} 组参数, {n_workers} 个进程")

    results = []
    params_for_pool = [(combo,) for combo in combinations]

    try:
        with multiprocessing.Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(
                lambda p: _run_single_grid(p, base_config), params_for_pool
            )):
                if r is not None:
                    results.append(r)
                if (i + 1) % 100 == 0:
                    print(f"  进度: {i + 1}/{len(combinations)}")
    except Exception as e:
        print(f"多进程失败 ({e})，切换为串行")
        for params in params_for_pool:
            r = _run_single_grid(params, base_config)
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # 应用过滤器
    if filters:
        if 'annual_return_gt' in filters:
            df = df[df['annual_return'] > filters['annual_return_gt']]
        if 'max_drawdown_lt' in filters:
            df = df[df['max_drawdown'] < filters['max_drawdown_lt']]

    # 按年化收益排序
    df = df.sort_values('annual_return', ascending=False).reset_index(drop=True)

    return df
