"""鲁棒性评估模块 — 三指标简化版 (v2)

实现：
  ① DSR（Deflated Sharpe Ratio）— 真实 alpha 概率
  ② MC 生存率 — 参数扰动后盈利概率（v2: 年化>10% AND DD<15%）
  ③ 基准相对胜率 — Walk-Forward 相对等权基准的超额比例
  ④ OAT 多级敏感度 — One-At-a-Time 参数敏感度分析（v2 新增）

参考：
  - Bailey & López de Prado (2014). "The Deflated Sharpe Ratio."
  - docs/ROBUSTNESS_V2_SPEC.md
"""

from __future__ import annotations

import json
import math
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import ETFS, load_nav_data, resample_weekly
from src.strategy import StrategyConfig, load_config
from src.utils import compute_sharpe


# ── Pure-numpy fallbacks for scipy functions ──────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation using numpy only (no scipy needed).

    Returns (rho, p_value). For large n, uses t-distribution approximation.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    # Compute ranks (average rank for ties)
    from collections import Counter

    def _rankdata(a: np.ndarray) -> np.ndarray:
        """Assign ranks to data, averaging ranks for ties."""
        n_items = len(a)
        sorter = np.argsort(a)
        ranks = np.empty(n_items, dtype=float)
        # Count occurrences of each value
        counts = Counter()
        for v in a:
            counts[float(v)] += 1
        rank = 1.0
        i = 0
        while i < n_items:
            val = float(a[sorter[i]])
            cnt = counts[val]
            avg_rank = rank + (cnt - 1) / 2.0
            for j in range(cnt):
                ranks[sorter[i + j]] = avg_rank
            rank += cnt
            i += cnt
        return ranks

    x_ranks = _rankdata(x)
    y_ranks = _rankdata(y)

    # Pearson correlation on ranks
    xm = x_ranks - np.mean(x_ranks)
    ym = y_ranks - np.mean(y_ranks)
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    if r_den == 0:
        return 0.0, 1.0
    rho = r_num / r_den

    # p-value using t-distribution approximation
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t_stat = rho * np.sqrt((n - 2) / (1 - rho ** 2))
        # Use regularized incomplete beta for t-distribution CDF
        # For simplicity, use normal approximation for large n
        if n > 20:
            p = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
        else:
            # Conservative: set p = 0.05 if rho is moderately large
            p = 0.05 if abs(rho) > 0.5 else 0.5

    return float(rho), float(min(p, 1.0))

# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class RobustnessResult:
    """鲁棒性评估结果"""
    dsr: float                               # Deflated Sharpe Ratio
    mc_survival_rate: float                  # MC 生存率 (0~1)
    benchmark_relative_win_rate: float       # 基准相对胜率 (0~1)
    strategy_config: str                     # 策略配置名
    strategy_metrics: dict                   # 基准回测指标
    oat_sensitivity: dict | None = None      # OAT 多级敏感度结果 (v2)
    pss: dict | None = None                  # PSS 参数稳定性评分 (v4)
    starting_point_sensitivity: dict | None = None  # SPS 起点敏感度 metrics (v3)
    full_grid: dict | None = None            # Phase 6 全参数网格结果 (v4)
    details: dict = field(default_factory=dict)  # 详细数据


@dataclass
class GridPointConfig:
    """Phase 6 网格格点配置 — 描述一个测试点的参数覆盖"""
    param_name: str          # 测试的参数名
    level: float             # 扰动级别（连续参数=百分比，离散参数=实际值）
    actual_value: float      # 实际参数值
    param_overrides: dict    # 需要覆盖的参数字典 {param_name: value}


@dataclass
class GridPointResult:
    """Phase 6 网格格点回测结果"""
    param_name: str          # 参数名
    level: float             # 扰动级别
    actual_value: float      # 实际参数值
    sharpe: float            # 全周期 Sharpe
    annual_return: float     # 年化收益
    max_drawdown: float      # 最大回撤
    relative_sharpe: float   # 策略 Sharpe - 等权基准 Sharpe
    mc_survival_rate: float  # 局部 MC 生存率
    mc_details: list[dict] = field(default_factory=list)  # 局部 MC 明细


# ── DSR (Deflated Sharpe Ratio) ──────────────────────────────────────────────────────────────────────

def compute_dsr(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float
) -> float:
    """计算 Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Args:
        sharpe: 观测到的年化 Sharpe
        n_trials: 尝试的变体数（多重测试矫正）
        n_obs: 周收益观测数
        skew: 周收益偏度
        kurtosis: 周收益峰度

    Returns:
        DSR 概率 (0~1)
    """
    euler = 0.5772156649

    # Expected max SR under null (N independent trials)
    # E[max(SR_N)] ≈ √(2·ln(N)) · (1 − γ·SR̂ + (γ²−1)/4 · SR̂²)
    e_max_sr = np.sqrt(2 * np.log(max(n_trials, 2))) * (
        1 - euler * sharpe + (euler**2 - 1) / 4 * sharpe**2
    )

    # Standard error of SR
    # SE(SR̂) ≈ √((1 + 0.5·SR̂² − γ₃·SR̂ + (γ₄−3)/4 · SR̂²) / n_obs)
    variance = (1 + 0.5 * sharpe**2 - skew * sharpe +
                (kurtosis - 3) / 4 * sharpe**2)
    if variance <= 0:
        # Degenerate case: very high kurtosis or negative variance
        variance = 1.0 / n_obs  # fallback to baseline SE
    se_sr = np.sqrt(variance / n_obs)

    # DSR = P[SR > E[max(SR_N)]] = 1 - Φ((E[max] - SR̂) / SE)
    z_stat = (sharpe - e_max_sr) / se_sr
    dsr = float(_norm_cdf(z_stat))

    return dsr


# ── PSS（参数稳定性评分, v4 替代 PBO）─────────────────────────────────────────

def compute_pss(mc_details: list[dict]) -> dict:
    """计算 PSS (Parameter Stability Score) — 参数稳定性评分 (v4)。

    直接从 MC 已有数据计算，零额外回测开销。

    Args:
        mc_details: MC 运行结果列表，每项含 annual_return, max_drawdown, sharpe_ratio

    Returns:
        dict: {return_p10, return_p50, return_p90, dd_p10, dd_p50, dd_p90,
               sharpe_p10, sharpe_p50, sharpe_p90, return_cv, dd_cv, sharpe_cv, n_total}
    """
    if not mc_details:
        return {
            'return_p10': 0.0, 'return_p50': 0.0, 'return_p90': 0.0,
            'dd_p10': 0.0, 'dd_p50': 0.0, 'dd_p90': 0.0,
            'sharpe_p10': 0.0, 'sharpe_p50': 0.0, 'sharpe_p90': 0.0,
            'return_cv': 0.0, 'dd_cv': 0.0, 'sharpe_cv': 0.0,
            'n_total': 0,
        }

    returns = np.array([r['annual_return'] for r in mc_details])
    dds = np.array([r['max_drawdown'] for r in mc_details])
    sharpes = np.array([r['sharpe_ratio'] for r in mc_details])

    return {
        'return_p10': float(np.percentile(returns, 10)),
        'return_p50': float(np.median(returns)),
        'return_p90': float(np.percentile(returns, 90)),
        'dd_p10': float(np.percentile(dds, 10)),
        'dd_p50': float(np.median(dds)),
        'dd_p90': float(np.percentile(dds, 90)),
        'sharpe_p10': float(np.percentile(sharpes, 10)),
        'sharpe_p50': float(np.median(sharpes)),
        'sharpe_p90': float(np.percentile(sharpes, 90)),
        'return_cv': float(np.std(returns) / np.mean(returns)) if np.mean(returns) != 0 else 0.0,
        'dd_cv': float(np.std(dds) / np.mean(dds)) if np.mean(dds) != 0 else 0.0,
        'sharpe_cv': float(np.std(sharpes) / np.mean(sharpes)) if np.mean(sharpes) != 0 else 0.0,
        'n_total': len(mc_details),
    }


# ── SPS 起点敏感性 (v3) ───────────────────────────────────────────────────────

def _sps_single_worker(args: tuple) -> dict | None:
    """Single SPS run worker (module-level for multiprocessing)."""
    start_dt_str, end_dt_str, base_cfg = args
    try:
        r = run_backtest(
            base_cfg,
            start_date=start_dt_str,
            end_date=end_dt_str,
        )
        if r.nav_series.empty:
            return None
        return {
            'start_date': start_dt_str,
            'end_date': end_dt_str,
            'annual_return': r.metrics['annual_return'],
            'sharpe_ratio': r.metrics['sharpe_ratio'],
            'max_drawdown': r.metrics['max_drawdown'],
            'final_nav': float(r.nav_series['nav'].iloc[-1]),
        }
    except Exception:
        return None


def compute_starting_point_sensitivity(
    config_path: str,
    horizon_years: int = 3,
    step_months: int = 1,
    n_jobs: int = -1,
) -> tuple[dict, pd.DataFrame]:
    """滚动起点敏感度分析 (v3).

    逐月滚动起点，每个起点投资固定 horizon_years，运行完整回测。
    统计最差起点、收益分布、负收益比例等。

    Args:
        config_path: 策略配置 YAML 路径
        horizon_years: 投资期限（默认 3 年）
        step_months: 滚动步长（默认 1 月）
        n_jobs: 并行进程数（-1 = 全部 CPU）

    Returns:
        (metrics, details_df): 汇总指标和每个起点的详细结果 DataFrame
    """
    project_root = Path(__file__).resolve().parent.parent
    config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_cfg = load_config(config_abs)

    # 先跑一次完整回测获取总日期范围
    result = run_backtest(base_cfg)
    if result.nav_series.empty:
        raise RuntimeError(f"Baseline backtest failed for {config_path}")

    full_start = result.nav_series.index[0]
    full_end = result.nav_series.index[-1]

    # 生成逐月起点列表
    start_dates = pd.date_range(
        full_start,
        full_end - pd.DateOffset(years=horizon_years),
        freq=pd.DateOffset(months=step_months),
    )

    if len(start_dates) == 0:
        raise RuntimeError(
            f"数据不足: 全区间太短，无法生成起点 (full: {full_start.date()} ~ {full_end.date()})"
        )

    # 构建 worker 参数列表
    sps_args = []
    for start_dt in start_dates:
        end_dt = start_dt + pd.DateOffset(years=horizon_years)
        sps_args.append((str(start_dt.date()), str(end_dt.date()), base_cfg))

    n_proc = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)

    if n_proc > 1 and len(sps_args) > 1:
        with multiprocessing.Pool(min(n_proc, len(sps_args))) as pool:
            raw_results = pool.map(_sps_single_worker, sps_args)
    else:
        raw_results = [_sps_single_worker(args_tuple) for args_tuple in sps_args]

    results_list = [r for r in raw_results if r is not None]

    if len(results_list) == 0:
        raise RuntimeError("所有起点回测均失败")

    details_df = pd.DataFrame(results_list)
    # 按 start_date 排序
    details_df = details_df.sort_values('start_date').reset_index(drop=True)

    annual_rets = details_df['annual_return'].values
    nav_list = details_df['final_nav'].values

    metrics = {
        'worst_annual_return': float(np.min(annual_rets)),
        'worst_start_date': str(details_df.loc[int(np.argmin(annual_rets)), 'start_date']),
        'best_annual_return': float(np.max(annual_rets)),
        'best_start_date': str(details_df.loc[int(np.argmax(annual_rets)), 'start_date']),
        'mean_annual_return': float(np.mean(annual_rets)),
        'std_annual_return': float(np.std(annual_rets, ddof=1)),
        'median_annual_return': float(np.median(annual_rets)),
        'negative_return_ratio': float(np.mean(annual_rets < 0)),
        'p10_annual_return': float(np.percentile(annual_rets, 10)),
        'p90_annual_return': float(np.percentile(annual_rets, 90)),
        'negative_nav_ratio': float(np.mean(nav_list < 1.0)),
        'n_starting_points': len(results_list),
        'horizon_years': horizon_years,
        'step_months': step_months,
    }

    return metrics, details_df


# ── MC 生存率 (multiprocessing) ──────────────────────────────────────────────

# 扰动参数列表（v2 扩展为 7 个核心参数）
MC_PARAMS = [
    'mom_w',                # 动量权重 (0.35)
    'vol_w',                # 波动率权重 (0.30)
    'def_alloc',            # 基准防御比例 (0.25)
    'step_high',            # vol三段式上限 (0.35)
    'step_low',             # vol三段式下限 (0.20)
    'momentum_window',      # D4动量窗口 (8, 整数)
    'momentum_threshold',   # D4动量阈值 (-0.075)
]
PERTURBATION = 0.15  # ±15%

# ── Phase 6: Full-Parameter Grid Constants ──────────────────────────────────

# 连续型参数（8 个，7 级比例扰动）
GRID_CONTINUOUS_PARAMS = [
    'mom_w', 'vol_w', 'def_alloc', 'step_low', 'step_high',
    'max_def', 'rebalance_threshold', 'stop_loss',
]

# 离散型参数（绝对级别，非百分比）
GRID_DISCRETE_PARAMS = [
    'top_n',  # 级别: 1, 2, 3
]

# D4 专属参数（仅在 D4 启用时测试，Phase A-1 hard-clamp）
GRID_D4_PARAMS = [
    'momentum_window',     # 整数，round 后 clamp [1, 8]
    'momentum_threshold',  # 连续，clamp [-0.07, 0.05]
]

# 所有网格参数
GRID_ALL_PARAMS = GRID_CONTINUOUS_PARAMS + GRID_DISCRETE_PARAMS + GRID_D4_PARAMS

# clamp 范围
GRID_CLAMP = {
    'mom_w': (0.05, 1.50),
    'vol_w': (0.05, 1.50),
    'def_alloc': (0.05, 0.60),
    'step_low': (0.05, 0.60),
    'step_high': (0.05, 0.60),
    'max_def': (0.70, 0.99),
    'rebalance_threshold': (0.03, 0.15),
    'stop_loss': (0.03, 0.20),
    'momentum_threshold': (-0.07, 0.05),
}

PERTURBATION_LEVELS = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
TOP_N_LEVELS = [1, 2, 3]


def _mc_single_worker(args: tuple) -> dict | None:
    """Single MC run worker (module-level for multiprocessing)."""
    params, base_cfg = args
    try:
        cfg = StrategyConfig(
            name=base_cfg.name,
            version=base_cfg.version,
            mom_w=min(max(params.get('mom_w', base_cfg.mom_w), 0.05), 1.50),
            vol_w=min(max(params.get('vol_w', base_cfg.vol_w), 0.05), 1.50),
            top_n=base_cfg.top_n,
            mom_window=base_cfg.mom_window,
            vol_window=base_cfg.vol_window,
            pe_window_years=base_cfg.pe_window_years,
            def_alloc=params.get('def_alloc', base_cfg.def_alloc),
            step_low=params.get('step_low', base_cfg.step_low),
            step_high=params.get('step_high', base_cfg.step_high),
            max_def=base_cfg.max_def,
            hongli_ratio=base_cfg.hongli_ratio,
            rebalance_threshold=base_cfg.rebalance_threshold,
            fee_rate=base_cfg.fee_rate,
            anchor=base_cfg.anchor,
            stop_loss=base_cfg.stop_loss,
            recovery_weeks=base_cfg.recovery_weeks,
            tiered_stop_loss=base_cfg.tiered_stop_loss,
            l1_drawdown=base_cfg.l1_drawdown,
            l1_defense=base_cfg.l1_defense,
            l2_drawdown=base_cfg.l2_drawdown,
            l2_defense=base_cfg.l2_defense,
            l3_weekly_drop=base_cfg.l3_weekly_drop,
            l3_down_weeks=base_cfg.l3_down_weeks,
            l3_window=base_cfg.l3_window,
            l2_recovery_weeks=base_cfg.l2_recovery_weeks,
            l3_recovery_weeks=base_cfg.l3_recovery_weeks,
            # Phase A-2: position-based tiered stop loss
            ptiered_stop_loss=base_cfg.ptiered_stop_loss,
            p_recovery_weeks=base_cfg.p_recovery_weeks,
            p_l1_dd_low=base_cfg.p_l1_dd_low,
            p_l1_dd_high=base_cfg.p_l1_dd_high,
            p_l1_position=base_cfg.p_l1_position,
            p_l2_dd_low=base_cfg.p_l2_dd_low,
            p_l2_dd_high=base_cfg.p_l2_dd_high,
            p_l2_position=base_cfg.p_l2_position,
            p_l3_dd_threshold=base_cfg.p_l3_dd_threshold,
            p_l3_position=base_cfg.p_l3_position,
            max_single_alloc=base_cfg.max_single_alloc,
            stateful_stop_loss=base_cfg.stateful_stop_loss,
            score_margin=base_cfg.score_margin,
            d4_enabled=base_cfg.d4_enabled,
            # Phase A-1: hard-clamp D4 params
            d4_momentum_window=min(params.get('momentum_window', base_cfg.d4_momentum_window), 8),
            d4_momentum_threshold=min(max(params.get('momentum_threshold', base_cfg.d4_momentum_threshold), -0.07), 0.05),
            d4_action=base_cfg.d4_action,
            d4_min_candidates=base_cfg.d4_min_candidates,
            d1_enabled=base_cfg.d1_enabled,
            d1_lookback=base_cfg.d1_lookback,
            d1_tq_low=base_cfg.d1_tq_low,
            d1_tq_high=base_cfg.d1_tq_high,
            d1_mom_w_low=base_cfg.d1_mom_w_low,
            d1_mom_w_high=base_cfg.d1_mom_w_high,
            d1_vol_w_low=base_cfg.d1_vol_w_low,
            d1_vol_w_high=base_cfg.d1_vol_w_high,
            d1_weight_sum=base_cfg.d1_weight_sum,
            nav_path=base_cfg.nav_path,
            pe_path=base_cfg.pe_path,
            start_date=base_cfg.start_date,
            end_date=base_cfg.end_date,
            risk_free_rate=base_cfg.risk_free_rate,
            # P1 Fix #2: 权重上限修复
            overflow_to_defense_only=base_cfg.overflow_to_defense_only,
            dynamic_weight_cap=base_cfg.dynamic_weight_cap,
            dc_bull_cap=base_cfg.dc_bull_cap,
            dc_normal_cap=base_cfg.dc_normal_cap,
            dc_correction_cap=base_cfg.dc_correction_cap,
            dc_crisis_cap=base_cfg.dc_crisis_cap,
            # Market state fields
            ms_bull_mom=base_cfg.ms_bull_mom,
            ms_correction_mom=base_cfg.ms_correction_mom,
            ms_crisis_mom=base_cfg.ms_crisis_mom,
            ms_low_vol_pct=base_cfg.ms_low_vol_pct,
            ms_mid_vol_pct=base_cfg.ms_mid_vol_pct,
            ms_high_vol_pct=base_cfg.ms_high_vol_pct,
            ms_shallow_dd=base_cfg.ms_shallow_dd,
            ms_moderate_dd=base_cfg.ms_moderate_dd,
            ms_deep_dd=base_cfg.ms_deep_dd,
            ss_bull_l1=base_cfg.ss_bull_l1,
            ss_bull_l1_def=base_cfg.ss_bull_l1_def,
            ss_bull_l2=base_cfg.ss_bull_l2,
            ss_bull_l2_def=base_cfg.ss_bull_l2_def,
            ss_bull_recovery=base_cfg.ss_bull_recovery,
            ss_normal_l1=base_cfg.ss_normal_l1,
            ss_normal_l1_def=base_cfg.ss_normal_l1_def,
            ss_normal_l2=base_cfg.ss_normal_l2,
            ss_normal_l2_def=base_cfg.ss_normal_l2_def,
            ss_normal_recovery=base_cfg.ss_normal_recovery,
            ss_correction_l1=base_cfg.ss_correction_l1,
            ss_correction_l1_def=base_cfg.ss_correction_l1_def,
            ss_correction_l2=base_cfg.ss_correction_l2,
            ss_correction_l2_def=base_cfg.ss_correction_l2_def,
            ss_correction_recovery=base_cfg.ss_correction_recovery,
            ss_crisis_l1=base_cfg.ss_crisis_l1,
            ss_crisis_l1_def=base_cfg.ss_crisis_l1_def,
            ss_crisis_l2=base_cfg.ss_crisis_l2,
            ss_crisis_l2_def=base_cfg.ss_crisis_l2_def,
            ss_crisis_recovery=base_cfg.ss_crisis_recovery,
            # D5: Softmax-Weighted Allocation
            softmax_enabled=base_cfg.softmax_enabled,
            softmax_temperature=base_cfg.softmax_temperature,
            softmax_min_candidates=base_cfg.softmax_min_candidates,
            softmax_hard_top_n_fallback=base_cfg.softmax_hard_top_n_fallback,
            # D6: Inv-Vol8 Weighted Allocation
            inv_vol_enabled=base_cfg.inv_vol_enabled,
            inv_vol_window=base_cfg.inv_vol_window,
            # Regime-conditional softmax
            softmax_regime_enabled=base_cfg.softmax_regime_enabled,
            softmax_regime_temperature=base_cfg.softmax_regime_temperature,
            # Regime classifier
            regime_enabled=base_cfg.regime_enabled,
            regime_data_path=base_cfg.regime_data_path,
            regime_overrides=base_cfg.regime_overrides,
            regime_3state=base_cfg.regime_3state,
            # Constituent signals
            constituent_signals_enabled=base_cfg.constituent_signals_enabled,
            constituent_signals_path=base_cfg.constituent_signals_path,
            cwm_weight=base_cfg.cwm_weight,
            conc_weight=base_cfg.conc_weight,
            cwm_window=base_cfg.cwm_window,
        )
        result = run_backtest(cfg)
        if result.nav_series.empty:
            return None
        return {
            **params,
            'sharpe_ratio': result.metrics['sharpe_ratio'],
            'annual_return': result.metrics['annual_return'],
            'max_drawdown': result.metrics['max_drawdown'],
        }
    except Exception:
        return None


def run_mc_survival_test(
    config_path: str,
    n_runs: int = 400,
    perturbation: float = 0.15,
    n_jobs: int = -1,
    mode: str = 'mc',
) -> tuple[float, list[dict]]:
    """Monte Carlo 参数扰动生存率测试 (v2)。

    对 7 个核心参数同时随机扰动，运行 N 次回测。
    支持两种模式：
      - 'mc': 全参数同时随机扰动 (默认)
      - 'oat': One-At-a-Time 多级敏感度分析

    Args:
        config_path: 策略配置 YAML 路径
        n_runs: Monte Carlo 运行次数 (默认 400)
        perturbation: 扰动比例（默认 0.15 = ±15%）
        n_jobs: 并行进程数（-1 = 全部 CPU）
        mode: 'mc' 或 'oat'

    Returns:
        (survival_rate, mc_details): 生存率 (年化>10% AND DD<15%) 和每次运行的详细结果
    """
    project_root = Path(__file__).resolve().parent.parent
    config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_cfg = load_config(config_abs)

    # 生成随机参数扰动组合
    rng = np.random.RandomState(42)

    baseline_params = {
        'mom_w': base_cfg.mom_w,
        'vol_w': base_cfg.vol_w,
        'def_alloc': base_cfg.def_alloc,
        'step_high': base_cfg.step_high,
        'step_low': base_cfg.step_low,
        'momentum_window': base_cfg.d4_momentum_window,
        'momentum_threshold': base_cfg.d4_momentum_threshold,
    }

    # OAT 模式：分发到独立的多级敏感度函数
    if mode == 'oat':
        oat_result = run_oat_sensitivity(str(config_abs), perturbation, n_jobs)
        # 返回 (0.0, []) 作为 dummy — OAT 结果通过 evaluate_robustness 取 oat_sensitivity
        return 0.0, [oat_result]  # 通过 details 传递 OAT 结果

    mc_args = []
    for _ in range(n_runs):
        params = {}
        for key, base_val in baseline_params.items():
            if key == 'momentum_window':
                # Phase A-1 clamp: max 8 (was [7,9])
                delta = base_val * perturbation
                noise = rng.uniform(-delta, delta)
                new_val = round(base_val + noise)
                new_val = max(1, min(8, new_val))
            else:
                delta = base_val * perturbation
                noise = rng.uniform(-delta, delta)
                new_val = base_val + noise
                # Clamp to reasonable bounds
                if key in ('mom_w', 'vol_w'):
                    new_val = max(0.05, min(1.50, new_val))
                elif key == 'def_alloc':
                    new_val = max(0.05, min(0.60, new_val))
                elif key in ('step_low', 'step_high'):
                    new_val = max(0.05, min(0.60, new_val))
                elif key == 'momentum_threshold':
                    # Phase A-1 clamp: min -0.07 (was [-0.20, 0.05])
                    new_val = max(-0.07, min(0.05, new_val))
            params[key] = new_val
        mc_args.append((params, base_cfg))

    # 并行运行
    n_proc = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    with multiprocessing.Pool(n_proc) as pool:
        results = pool.map(_mc_single_worker, mc_args)

    mc_details = [r for r in results if r is not None]

    if len(mc_details) == 0:
        return 0.0, []

    # 生存率 = 年化 > 10% AND 最大回撤 < 15%（v2 收紧标准）
    n_survived = sum(1 for r in mc_details
                     if r['annual_return'] > 0.10 and r['max_drawdown'] < 0.15)
    survival_rate = n_survived / len(mc_details)

    return survival_rate, mc_details


# ── OAT 多级敏感度 (v2 新增) ──────────────────────────────────────────────────

def _oat_single_worker(args: tuple) -> dict | None:
    """OAT single run worker — wraps _mc_single_worker with param_name/level tags."""
    params, base_cfg, param_name, level = args
    result = _mc_single_worker((params, base_cfg))
    if result is not None:
        result['_param_name'] = param_name
        result['level'] = level
    return result


def run_oat_sensitivity(
    config_path: str,
    perturbation: float = 0.15,
    n_jobs: int = -1,
    perturbation_levels: list[float] | None = None,
) -> dict[str, list[dict]]:
    """OAT (One-At-a-Time) 多级敏感度分析 (v2)。

    对每个参数，在 7 个扰动级别下单独测试（-15%, -10%, -5%, 0%, +5%, +10%, +15%），
    其他参数保持基线值不变。共 7×7 = 49 次回测。

    Args:
        config_path: 策略配置 YAML 路径
        perturbation: 扰动幅度参考值（默认 0.15）
        n_jobs: 并行进程数（-1 = 全部 CPU）
        perturbation_levels: 扰动级别列表，默认 [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

    Returns:
        {param_name: [{level, sharpe, ret, dd}, ...], ...}
        每个 param_name 对应 7 个 level 的结果（按 level 升序排列）
    """
    if perturbation_levels is None:
        perturbation_levels = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

    project_root = Path(__file__).resolve().parent.parent
    config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_cfg = load_config(config_abs)

    baseline_params = {
        'mom_w': base_cfg.mom_w,
        'vol_w': base_cfg.vol_w,
        'def_alloc': base_cfg.def_alloc,
        'step_high': base_cfg.step_high,
        'step_low': base_cfg.step_low,
        'momentum_window': base_cfg.d4_momentum_window,
        'momentum_threshold': base_cfg.d4_momentum_threshold,
    }

    # 构建所有 OAT 参数组合：7 参数 × 7 级别 = 49
    oat_args = []
    for param_name, base_val in baseline_params.items():
        for level in perturbation_levels:
            params = dict(baseline_params)  # 复制基线
            if param_name == 'momentum_window':
                new_val = round(base_val * (1 + level))
                new_val = max(1, min(8, new_val))  # Phase A-1 clamp
            else:
                new_val = base_val * (1 + level)
                if param_name in ('mom_w', 'vol_w'):
                    new_val = max(0.05, min(1.50, new_val))
                elif param_name == 'def_alloc':
                    new_val = max(0.05, min(0.60, new_val))
                elif param_name in ('step_low', 'step_high'):
                    new_val = max(0.05, min(0.60, new_val))
                elif param_name == 'momentum_threshold':
                    new_val = max(-0.07, min(0.05, new_val))
            params[param_name] = new_val
            oat_args.append((params, base_cfg, param_name, level))

    # 并行运行
    n_proc = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    with multiprocessing.Pool(n_proc) as pool:
        results = pool.map(_oat_single_worker, oat_args)

    # 按参数整理结果
    sensitivity: dict[str, list[dict]] = {}
    for r in results:
        if r is None:
            continue
        pname = r.pop('_param_name')
        if pname not in sensitivity:
            sensitivity[pname] = []
        sensitivity[pname].append({
            'level': r['level'],
            'sharpe': r['sharpe_ratio'],
            'ret': r['annual_return'],
            'dd': r['max_drawdown'],
        })

    # 按 level 升序排列
    for pname in sensitivity:
        sensitivity[pname].sort(key=lambda x: x['level'])

    return sensitivity


# ── Phase 6: Full-Parameter Grid ─────────────────────────────────────────────

def _apply_grid_overrides(
    base_cfg: StrategyConfig,
    overrides: dict,
) -> StrategyConfig:
    """Apply parameter overrides to create a modified StrategyConfig.

    Uses base_cfg defaults for all fields, overriding only the keys in overrides.
    """
    return StrategyConfig(
        name=base_cfg.name,
        version=base_cfg.version,
        mom_w=overrides.get('mom_w', base_cfg.mom_w),
        vol_w=overrides.get('vol_w', base_cfg.vol_w),
        top_n=overrides.get('top_n', base_cfg.top_n),
        score_margin=base_cfg.score_margin,
        mom_window=base_cfg.mom_window,
        vol_window=base_cfg.vol_window,
        pe_window_years=base_cfg.pe_window_years,
        def_alloc=overrides.get('def_alloc', base_cfg.def_alloc),
        step_low=overrides.get('step_low', base_cfg.step_low),
        step_high=overrides.get('step_high', base_cfg.step_high),
        max_def=overrides.get('max_def', base_cfg.max_def),
        hongli_ratio=base_cfg.hongli_ratio,
        rebalance_threshold=overrides.get('rebalance_threshold', base_cfg.rebalance_threshold),
        fee_rate=base_cfg.fee_rate,
        anchor=base_cfg.anchor,
        stop_loss=overrides.get('stop_loss', base_cfg.stop_loss),
        recovery_weeks=base_cfg.recovery_weeks,
        tiered_stop_loss=base_cfg.tiered_stop_loss,
        l1_drawdown=base_cfg.l1_drawdown,
        l1_defense=base_cfg.l1_defense,
        l2_drawdown=base_cfg.l2_drawdown,
        l2_defense=base_cfg.l2_defense,
        l3_weekly_drop=base_cfg.l3_weekly_drop,
        l3_down_weeks=base_cfg.l3_down_weeks,
        l3_window=base_cfg.l3_window,
        l2_recovery_weeks=base_cfg.l2_recovery_weeks,
        l3_recovery_weeks=base_cfg.l3_recovery_weeks,
        # Phase A-2: position-based tiered stop loss
        ptiered_stop_loss=base_cfg.ptiered_stop_loss,
        p_recovery_weeks=base_cfg.p_recovery_weeks,
        p_l1_dd_low=base_cfg.p_l1_dd_low,
        p_l1_dd_high=base_cfg.p_l1_dd_high,
        p_l1_position=base_cfg.p_l1_position,
        p_l2_dd_low=base_cfg.p_l2_dd_low,
        p_l2_dd_high=base_cfg.p_l2_dd_high,
        p_l2_position=base_cfg.p_l2_position,
        p_l3_dd_threshold=base_cfg.p_l3_dd_threshold,
        p_l3_position=base_cfg.p_l3_position,
        max_single_alloc=base_cfg.max_single_alloc,
        stateful_stop_loss=base_cfg.stateful_stop_loss,
        d4_enabled=base_cfg.d4_enabled,
        d4_momentum_window=overrides.get('momentum_window', base_cfg.d4_momentum_window),
        d4_momentum_threshold=overrides.get('momentum_threshold', base_cfg.d4_momentum_threshold),
        d4_action=base_cfg.d4_action,
        d4_min_candidates=base_cfg.d4_min_candidates,
        d1_enabled=base_cfg.d1_enabled,
        d1_lookback=base_cfg.d1_lookback,
        d1_tq_low=base_cfg.d1_tq_low,
        d1_tq_high=base_cfg.d1_tq_high,
        d1_mom_w_low=base_cfg.d1_mom_w_low,
        d1_mom_w_high=base_cfg.d1_mom_w_high,
        d1_vol_w_low=base_cfg.d1_vol_w_low,
        d1_vol_w_high=base_cfg.d1_vol_w_high,
        d1_weight_sum=base_cfg.d1_weight_sum,
        nav_path=base_cfg.nav_path,
        pe_path=base_cfg.pe_path,
        start_date=base_cfg.start_date,
        end_date=base_cfg.end_date,
        risk_free_rate=base_cfg.risk_free_rate,
        overflow_to_defense_only=base_cfg.overflow_to_defense_only,
        dynamic_weight_cap=base_cfg.dynamic_weight_cap,
        dc_bull_cap=base_cfg.dc_bull_cap,
        dc_normal_cap=base_cfg.dc_normal_cap,
        dc_correction_cap=base_cfg.dc_correction_cap,
        dc_crisis_cap=base_cfg.dc_crisis_cap,
        ms_bull_mom=base_cfg.ms_bull_mom,
        ms_correction_mom=base_cfg.ms_correction_mom,
        ms_crisis_mom=base_cfg.ms_crisis_mom,
        ms_low_vol_pct=base_cfg.ms_low_vol_pct,
        ms_mid_vol_pct=base_cfg.ms_mid_vol_pct,
        ms_high_vol_pct=base_cfg.ms_high_vol_pct,
        ms_shallow_dd=base_cfg.ms_shallow_dd,
        ms_moderate_dd=base_cfg.ms_moderate_dd,
        ms_deep_dd=base_cfg.ms_deep_dd,
        ss_bull_l1=base_cfg.ss_bull_l1,
        ss_bull_l1_def=base_cfg.ss_bull_l1_def,
        ss_bull_l2=base_cfg.ss_bull_l2,
        ss_bull_l2_def=base_cfg.ss_bull_l2_def,
        ss_bull_recovery=base_cfg.ss_bull_recovery,
        ss_normal_l1=base_cfg.ss_normal_l1,
        ss_normal_l1_def=base_cfg.ss_normal_l1_def,
        ss_normal_l2=base_cfg.ss_normal_l2,
        ss_normal_l2_def=base_cfg.ss_normal_l2_def,
        ss_normal_recovery=base_cfg.ss_normal_recovery,
        ss_correction_l1=base_cfg.ss_correction_l1,
        ss_correction_l1_def=base_cfg.ss_correction_l1_def,
        ss_correction_l2=base_cfg.ss_correction_l2,
        ss_correction_l2_def=base_cfg.ss_correction_l2_def,
        ss_correction_recovery=base_cfg.ss_correction_recovery,
        ss_crisis_l1=base_cfg.ss_crisis_l1,
        ss_crisis_l1_def=base_cfg.ss_crisis_l1_def,
        ss_crisis_l2=base_cfg.ss_crisis_l2,
        ss_crisis_l2_def=base_cfg.ss_crisis_l2_def,
        ss_crisis_recovery=base_cfg.ss_crisis_recovery,
        softmax_enabled=base_cfg.softmax_enabled,
        softmax_temperature=base_cfg.softmax_temperature,
        softmax_min_candidates=base_cfg.softmax_min_candidates,
        softmax_hard_top_n_fallback=base_cfg.softmax_hard_top_n_fallback,
        softmax_regime_enabled=base_cfg.softmax_regime_enabled,
        softmax_regime_temperature=base_cfg.softmax_regime_temperature,
        regime_enabled=base_cfg.regime_enabled,
        regime_data_path=base_cfg.regime_data_path,
        regime_overrides=base_cfg.regime_overrides,
        regime_3state=base_cfg.regime_3state,
        constituent_signals_enabled=base_cfg.constituent_signals_enabled,
        constituent_signals_path=base_cfg.constituent_signals_path,
        cwm_weight=base_cfg.cwm_weight,
        conc_weight=base_cfg.conc_weight,
        cwm_window=base_cfg.cwm_window,
        # D6: Inv-Vol8 Weighted Allocation
        inv_vol_enabled=base_cfg.inv_vol_enabled,
        inv_vol_window=base_cfg.inv_vol_window,
    )


def build_full_grid(config_path: str) -> list[GridPointConfig]:
    """Build the full-parameter 7-level grid point configuration list.

    Auto-detects D4 enabled/disabled and includes/excludes D4-specific params.
    """
    project_root = Path(__file__).resolve().parent.parent
    config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_cfg = load_config(config_abs)

    baseline_values: dict[str, float] = {
        'mom_w': base_cfg.mom_w,
        'vol_w': base_cfg.vol_w,
        'def_alloc': base_cfg.def_alloc,
        'step_low': base_cfg.step_low,
        'step_high': base_cfg.step_high,
        'max_def': base_cfg.max_def,
        'rebalance_threshold': base_cfg.rebalance_threshold,
        'stop_loss': base_cfg.stop_loss,
        'top_n': float(base_cfg.top_n),
        'momentum_window': float(base_cfg.d4_momentum_window),
        'momentum_threshold': base_cfg.d4_momentum_threshold,
    }

    grid: list[GridPointConfig] = []

    # Continuous params (8 x 7 = 56)
    for param in GRID_CONTINUOUS_PARAMS:
        base_val = baseline_values[param]
        lo, hi = GRID_CLAMP[param]
        for level in PERTURBATION_LEVELS:
            new_val = base_val * (1.0 + level)
            new_val = max(lo, min(hi, new_val))
            grid.append(GridPointConfig(
                param_name=param,
                level=level,
                actual_value=new_val,
                param_overrides={param: new_val},
            ))

    # Discrete param: top_n (3 levels)
    for val in TOP_N_LEVELS:
        grid.append(GridPointConfig(
            param_name='top_n',
            level=float(val),
            actual_value=float(val),
            param_overrides={'top_n': val},
        ))

    # D4 params (only when D4 enabled)
    if base_cfg.d4_enabled:
        base_mom_window = baseline_values['momentum_window']
        for level in PERTURBATION_LEVELS:
            new_val = round(base_mom_window * (1.0 + level))
            new_val = max(1, min(8, new_val))  # Phase A-1 clamp
            grid.append(GridPointConfig(
                param_name='momentum_window',
                level=level,
                actual_value=float(new_val),
                param_overrides={'momentum_window': int(new_val)},
            ))

        base_mom_thresh = baseline_values['momentum_threshold']
        lo_thresh, hi_thresh = GRID_CLAMP['momentum_threshold']
        for level in PERTURBATION_LEVELS:
            new_val = base_mom_thresh * (1.0 + level)
            new_val = max(lo_thresh, min(hi_thresh, new_val))
            grid.append(GridPointConfig(
                param_name='momentum_threshold',
                level=level,
                actual_value=new_val,
                param_overrides={'momentum_threshold': new_val},
            ))

    return grid


def _compute_grid_point_metrics(
    cfg: StrategyConfig,
    risk_free: float = 0.025,
) -> dict:
    """Run a single backtest and compute Sharpe / AnnRet / MaxDD / RelativeSharpe."""
    result = run_backtest(cfg)
    if result.nav_series.empty:
        return {
            'sharpe': float('nan'),
            'annual_return': float('nan'),
            'max_drawdown': float('nan'),
            'relative_sharpe': float('nan'),
        }

    metrics = result.metrics
    strat_sharpe = metrics['sharpe_ratio']
    ann_ret = metrics['annual_return']
    max_dd = metrics['max_drawdown']

    # Compute equal-weight benchmark Sharpe for the full period
    project_root = Path(__file__).resolve().parent.parent
    nav_df = load_nav_data(project_root / cfg.nav_path)
    weekly_nav = resample_weekly(nav_df, anchor=cfg.anchor)
    full_start = result.nav_series.index[0]
    full_end = result.nav_series.index[-1]
    bench_sharpe = _compute_equal_weight_benchmark_sharpe(
        weekly_nav, full_start, full_end, risk_free
    )

    rel_sharpe = strat_sharpe - bench_sharpe if not np.isnan(bench_sharpe) else float('nan')

    return {
        'sharpe': strat_sharpe,
        'annual_return': ann_ret,
        'max_drawdown': max_dd,
        'relative_sharpe': rel_sharpe,
    }


def _local_mc_single_worker(args: tuple) -> dict | None:
    """Single local-MC run worker (module-level for multiprocessing)."""
    overrides, config_path = args
    try:
        project_root = Path(__file__).resolve().parent.parent
        config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
        base_cfg = load_config(config_abs)
        cfg = _apply_grid_overrides(base_cfg, overrides)
        result = run_backtest(cfg)
        if result.nav_series.empty:
            return None
        return {
            'sharpe_ratio': result.metrics['sharpe_ratio'],
            'annual_return': result.metrics['annual_return'],
            'max_drawdown': result.metrics['max_drawdown'],
            **overrides,
        }
    except Exception:
        return None


def _run_local_mc(
    base_cfg: StrategyConfig,
    fixed_param: str,
    fixed_value: float,
    n_runs: int,
    config_path: str,
) -> tuple[float, list[dict]]:
    """Run local MC around a grid point.

    Fixes *fixed_param* at *fixed_value*, then perturbs all other active
    parameters +/-15% for n_runs iterations.
    """
    # Determine which parameters to perturb (all active params except fixed)
    d4_enabled = base_cfg.d4_enabled
    if d4_enabled:
        active_params = list(GRID_ALL_PARAMS)
    else:
        active_params = list(GRID_CONTINUOUS_PARAMS) + list(GRID_DISCRETE_PARAMS)

    perturb_params = [p for p in active_params if p != fixed_param]

    # Build baseline values
    baseline_values: dict[str, float] = {
        'mom_w': base_cfg.mom_w,
        'vol_w': base_cfg.vol_w,
        'def_alloc': base_cfg.def_alloc,
        'step_low': base_cfg.step_low,
        'step_high': base_cfg.step_high,
        'max_def': base_cfg.max_def,
        'rebalance_threshold': base_cfg.rebalance_threshold,
        'stop_loss': base_cfg.stop_loss,
        'top_n': float(base_cfg.top_n),
        'momentum_window': float(base_cfg.d4_momentum_window),
        'momentum_threshold': base_cfg.d4_momentum_threshold,
    }

    rng = np.random.RandomState(42)

    mc_args = []
    for _ in range(n_runs):
        overrides: dict = {}
        # Fix the grid point param
        if fixed_param == 'top_n':
            overrides[fixed_param] = int(fixed_value)
        elif fixed_param == 'momentum_window':
            overrides[fixed_param] = int(fixed_value)
        else:
            overrides[fixed_param] = fixed_value

        # Perturb all other active params
        for p in perturb_params:
            base_val = baseline_values[p]
            if p in GRID_CLAMP:
                lo, hi = GRID_CLAMP[p]
            else:
                lo, hi = -float('inf'), float('inf')

            if p == 'top_n':
                overrides[p] = int(rng.choice(TOP_N_LEVELS))
            elif p == 'momentum_window':
                delta = base_val * PERTURBATION
                noise = rng.uniform(-delta, delta)
                new_val = round(base_val + noise)
                new_val = max(1, min(8, new_val))  # Phase A-1 clamp
                overrides[p] = int(new_val)
            else:
                delta = base_val * PERTURBATION
                noise = rng.uniform(-delta, delta)
                new_val = base_val + noise
                new_val = max(lo, min(hi, new_val))
                overrides[p] = new_val

        mc_args.append((overrides, config_path))

    # Run in parallel
    n_proc = min(multiprocessing.cpu_count(), n_runs)
    if n_proc > 1 and len(mc_args) > 1:
        with multiprocessing.Pool(n_proc) as pool:
            raw_results = pool.map(_local_mc_single_worker, mc_args)
    else:
        raw_results = [_local_mc_single_worker(a) for a in mc_args]

    mc_details = [r for r in raw_results if r is not None]

    if len(mc_details) == 0:
        return 0.0, []

    # Survival = annual_return > 10% AND max_drawdown < 15%
    n_survived = sum(
        1 for r in mc_details
        if r['annual_return'] > 0.10 and r['max_drawdown'] < 0.15
    )
    survival_rate = n_survived / len(mc_details)

    return survival_rate, mc_details


def _grid_point_worker(args: tuple) -> tuple[str, dict] | None:
    """Single grid-point worker (module-level for multiprocessing).

    Runs the baseline backtest + metrics + local MC for one grid point.
    """
    gp_dict, config_path, n_local_mc = args
    try:
        project_root = Path(__file__).resolve().parent.parent
        config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
        base_cfg = load_config(config_abs)

        param_name = gp_dict['param_name']
        level = gp_dict['level']
        actual_value = gp_dict['actual_value']
        overrides = gp_dict['param_overrides']

        # Build modified config
        cfg = _apply_grid_overrides(base_cfg, overrides)

        # Compute metrics (Sharpe, AnnRet, MaxDD, RelativeSharpe)
        metrics = _compute_grid_point_metrics(cfg, base_cfg.risk_free_rate)
        if np.isnan(metrics['sharpe']):
            return None

        # Local MC
        mc_rate, mc_details = _run_local_mc(
            base_cfg, param_name, actual_value, n_local_mc, config_path
        )

        return (param_name, {
            'param_name': param_name,
            'level': level,
            'actual_value': actual_value,
            'sharpe': metrics['sharpe'],
            'annual_return': metrics['annual_return'],
            'max_drawdown': metrics['max_drawdown'],
            'relative_sharpe': metrics['relative_sharpe'],
            'mc_survival_rate': mc_rate,
            'mc_details': mc_details,
        })
    except Exception:
        return None


def run_full_grid(
    config_path: str,
    n_local_mc: int = 50,
    n_jobs: int = -1,
) -> dict[str, list[GridPointResult]]:
    """Run the full-parameter 7-level grid robustness evaluation (Phase 6).

    For each grid point:
      1. Run baseline backtest -> Sharpe, AnnRet, MaxDD
      2. Compute relative Sharpe vs equal-weight benchmark
      3. Run local MC (n_local_mc iterations) -> MC survival rate
    """
    grid_configs = build_full_grid(config_path)

    # Build worker args
    worker_args = []
    for gc in grid_configs:
        gp_dict = {
            'param_name': gc.param_name,
            'level': gc.level,
            'actual_value': gc.actual_value,
            'param_overrides': gc.param_overrides,
        }
        worker_args.append((gp_dict, config_path, n_local_mc))

    # Run in parallel
    n_proc = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    if n_proc > 1 and len(worker_args) > 1:
        with multiprocessing.Pool(min(n_proc, len(worker_args))) as pool:
            raw_results = pool.map(_grid_point_worker, worker_args)
    else:
        raw_results = [_grid_point_worker(a) for a in worker_args]

    # Organize results by param_name
    grid_results: dict[str, list[GridPointResult]] = {}
    for r in raw_results:
        if r is None:
            continue
        pname, gp_dict = r
        if pname not in grid_results:
            grid_results[pname] = []
        gpr = GridPointResult(
            param_name=gp_dict['param_name'],
            level=gp_dict['level'],
            actual_value=gp_dict['actual_value'],
            sharpe=gp_dict['sharpe'],
            annual_return=gp_dict['annual_return'],
            max_drawdown=gp_dict['max_drawdown'],
            relative_sharpe=gp_dict['relative_sharpe'],
            mc_survival_rate=gp_dict['mc_survival_rate'],
            mc_details=gp_dict['mc_details'],
        )
        grid_results[pname].append(gpr)

    # Sort each param's results by level
    for pname in grid_results:
        grid_results[pname].sort(key=lambda x: x.level)

    return grid_results


# ── 基准相对胜率 (Walk-Forward) ──────────────────────────────────────────────

def _compute_equal_weight_benchmark_sharpe(
    weekly_nav: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    risk_free: float = 0.025,
) -> float:
    """计算等权 5 ETF 基准的 Sharpe。

    基准：5 只 ETF（纳指、红利低波、中证500、黄金、国债）各 20%，每周再平衡。
    """
    mask = (weekly_nav.index >= start_dt) & (weekly_nav.index < end_dt)
    window_nav = weekly_nav.loc[mask]

    if len(window_nav) < 5:
        return float('nan')

    # 周收益
    weekly_rets = window_nav.pct_change().dropna()

    # 等权组合：每只 ETF 20%
    if len(weekly_rets.columns) >= 5:
        etf_cols = [c for c in ETFS if c in weekly_rets.columns]
        if len(etf_cols) >= 5:
            eq_returns = weekly_rets[etf_cols].mean(axis=1)
            return compute_sharpe(eq_returns, risk_free)

    # Fallback: all available columns
    eq_returns = weekly_rets.mean(axis=1)
    return compute_sharpe(eq_returns, risk_free)


def compute_benchmark_relative_win_rate(
    config_path: str,
    n_windows: int = 9,
) -> tuple[float, list[dict]]:
    """Walk-Forward 基准相对胜率。

    滚动窗口（1 年窗口），每窗口计算：
      策略 Sharpe - 等权 ETF 基准 Sharpe

    等权基准：5 ETF 各 20%，周频再平衡。

    Args:
        config_path: 策略配置 YAML 路径
        n_windows: 滚动窗口数

    Returns:
        (win_rate, wf_details): 策略 Sharpe > 基准 Sharpe 的窗口比例和每窗口详细结果
    """
    project_root = Path(__file__).resolve().parent.parent
    config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_cfg = load_config(config_abs)

    # 加载基准回测数据获取 NAV
    result = run_backtest(base_cfg)
    if result.nav_series.empty:
        return 0.0, []

    # 也加载原始 NAV 数据用于等权基准计算
    nav_df = load_nav_data(project_root / base_cfg.nav_path)
    weekly_nav = resample_weekly(nav_df, anchor=base_cfg.anchor)

    # 确定日期范围
    nav_series = result.nav_series
    full_start = nav_series.index[0]
    full_end = nav_series.index[-1]

    # 生成 N 个 1 年窗口，均匀分布
    total_days = (full_end - full_start).days
    window_days = 365
    if total_days <= window_days:
        # 数据不足 1 年，单窗口
        step_days = 0
        n_windows = 1
    else:
        step_days = (total_days - window_days) // max(n_windows - 1, 1)

    wf_details = []
    for i in range(n_windows):
        win_start = full_start + pd.Timedelta(days=i * step_days)
        win_end = min(win_start + pd.Timedelta(days=window_days), full_end)

        if (win_end - win_start).days < 30:  # 窗口太短，跳过
            continue

        # 策略在该窗口内的周收益
        mask = (nav_series.index >= win_start) & (nav_series.index < win_end)
        window_returns = nav_series.loc[mask, 'weekly_return']

        if len(window_returns) < 5:
            continue

        strat_sharpe = compute_sharpe(window_returns, base_cfg.risk_free_rate)

        # 基准 Sharpe
        bench_sharpe = _compute_equal_weight_benchmark_sharpe(
            weekly_nav, win_start, win_end, base_cfg.risk_free_rate
        )

        if np.isnan(bench_sharpe):
            continue

        wf_details.append({
            'window': i,
            'start': str(win_start.date()),
            'end': str(win_end.date()),
            'strategy_sharpe': round(strat_sharpe, 6),
            'benchmark_sharpe': round(bench_sharpe, 6),
            'relative_sharpe': round(strat_sharpe - bench_sharpe, 6),
            'beat_benchmark': strat_sharpe > bench_sharpe,
        })

    if len(wf_details) == 0:
        return 0.0, []

    n_wins = sum(1 for w in wf_details if w['beat_benchmark'])
    win_rate = n_wins / len(wf_details)

    return win_rate, wf_details


# ── 完整评估 ─────────────────────────────────────────────────────────────────

def evaluate_robustness(
    config_path: str,
    n_mc: int = 400,
    n_wf_windows: int = 9,
    n_trials: int = 2,
    n_jobs: int = -1,
    perturbation: float = 0.15,
    oat: bool = False,
    sps: bool = False,
    sps_horizon: int = 3,
    full_grid: bool = False,
    n_local_mc: int = 50,
) -> RobustnessResult:
    """完整鲁棒性评估 (v4: PSS 替代 PBO + Phase 6 full_grid)。

    1. 运行基准回测 → 获取 Sharpe, skew, kurtosis
    2. 计算 DSR
    3. 运行 MC 生存率测试 (v2 收紧标准)
    4. 计算 PSS 参数稳定性评分 (v4，从 MC 数据零成本计算)
    5. 可选：运行 OAT 多级敏感度分析 (v2 新增)
    6. 运行基准相对胜率 Walk-Forward
    7. 可选：运行 SPS 起点敏感性 (v3 新增)
    8. 可选：运行 Phase 6 全参数网格评估 (v4 新增)
    9. 汇总为 RobustnessResult

    Args:
        config_path: 策略配置 YAML 路径
        n_mc: MC 运行次数 (默认 400)
        n_wf_windows: Walk-Forward 窗口数
        n_trials: 多重测试矫正的变体数
        n_jobs: 并行进程数
        perturbation: MC 扰动幅度 (默认 0.15)
        oat: 是否运行 OAT 多级敏感度 (v2 新增)
        sps: 是否运行 SPS 起点敏感性 (v3 新增)
        sps_horizon: SPS 投资期限 (默认 3 年)
        full_grid: 是否运行 Phase 6 全参数网格 (v4 新增)
        n_local_mc: Phase 6 局部 MC 运行次数 (默认 50)

    Returns:
        RobustnessResult
    """
    project_root = Path(__file__).resolve().parent.parent
    config_abs = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    base_cfg = load_config(config_abs)

    # 1. 基准回测
    result = run_backtest(base_cfg)
    if result.nav_series.empty:
        raise RuntimeError(f"Baseline backtest failed for {config_path}")

    metrics = result.metrics
    weekly_returns = result.nav_series['weekly_return'].dropna()
    sharpe = metrics['sharpe_ratio']
    n_obs = len(weekly_returns)
    skew = float(weekly_returns.skew())
    kurtosis = float(weekly_returns.kurtosis())

    # 2. DSR
    dsr = compute_dsr(sharpe, n_trials, n_obs, skew, kurtosis)

    # 3. MC 生存率 (v2: perturbation param, tightened criterion)
    mc_rate, mc_details = run_mc_survival_test(
        config_path, n_runs=n_mc, perturbation=perturbation, n_jobs=n_jobs
    )

    # 4. PSS 参数稳定性评分 (v4，从 MC 数据零成本计算)
    pss = compute_pss(mc_details)

    # 5. OAT 多级敏感度 (v2 新增)
    oat_result = None
    if oat:
        oat_result = run_oat_sensitivity(config_path, perturbation=perturbation, n_jobs=n_jobs)

    # 6. WF 基准相对胜率
    wf_rate, wf_details = compute_benchmark_relative_win_rate(config_path, n_windows=n_wf_windows)

    # 7. SPS (v3 新增)
    sps_metrics = None
    sps_details = None
    if sps:
        sps_metrics, sps_details = compute_starting_point_sensitivity(
            config_path, horizon_years=sps_horizon, n_jobs=n_jobs
        )

    # 8. Phase 6 全参数网格 (v4 新增)
    grid_result = None
    if full_grid:
        grid_result = run_full_grid(config_path, n_local_mc=n_local_mc, n_jobs=n_jobs)

    # 9. 汇总
    strategy_metrics = {
        'annual_return': metrics['annual_return'],
        'max_drawdown': metrics['max_drawdown'],
        'sharpe_ratio': metrics['sharpe_ratio'],
    }

    result_details = {
        'mc_runs': mc_details,
        'wf_windows': wf_details,
        'dsr_debug': {
            'sharpe': sharpe,
            'n_trials': n_trials,
            'n_obs': n_obs,
            'skew': skew,
            'kurtosis': kurtosis,
        },
    }
    if sps_details is not None and not sps_details.empty:
        result_details['sps_details'] = sps_details.to_dict(orient='records')

    return RobustnessResult(
        dsr=dsr,
        mc_survival_rate=mc_rate,
        benchmark_relative_win_rate=wf_rate,
        strategy_config=str(config_path),
        strategy_metrics=strategy_metrics,
        oat_sensitivity=oat_result,
        pss=pss,
        starting_point_sensitivity=sps_metrics,
        full_grid=grid_result,
        details=result_details,
    )


# ── 报告生成 ─────────────────────────────────────────────────────────────────

def _traffic_light(value: float, kind: str) -> str:
    """Return traffic light emoji for a metric."""
    if kind == 'dsr':
        return '🟢' if value > 0.95 else ('🟡' if value >= 0.80 else '🔴')
    elif kind == 'mc':
        return '🟢' if value > 0.80 else ('🟡' if value >= 0.50 else '🔴')
    elif kind == 'wf':
        return '🟢' if value > 0.80 else ('🟡' if value >= 0.60 else '🔴')
    elif kind == 'pss':
        rp10 = value.get('return_p10', 0) if isinstance(value, dict) else 0
        dp90 = value.get('dd_p90', 1) if isinstance(value, dict) else 1
        if rp10 > 0.10 and dp90 < 0.15:
            return '🟢'
        elif value.get('return_p50', 0) > 0.10 and value.get('dd_p50', 1) < 0.15:
            return '🟡'
        else:
            return '🔴'
    elif kind == 'sps_worst':
        # value = worst annual return (a percentage, e.g. 0.05 = 5%)
        return '🟢' if value > 0.05 else ('🟡' if value >= 0.0 else '🔴')
    return ''


def _overall_verdict(results: list[RobustnessResult]) -> tuple[str, str]:
    """Generate overall verdict from robustness results (v3: 5-indicator system)."""
    if not results:
        return '🔴', '无数据'
    # Pick the best result as primary verdict
    best = max(results, key=lambda r: r.strategy_metrics.get('sharpe_ratio', 0))
    dsr = best.dsr
    mc = best.mc_survival_rate
    wf = best.benchmark_relative_win_rate
    pss_val = best.pss
    sps = best.starting_point_sensitivity

    # Count lights
    lights = [
        _traffic_light(dsr, 'dsr'),
        _traffic_light(mc, 'mc'),
        _traffic_light(wf, 'wf'),
    ]
    if pss_val is not None:
        lights.append(_traffic_light(pss_val, 'pss'))
    if sps is not None:
        lights.append(_traffic_light(sps.get('worst_annual_return', 0), 'sps_worst'))

    green_count = sum(1 for x in lights if x == '🟢')
    red_count = sum(1 for x in lights if x == '🔴')
    total = len(lights)

    if green_count >= total:
        return '🟢', '实盘可上 — 高度鲁棒（5/5 绿）'
    elif green_count >= total - 1 and red_count == 0:
        return '🟢', f'实盘可上 — 高度鲁棒（{green_count}/{total} 绿）'
    elif green_count >= 3 and red_count == 0:
        return '🟡', f'可小资金试（{green_count}/{total} 绿，0 红）'
    elif red_count >= 2:
        return '🔴', f'需要改进（{red_count} 红）'
    else:
        return '🟡', f'可小资金试（{green_count}/{total} 绿）'


# ── Phase 6 全参数网格报告生成 ────────────────────────────────────────────────

def _format_grid_level(param_name: str, level: float) -> str:
    """Format grid level for display: percentage for continuous, raw value for discrete."""
    if param_name == 'top_n':
        return f'{int(level)}'
    elif param_name == 'momentum_window':
        return f'{int(level)}'
    else:
        return f'{level:+.0%}'


def _grid_result_to_dict(gpr) -> dict:
    """Convert GridPointResult to plain dict."""
    return {
        'param_name': gpr.param_name,
        'level': gpr.level,
        'actual_value': gpr.actual_value,
        'sharpe': gpr.sharpe,
        'annual_return': gpr.annual_return,
        'max_drawdown': gpr.max_drawdown,
        'relative_sharpe': gpr.relative_sharpe,
        'mc_survival_rate': gpr.mc_survival_rate,
        'mc_details': gpr.mc_details,
    }


def _generate_grid_per_param_table(
    param_name: str,
    results: list[GridPointResult],
    is_d4_off: bool = False,
) -> list[str]:
    """Generate a per-parameter heatmap table for the grid results."""
    lines = []
    lines.append(f'**{param_name}**')
    lines.append('')
    lines.append('| 扰动 | 实际值 | Sharpe | 年化收益 | 最大回撤 | 相对Sharpe | 局部MC |')
    lines.append('|:----:|:-----:|:------:|:-------:|:-------:|:---------:|:-----:|')
    for gpr in results:
        level_str = _format_grid_level(param_name, gpr.level)
        val_str = (f'{gpr.actual_value:.4f}' if param_name not in ('top_n', 'momentum_window')
                   else f'{int(gpr.actual_value)}')
        sharpe_str = f'{gpr.sharpe:.4f}' if not np.isnan(gpr.sharpe) else 'N/A'
        ret_str = f'{gpr.annual_return*100:.2f}%' if not np.isnan(gpr.annual_return) else 'N/A'
        dd_str = f'{gpr.max_drawdown*100:.2f}%' if not np.isnan(gpr.max_drawdown) else 'N/A'
        rel_str = f'{gpr.relative_sharpe:+.4f}' if not np.isnan(gpr.relative_sharpe) else 'N/A'
        mc_str = f'{gpr.mc_survival_rate*100:.0f}%'
        lines.append(f'| {level_str} | {val_str} | {sharpe_str} | {ret_str} | {dd_str} | {rel_str} | {mc_str} |')
    lines.append('')
    return lines


def _generate_cliff_summary(
    grid_results: dict[str, list[GridPointResult]],
    label: str,
) -> list[str]:
    """Generate cliff-effect summary table."""
    lines = []
    lines.append(f'### 悬崖效应汇总 — {label}')
    lines.append('')
    lines.append('| 参数 | 悬崖位置 | 触发条件 | Sharpe 变化 | DD 变化 | 局部MC 变化 | 严重度 |')
    lines.append('|------|:------:|---------|:---------:|:------:|:---------:|:-----:|')

    for param_name, results in grid_results.items():
        if len(results) < 2:
            continue
        # Find max delta in Sharpe and DD between consecutive levels
        max_sharpe_drop = 0.0
        max_dd_jump = 0.0
        cliff_level = results[0].level
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            if np.isnan(prev.sharpe) or np.isnan(curr.sharpe):
                continue
            sharpe_drop = prev.sharpe - curr.sharpe
            dd_jump = curr.max_drawdown - prev.max_drawdown
            mc_drop = prev.mc_survival_rate - curr.mc_survival_rate
            if sharpe_drop > 0.15 or dd_jump > 0.03:
                if abs(sharpe_drop) + abs(dd_jump) > abs(max_sharpe_drop) + abs(max_dd_jump):
                    max_sharpe_drop = sharpe_drop
                    max_dd_jump = dd_jump
                    cliff_level = curr.level
                    cliff_mc_drop = mc_drop

        if max_sharpe_drop > 0.15 or max_dd_jump > 0.03:
            level_str = _format_grid_level(param_name, cliff_level)
            severity = '🔴 致命' if max_sharpe_drop > 0.30 else (
                '🔴 高' if max_dd_jump > 0.05 else '🟡 中'
            )
            trigger = f'{param_name} 达到 {level_str}'
            lines.append(
                f'| {param_name} | {level_str} | {trigger} | '
                f'{max_sharpe_drop:+.3f} | {max_dd_jump*100:+.1f}pp | '
                f'{cliff_mc_drop*100:+.0f}% | {severity} |'
            )

    if len(lines) <= 4:  # Only header rows, no cliffs found
        lines.append('| — | — | 未发现显著悬崖 | — | — | — | — |')
    lines.append('')
    return lines


def _generate_sharpe_heatmap_overview(
    grid_results: dict[str, list[GridPointResult]],
) -> list[str]:
    """Generate ASCII Sharpe heatmap overview."""
    lines = []
    lines.append('### Sharpe 热力图（概览）')
    lines.append('')

    # Header
    headers = ['参数'] + [f'{lvl:+.0%}' for lvl in PERTURBATION_LEVELS]
    lines.append(' | '.join(headers))
    lines.append('|'.join(['------'] + [':----:'] * 7))

    for param_name in GRID_CONTINUOUS_PARAMS:
        if param_name not in grid_results:
            continue
        results = {gpr.level: gpr for gpr in grid_results[param_name]}
        vals = []
        for lvl in PERTURBATION_LEVELS:
            gpr = results.get(lvl)
            if gpr and not np.isnan(gpr.sharpe):
                vals.append(f'{gpr.sharpe:.3f}')
            else:
                vals.append('N/A')
        lines.append(f'{param_name} | ' + ' | '.join(vals))

    # top_n separately
    if 'top_n' in grid_results:
        top_vals = []
        for gpr in grid_results['top_n']:
            if not np.isnan(gpr.sharpe):
                top_vals.append(f'n={int(gpr.actual_value)}:{gpr.sharpe:.3f}')
        lines.append(f'top_n | ' + ' | '.join(top_vals) + ' |')

    lines.append('')
    return lines


def _write_grid_csv(
    grid_results: dict[str, list[GridPointResult]],
    output_path: Path,
    label: str,
) -> str:
    """Write grid results to CSV and return path."""
    import csv
    safe_label = label.replace(' ', '_').replace('.', '_').replace('+', '_')
    csv_path = output_path / 'grid_data' / f'{safe_label}_grid.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'param_name', 'level', 'actual_value', 'sharpe', 'annual_return',
            'max_drawdown', 'relative_sharpe', 'mc_survival_rate',
        ])
        for param_name, results in grid_results.items():
            for gpr in results:
                writer.writerow([
                    gpr.param_name, gpr.level, gpr.actual_value,
                    gpr.sharpe, gpr.annual_return, gpr.max_drawdown,
                    gpr.relative_sharpe, gpr.mc_survival_rate,
                ])

    return str(csv_path)


def _generate_grid_section(
    r: RobustnessResult,
    label: str,
    output_path: Path,
    next_section_num: int,
) -> list[str]:
    """Generate Phase 6 full-grid section for one strategy."""
    lines = []
    grid_results = r.full_grid
    if grid_results is None:
        return lines

    base_cfg_path = r.strategy_config
    d4_enabled = 'd4_enabled' not in base_cfg_path.lower()  # heuristic

    lines.append('---')
    lines.append('')
    lines.append(f'## {next_section_num}. Phase 6 全参数 7 级网格 — {label}')
    lines.append('')

    # Stat summary
    n_params = len(grid_results)
    n_points = sum(len(v) for v in grid_results.values())
    lines.append(f'- **参数数**: {n_params}')
    lines.append(f'- **格点数**: {n_points}')
    lines.append('')

    # Per-parameter heatmap tables
    lines.append(f'### 每参数热力表 — {label}')
    lines.append('')
    param_order = (GRID_CONTINUOUS_PARAMS + GRID_DISCRETE_PARAMS
                   + (GRID_D4_PARAMS if any(p in grid_results for p in GRID_D4_PARAMS) else []))
    for param_name in param_order:
        if param_name not in grid_results:
            continue
        param_lines = _generate_grid_per_param_table(
            param_name, grid_results[param_name]
        )
        lines.extend(param_lines)

    # Cliff summary
    lines.extend(_generate_cliff_summary(grid_results, label))

    # Sharpe heatmap
    lines.extend(_generate_sharpe_heatmap_overview(grid_results))

    # CSV output
    csv_path = _write_grid_csv(grid_results, output_path, label)
    lines.append(f'- 格点 CSV: `{csv_path}`')
    lines.append('')

    return lines



def generate_robustness_report(
    results: list[RobustnessResult],
    output_dir: str,
    labels: list[str] | None = None,
) -> str:
    """生成 Markdown 鲁棒性评估报告 (v2)。

    Args:
        results: 评估结果列表（每个策略一个）
        output_dir: 输出目录
        labels: 策略标签（与 results 对应）

    Returns:
        报告文本
    """
    if labels is None:
        labels = [r.strategy_config for r in results]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append('# 鲁棒性对比评估报告 (v3)')
    lines.append('')
    lines.append(f'**评估日期**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'**策略数**: {len(results)}')
    lines.append('')

    # ── 策略对比 ──
    lines.append('---')
    lines.append('')
    lines.append('## 策略对比')
    lines.append('')

    # Header
    header = '| 指标 | ' + ' | '.join(labels) + ' |'
    sep = '|------|' + ':--------:|' * len(labels)
    lines.append(header)
    lines.append(sep)

    for metric_key, metric_name in [
        ('annual_return', '年化收益'),
        ('max_drawdown', '最大回撤'),
        ('sharpe_ratio', '夏普比率'),
    ]:
        vals = []
        for r in results:
            v = r.strategy_metrics.get(metric_key, 0)
            if metric_key in ('annual_return', 'max_drawdown'):
                vals.append(f'{v*100:.2f}%')
            else:
                vals.append(f'{v:.3f}')
        lines.append(f'| {metric_name} | ' + ' | '.join(vals) + ' |')

    lines.append('')

    # ── 鲁棒性三指标 ──
    lines.append('---')
    lines.append('')
    lines.append('## 鲁棒性指标')
    lines.append('')

    # DSR
    lines.append('### ① DSR（Deflated Sharpe Ratio）')
    lines.append('')
    lines.append('| 策略 | DSR | 评级 | 解读 |')
    lines.append('|------|:---:|:----:|------|')
    for r, label in zip(results, labels):
        light = _traffic_light(r.dsr, 'dsr')
        if r.dsr > 0.95:
            interpret = '95%+ 概率是真 alpha'
        elif r.dsr >= 0.80:
            interpret = '可接受，有一定过拟合风险'
        else:
            interpret = '很可能只是运气好选中了'
        lines.append(f'| {label} | {r.dsr:.4f} | {light} | {interpret} |')
    lines.append('')

    # MC 生存率 (v2 收紧标准)
    lines.append('### ② MC 生存率（参数扰动盈利概率，v2 收紧标准）')
    lines.append('')
    lines.append('| 策略 | 生存率 | 评级 | 解读 |')
    lines.append('|------|:-----:|:----:|------|')
    for r, label in zip(results, labels):
        light = _traffic_light(r.mc_survival_rate, 'mc')
        if r.mc_survival_rate > 0.80:
            interpret = '参数平面平坦，高度鲁棒'
        elif r.mc_survival_rate >= 0.50:
            interpret = '有一定参数敏感性，但大概率仍满足盈利条件'
        else:
            interpret = '在刀刃上，参数略微偏离就可能亏损'
        lines.append(f'| {label} | {r.mc_survival_rate*100:.1f}% | {light} | {interpret} |')
    lines.append('')

    # 基准相对胜率
    lines.append('### ③ 基准相对胜率（Walk-Forward 相对等权基准）')
    lines.append('')
    lines.append('| 策略 | 胜率 | 评级 | 解读 |')
    lines.append('|------|:---:|:----:|------|')
    for r, label in zip(results, labels):
        light = _traffic_light(r.benchmark_relative_win_rate, 'wf')
        if r.benchmark_relative_win_rate > 0.80:
            interpret = '绝大部分时间跑赢等权持有，真正 alpha'
        elif r.benchmark_relative_win_rate >= 0.60:
            interpret = '多数时间优于基准，可接受'
        else:
            interpret = '大部分时间不如简单等权持有'
        lines.append(f'| {label} | {r.benchmark_relative_win_rate*100:.1f}% | {light} | {interpret} |')
    lines.append('')

    # ── PSS（参数稳定性评分, v4 替代 PBO）──
    pss_available = any(r.pss is not None for r in results)
    if pss_available:
        lines.append('---')
        lines.append('')
        lines.append('## ④ PSS（参数稳定性评分）')
        lines.append('')
        lines.append('| 策略 | 年化 P10/P50/P90 | DD P10/P50/P90 | Sharpe P10/P50/P90 | CV | 评级 |')
        lines.append('|------|:---:|:---:|:---:|:--:|:--:|')
        for r, label in zip(results, labels):
            pss_val = r.pss
            if pss_val is None:
                continue
            light = _traffic_light(pss_val, 'pss')
            ret_str = f"{pss_val.get('return_p10', 0)*100:.1f}%/{pss_val.get('return_p50', 0)*100:.1f}%/{pss_val.get('return_p90', 0)*100:.1f}%"
            dd_str = f"{pss_val.get('dd_p10', 0)*100:.1f}%/{pss_val.get('dd_p50', 0)*100:.1f}%/{pss_val.get('dd_p90', 0)*100:.1f}%"
            sharpe_str = f"{pss_val.get('sharpe_p10', 0):.2f}/{pss_val.get('sharpe_p50', 0):.2f}/{pss_val.get('sharpe_p90', 0):.2f}"
            cv_str = f"r:{pss_val.get('return_cv', 0):.2f} d:{pss_val.get('dd_cv', 0):.2f} s:{pss_val.get('sharpe_cv', 0):.2f}"
            lines.append(f'| {label} | {ret_str} | {dd_str} | {sharpe_str} | {cv_str} | {light} |')
        lines.append('')

    # ── SPS 起点敏感性 (v3 新增) ──
    sps_available = any(r.starting_point_sensitivity is not None for r in results)
    if sps_available:
        lines.append('---')
        lines.append('')
        lines.append('## ⑤ SPS 起点敏感性分析')
        lines.append('')
        lines.append('| 指标 | ' + ' | '.join(labels) + ' |')
        lines.append('|------|' + ':--------:|' * len(labels))
        sps_keys = [
            ('worst_annual_return', '最差起点年化', lambda v: f'{v*100:.2f}%'),
            ('worst_start_date', '最差起点日期', lambda v: str(v)),
            ('mean_annual_return', '均值年化', lambda v: f'{v*100:.2f}%'),
            ('std_annual_return', '年化标准差', lambda v: f'{v*100:.2f}%'),
            ('negative_return_ratio', '负收益比例', lambda v: f'{v*100:.1f}%'),
            ('p10_annual_return', 'P10 年化', lambda v: f'{v*100:.2f}%'),
            ('best_annual_return', '最佳起点年化', lambda v: f'{v*100:.2f}%'),
            ('n_starting_points', '起点数', lambda v: str(v)),
        ]
        for key, name, fmt in sps_keys:
            vals = []
            for r in results:
                sps = r.starting_point_sensitivity
                if sps is None:
                    vals.append('N/A')
                else:
                    vals.append(fmt(sps.get(key, 0) if key != 'worst_start_date' else sps.get(key, 'N/A')))
            lines.append(f'| {name} | ' + ' | '.join(vals) + ' |')
        lines.append('')

        # SPS 解读
        for r, label in zip(results, labels):
            sps = r.starting_point_sensitivity
            if sps is None:
                continue
            worst = sps.get('worst_annual_return', 0)
            worst_date = sps.get('worst_start_date', 'N/A')
            light = _traffic_light(worst, 'sps_worst')
            lines.append(f'- **{label}**: 最差起点 {worst_date}，年化 {worst*100:.2f}% {light}')
        lines.append('')

    # ── OAT 多级敏感度 (v2 新增) ──
    oat_available = any(r.oat_sensitivity is not None for r in results)
    if oat_available:
        lines.append('---')
        lines.append('')
        section_num = '⑥' if sps_available else '⑤'
        lines.append(f'## {section_num} OAT 多级敏感度分析 (v2)')
        lines.append('')
        lines.append('每个策略对 7 个核心参数在 7 个扰动级别（-15% ~ +15%）下的表现。')
        lines.append('')

        for r, label in zip(results, labels):
            oat_data = r.oat_sensitivity
            if oat_data is None:
                continue
            lines.append(f'### {label}')
            lines.append('')
            # 按 MC_PARAMS 顺序排列
            for param_name in MC_PARAMS:
                if param_name not in oat_data:
                    continue
                param_results = oat_data[param_name]
                lines.append(f'**{param_name}**')
                lines.append('')
                lines.append('| 扰动 | Sharpe | 年化收益 | 最大回撤 |')
                lines.append('|:----:|:------:|:-------:|:-------:|')
                for pr in param_results:
                    ret_str = f'{pr["ret"]*100:.2f}%'
                    dd_str = f'{pr["dd"]*100:.2f}%'
                    lines.append(
                        f'| {pr["level"]:+.0%} | {pr["sharpe"]:.4f} | {ret_str} | {dd_str} |'
                    )
                lines.append('')
        lines.append('')

    # ── 五指标综合对比表 (v3 新增) ──
    lines.append('---')
    lines.append('')
    lines.append('## 五指标综合对比')
    lines.append('')
    lines.append('| 指标 | ' + ' | '.join(labels) + ' | 优胜 |')
    lines.append('|------|' + ':--------:|' * len(labels) + ':----:|')

    # ① DSR
    dsr_vals = [f'{r.dsr:.4f} {_traffic_light(r.dsr, "dsr")}' for r in results]
    dsr_best = labels[max(range(len(results)), key=lambda i: results[i].dsr)]
    lines.append('| ① DSR | ' + ' | '.join(dsr_vals) + f' | {dsr_best} |')

    # ② MC 生存率
    mc_vals = [f'{r.mc_survival_rate*100:.1f}% {_traffic_light(r.mc_survival_rate, "mc")}' for r in results]
    mc_best = labels[max(range(len(results)), key=lambda i: results[i].mc_survival_rate)]
    lines.append('| ② MC 生存率 | ' + ' | '.join(mc_vals) + f' | {mc_best} |')

    # ③ WF 相对胜率
    wf_vals = [f'{r.benchmark_relative_win_rate*100:.1f}% {_traffic_light(r.benchmark_relative_win_rate, "wf")}' for r in results]
    wf_best = labels[max(range(len(results)), key=lambda i: results[i].benchmark_relative_win_rate)]
    lines.append('| ③ WF 相对胜率 | ' + ' | '.join(wf_vals) + f' | {wf_best} |')

    # ④ PSS
    if pss_available:
        pss_vals = []
        for r in results:
            pv = r.pss
            if pv is None:
                pss_vals.append('N/A')
            else:
                pss_vals.append(f'{pv.get("return_p10", 0)*100:.1f}%/{pv.get("dd_p90", 0)*100:.1f}% {_traffic_light(pv, "pss")}')
        pss_best = 'N/A'
        pss_candidates = [(i, r.pss) for i, r in enumerate(results) if r.pss is not None]
        if pss_candidates:
            pss_best = labels[max(pss_candidates, key=lambda x: x[1].get('return_p10', 0))[0]]
        lines.append('| **④ PSS** | ' + ' | '.join(pss_vals) + f' | {pss_best} |')

    # ⑤ SPS 最差起点
    if sps_available:
        sps_vals = []
        for r in results:
            sps = r.starting_point_sensitivity
            if sps is None:
                sps_vals.append('N/A')
            else:
                worst = sps.get('worst_annual_return', 0)
                sps_vals.append(f'{worst*100:.2f}% {_traffic_light(worst, "sps_worst")}')
        sps_best = 'N/A'
        sps_candidates = [(i, r.starting_point_sensitivity.get('worst_annual_return', -999))
                          for i, r in enumerate(results) if r.starting_point_sensitivity is not None]
        if sps_candidates:
            sps_best = labels[max(sps_candidates, key=lambda x: x[1])[0]]
        lines.append('| **⑤ SPS 最差起点** | ' + ' | '.join(sps_vals) + f' | {sps_best} |')

    # 综合
    comp_verdicts = []
    for r in results:
        all_lights = [
            _traffic_light(r.dsr, 'dsr'),
            _traffic_light(r.mc_survival_rate, 'mc'),
            _traffic_light(r.benchmark_relative_win_rate, 'wf'),
        ]
        if r.pss is not None:
            all_lights.append(_traffic_light(r.pss, 'pss'))
        if r.starting_point_sensitivity is not None:
            all_lights.append(_traffic_light(r.starting_point_sensitivity.get('worst_annual_return', 0), 'sps_worst'))
        gc = sum(1 for x in all_lights if x == '🟢')
        rc = sum(1 for x in all_lights if x == '🔴')
        comp_verdicts.append(f'{gc}/5 绿' if rc == 0 else f'{gc}/5 绿, {rc} 红')
    lines.append('| **综合** | ' + ' | '.join(comp_verdicts) + ' | |')

    lines.append('')

    # ── 综合判定 ──
    lines.append('---')
    lines.append('')
    lines.append('## 综合判定')
    lines.append('')

    verdict, suggestion = _overall_verdict(results)
    lines.append(f'**综合评级**: {verdict} {suggestion}')
    lines.append('')

    lines.append('| 策略 | 综合评级 | 建议 |')
    lines.append('|------|:------:|------|')
    for r, label in zip(results, labels):
        dsr_light = _traffic_light(r.dsr, 'dsr')
        mc_light = _traffic_light(r.mc_survival_rate, 'mc')
        wf_light = _traffic_light(r.benchmark_relative_win_rate, 'wf')

        lights_list = [dsr_light, mc_light, wf_light]
        if r.pss is not None:
            lights_list.append(_traffic_light(r.pss, 'pss'))
        if r.starting_point_sensitivity is not None:
            lights_list.append(_traffic_light(r.starting_point_sensitivity.get('worst_annual_return', 0), 'sps_worst'))

        green_count = sum(1 for x in lights_list if x == '🟢')
        red_count = sum(1 for x in lights_list if x == '🔴')
        total_n = len(lights_list)

        if green_count >= total_n:
            overall = '🟢'
            advice = '实盘可上 — 高度鲁棒'
        elif green_count >= total_n - 1 and red_count == 0:
            overall = '🟢'
            advice = '实盘可上'
        elif green_count >= 3 and red_count == 0:
            overall = '🟡'
            advice = '可小资金试'
        elif red_count >= 2:
            overall = '🔴'
            advice = '需要改进'
        else:
            overall = '🟡'
            advice = '可小资金试'

        lines.append(f'| {label} | {overall} | {advice} |')

    lines.append('')

    # ── 详细数据 ──
    lines.append('---')
    lines.append('')
    lines.append('## 详细数据')
    lines.append('')

    for r, label in zip(results, labels):
        lines.append(f'### {label}')
        lines.append('')

        # WF 窗口明细
        wf_windows = r.details.get('wf_windows', [])
        if wf_windows:
            lines.append('**Walk-Forward 窗口明细**:')
            lines.append('')
            lines.append('| 窗口 | 起始 | 结束 | 策略 Sharpe | 基准 Sharpe | 相对 Sharpe | 跑赢 |')
            lines.append('|------|------|------|:----------:|:----------:|:---------:|:---:|')
            for w in wf_windows:
                flag = '✅' if w['beat_benchmark'] else '❌'
                lines.append(
                    f'| {w["window"]} | {w["start"]} | {w["end"]} | '
                    f'{w["strategy_sharpe"]:.4f} | {w["benchmark_sharpe"]:.4f} | '
                    f'{w["relative_sharpe"]:.4f} | {flag} |'
                )
            lines.append('')

        # MC 统计 (v2 收紧标准)
        mc_runs = r.details.get('mc_runs', [])
        if mc_runs:
            sharpes = [m['sharpe_ratio'] for m in mc_runs]
            rets = [m['annual_return'] for m in mc_runs]
            dds = [m['max_drawdown'] for m in mc_runs]
            lines.append(f'**MC 统计** ({len(mc_runs)} 次有效运行):')
            lines.append(f'- Sharpe 均值: {np.mean(sharpes):.4f}')
            lines.append(f'- Sharpe 标准差: {np.std(sharpes):.4f}')
            lines.append(f'- Sharpe 范围: [{np.min(sharpes):.4f}, {np.max(sharpes):.4f}]')
            lines.append(f'- 年化收益均值: {np.mean(rets)*100:.2f}%')
            lines.append(f'- 最大回撤均值: {np.mean(dds)*100:.2f}%')
            lines.append(f'- 年化>10% AND DD<15% 次数: '
                         f'{sum(1 for rr, dd in zip(rets, dds) if rr > 0.10 and dd < 0.15)}/{len(mc_runs)}')
            lines.append(f'- MC 生存率: {r.mc_survival_rate*100:.1f}%')
            lines.append('')

        # DSR 调试信息
        dsr_debug = r.details.get('dsr_debug', {})
        if dsr_debug:
            lines.append(f'**DSR 计算参数**:')
            lines.append(f'- 观测 Sharpe: {dsr_debug.get("sharpe", 0):.4f}')
            lines.append(f'- 试验数: {dsr_debug.get("n_trials", 0)}')
            lines.append(f'- 观测数: {dsr_debug.get("n_obs", 0)}')
            lines.append(f'- 偏度: {dsr_debug.get("skew", 0):.4f}')
            lines.append(f'- 峰度: {dsr_debug.get("kurtosis", 0):.4f}')
            lines.append('')

        # PSS 详细数据 (v4)
        pss_val = r.pss
        if pss_val:
            lines.append(f'**PSS 参数稳定性详情**:')
            lines.append(f'- 年化收益 P10/P50/P90: {pss_val.get("return_p10", 0)*100:.2f}% / {pss_val.get("return_p50", 0)*100:.2f}% / {pss_val.get("return_p90", 0)*100:.2f}%')
            lines.append(f'- 最大回撤 P10/P50/P90: {pss_val.get("dd_p10", 0)*100:.2f}% / {pss_val.get("dd_p50", 0)*100:.2f}% / {pss_val.get("dd_p90", 0)*100:.2f}%')
            lines.append(f'- Sharpe P10/P50/P90: {pss_val.get("sharpe_p10", 0):.4f} / {pss_val.get("sharpe_p50", 0):.4f} / {pss_val.get("sharpe_p90", 0):.4f}')
            lines.append(f'- CV (收益/回撤/Sharpe): {pss_val.get("return_cv", 0):.4f} / {pss_val.get("dd_cv", 0):.4f} / {pss_val.get("sharpe_cv", 0):.4f}')
            lines.append(f'- 样本数: {pss_val.get("n_total", 0)}')
            lines.append('')

        # SPS 详细数据 (v3)
        sps = r.starting_point_sensitivity
        if sps:
            lines.append(f'**SPS 起点敏感性详情**:')
            lines.append(f'- 起点数: {sps.get("n_starting_points", 0)}')
            lines.append(f'- 投资期限: {sps.get("horizon_years", 0)} 年')
            lines.append(f'- 最差起点年化: {sps.get("worst_annual_return", 0)*100:.2f}% ({sps.get("worst_start_date", "N/A")})')
            lines.append(f'- 最佳起点年化: {sps.get("best_annual_return", 0)*100:.2f}%')
            lines.append(f'- 均值年化: {sps.get("mean_annual_return", 0)*100:.2f}%')
            lines.append(f'- 年化标准差: {sps.get("std_annual_return", 0)*100:.2f}%')
            lines.append(f'- 负收益比例: {sps.get("negative_return_ratio", 0)*100:.1f}%')
            lines.append(f'- 负净值比例: {sps.get("negative_nav_ratio", 0)*100:.1f}%')
            lines.append(f'- P10 年化: {sps.get("p10_annual_return", 0)*100:.2f}%')
            lines.append(f'- P90 年化: {sps.get("p90_annual_return", 0)*100:.2f}%')
            lines.append('')

    # ── Phase 6 全参数网格 (v4 新增) ──
    grid_available = any(r.full_grid is not None for r in results)
    if grid_available:
        # Determine next section number
        sec_base = 6  # after 5-indicator table
        for r, label in zip(results, labels):
            grid_lines = _generate_grid_section(r, label, output_path, sec_base)
            if grid_lines:
                lines.extend(grid_lines)
                sec_base += 1

    # ── 数据文件 ──
    lines.append('---')
    lines.append('')
    lines.append('## 数据文件')
    lines.append('')
    lines.append(f'- 结构化结果: `{output_dir}/robustness_results.json`')
    if grid_available:
        lines.append(f'- 格点 CSV: `{output_dir}/grid_data/`')
    lines.append('')

    report = '\n'.join(lines)

    # 写入报告
    report_path = output_path / 'ROBUSTNESS_COMPARISON_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)

    # 写入结构化结果 JSON (v4: 包含 pss, sps)
    json_path = output_path / 'robustness_results.json'
    json_data = []
    for r, label in zip(results, labels):
        entry = {
            'label': label,
            'config': r.strategy_config,
            'dsr': r.dsr,
            'mc_survival_rate': r.mc_survival_rate,
            'benchmark_relative_win_rate': r.benchmark_relative_win_rate,
            'pss': r.pss,
            'starting_point_sensitivity': r.starting_point_sensitivity,
            'full_grid': {k: [_grid_result_to_dict(gpr) for gpr in v]
                          for k, v in (r.full_grid or {}).items()},
            'strategy_metrics': r.strategy_metrics,
            'oat_sensitivity': r.oat_sensitivity,
            'dsr_debug': r.details.get('dsr_debug', {}),
            'pss_details': r.details.get('pss_details', {}),
            'sps_details': r.details.get('sps_details', []),
            'mc_summary': {
                'n_runs': len(r.details.get('mc_runs', [])),
                'sharpe_mean': float(np.mean([m['sharpe_ratio'] for m in r.details.get('mc_runs', [])])) if r.details.get('mc_runs') else 0.0,
                'sharpe_std': float(np.std([m['sharpe_ratio'] for m in r.details.get('mc_runs', [])])) if r.details.get('mc_runs') else 0.0,
            },
            'mc_runs': r.details.get('mc_runs', []),
            'wf_windows': r.details.get('wf_windows', []),
        }
        json_data.append(entry)

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str, ensure_ascii=False)

    return report