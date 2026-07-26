"""统一回测引擎 — 单次回测与绩效计算。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import (
    ETFS, OFFENSIVE_IDX, DEFENSIVE_IDX,
    load_nav_data, load_pe_percentile, resample_weekly, classify_etfs
)
from src.factors import compute_all_factors
from src.engine_core import (
    compute_crisis_boost, compute_dynamic_hongli,
    compute_inv_vol_weights, compute_score_margin,
    apply_trend_confirmation,
)
from src.strategy import (
    StrategyConfig, load_config,
    score_offensive, select_top, calculate_defense_ratio,
    allocate, check_rebalance, check_stop_loss,
    check_stop_loss_tiered, check_stop_loss_ptiered, MarketState,
    detect_market_state, check_stop_loss_stateful,
    apply_max_alloc_cap,
    apply_individual_momentum_filter, compute_dynamic_weights,
    compute_softmax_allocation
)
# regime features are disabled in final config; import only when available
try:
    from src.regime import (
        load_regime_data, build_regime_lookup, build_regime_lookup_3state,
        get_regime_overrides, get_regime_overrides_3state,
        Regime
    )
except ImportError:
    # Stubs for when regime module is not available
    class Regime:
        RISK_ON = "RISK_ON"
        CAUTIOUS = "CAUTIOUS"
        DEFENSIVE = "DEFENSIVE"
        BUBBLE_WARN = "BUBBLE_WARN"
        CRISIS = "CRISIS"
        @staticmethod
        def _missing_(name):  # support Regime(name) calls used in the code
            return name
    def load_regime_data(*args, **kwargs) -> None: return None
    def build_regime_lookup(*args, **kwargs) -> None: return None
    def build_regime_lookup_3state(*args, **kwargs) -> None: return None
    def get_regime_overrides(*args, **kwargs) -> dict: return {}
    def get_regime_overrides_3state(*args, **kwargs) -> dict: return {}
from src.utils import (
    annualize_return, compute_max_drawdown, compute_sharpe,
    compute_calmar, compute_annual_volatility
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
    5. 汇总绩效指标 (D5 softmax integrated)

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
            'vol_ddof': config.vol_ddof,
            'pe_window_years': config.pe_window_years,
        }
    }
    factors = compute_all_factors(weekly_nav, pe_df, config_dict)
    momentum = factors['momentum']
    volatility = factors['volatility']

    # === T32: 成分股信号加载（CWM + CONC）===
    _constituent_raw = {}  # end_date_str -> {etf_name -> {'cwm': float, 'conc': float}}
    _constituent_enabled = config.constituent_signals_enabled
    if _constituent_enabled:
        signals_path = project_root / config.constituent_signals_path
        if signals_path.exists():
            import csv as _csv
            with open(signals_path, newline='', encoding='utf-8') as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    d = row['end_date'].strip()
                    etf = row['etf_name'].strip()
                    _constituent_raw.setdefault(d, {})[etf] = {
                        'cwm': float(row.get('cwm', 0) or 0),
                        'conc': float(row.get('conc', 0) or 0),
                    }

    # === 4. 转为 numpy 加速回测 ===
    w_prices = weekly_nav.values
    w_index = weekly_nav.index
    n_weeks = len(w_index)
    w_rets = np.diff(w_prices, axis=0) / w_prices[:-1]

    # 因子转为 numpy（对齐回测逻辑）
    mom_values = momentum.values
    vol_values = volatility.values

    # ETF 索引（动态分类）
    etf_names = list(weekly_nav.columns)
    n_etfs = len(etf_names)
    off_idx, def_idx, NASDAQ_IDX = classify_etfs(etf_names)

    # === T32: 构建成分股信号前向填充查找表（必须在此处，w_index 已定义）===
    constituent_signal_lookup = {}  # week_date_str -> {etf_name -> {'cwm': float, 'conc': float}}
    if _constituent_enabled and _constituent_raw:
        sorted_sig_dates = sorted(_constituent_raw.keys())
        for week_dt in w_index:
            week_str = week_dt.strftime('%Y%m%d')
            # 找到 <= week_str 的最新季度末信号日期
            best_date = None
            for sd in sorted_sig_dates:
                if sd <= week_str:
                    best_date = sd
                else:
                    break
            if best_date:
                constituent_signal_lookup[week_str] = _constituent_raw.get(best_date, {})

    # === T35: 市场状态分类器加载 ===
    _regime_enabled = config.regime_enabled
    _regime_lookup = {}
    _regime_3state = config.regime_3state
    if _regime_enabled:
        regime_path = project_root / config.regime_data_path
        if regime_path.exists():
            regime_df = load_regime_data(str(regime_path))
            if _regime_3state:
                _regime_lookup = build_regime_lookup_3state(regime_df)
                print(f"  [regime] Loaded {len(_regime_lookup)} 3-state regime classifications")
            else:
                _regime_lookup = build_regime_lookup(regime_df)
                print(f"  [regime] Loaded {len(_regime_lookup)} regime classifications")
        else:
            print(f"  [regime] WARNING: regime data not found at {regime_path}")
            _regime_enabled = False

    # === 5. 逐周回测 ===
    start_idx = max(config.vol_window, config.mom_window)  # 需要两个因子窗口中较大的一个来预热
    nav = 1.0
    peak = 1.0
    last_alloc = np.zeros(n_etfs)
    last_selected = None  # 上一次选中的进攻 ETF（用于 score_margin 过滤）
    pending_selected = None  # 趋势确认：候选进攻对（indices frozenset）
    pending_count = 0        # 候选对连续保持的周数
    gap_history = []         # 动态margin: score gap历史记录
    max_dd = 0.0

    # 止损状态（原始 + 三层 + 状态感知）
    in_stop_loss = False
    stop_loss_weeks = 0
    stop_loss_level = 0      # 三层止损当前层级 (0=无, 1=L1, 2=L2, 3=L3)
    tiered_recovery_weeks = 0  # 三层止损恢复计数
    recent_weekly_rets = []   # 最近几周策略收益率（用于 L3 熔断判断）

    # 市场状态感知止损状态
    recovery_ctr = 0
    in_recovery = False
    previous_def = 0.0
    market_state = MarketState.NORMAL  # 当前市场状态

    weekly_records = []

    for i in range(start_idx, n_weeks - 1):
        date = w_index[i]

        # --- D1: 动态动量/波动率权重 (P1, 默认关闭, 纯无状态) ---
        if config.d1_enabled:
            d1_mom_w, d1_vol_w = compute_dynamic_weights(w_rets, i, config, off_idx=off_idx)
        else:
            d1_mom_w, d1_vol_w = config.mom_w, config.vol_w

        # --- T35: 市场状态分类器 → 参数覆写 ---
        current_regime = Regime.CAUTIOUS  # default
        eff_mom_w = d1_mom_w
        eff_vol_w = d1_vol_w
        eff_top_n = config.top_n
        eff_def_alloc = config.def_alloc
        eff_stop_loss = config.stop_loss

        if _regime_enabled and _regime_lookup:
            date_str = date.strftime('%Y%m%d')
            regime_info = _regime_lookup.get(date_str)
            if regime_info:
                current_regime = regime_info['regime']
                if _regime_3state:
                    overrides = get_regime_overrides_3state(current_regime, config.regime_overrides)
                else:
                    overrides = get_regime_overrides(current_regime, config.regime_overrides)
                if overrides:
                    eff_mom_w = overrides.get('mom_w_override', eff_mom_w)
                    eff_vol_w = overrides.get('vol_w_override', eff_vol_w)
                    eff_top_n = overrides.get('top_n_override', eff_top_n)
                    eff_def_alloc = overrides.get('def_alloc_override', eff_def_alloc)
                    eff_stop_loss = overrides.get('stop_loss_override', eff_stop_loss)

        # --- Phase 5b: Regime-conditional softmax resolution ---
        eff_softmax_enabled = config.softmax_enabled
        eff_softmax_temperature = config.softmax_temperature
        eff_softmax_hard_top_n = config.softmax_hard_top_n_fallback

        if current_regime in config.softmax_regime_enabled:
            eff_softmax_enabled = config.softmax_regime_enabled[current_regime]
        if current_regime in config.softmax_regime_temperature:
            eff_softmax_temperature = config.softmax_regime_temperature[current_regime]

        # --- 评分 ---
        scores_vec = np.full(n_etfs, -np.inf)
        for j in off_idx:
            mom_val = mom_values[i, j]
            vol_val = vol_values[i, j]
            if not np.isnan(mom_val) and not np.isnan(vol_val):
                scores_vec[j] = eff_mom_w * mom_val - eff_vol_w * vol_val

        # --- T32: 成分股信号加分（CWM + CONC）---
        if _constituent_enabled and constituent_signal_lookup:
            date_str = date.strftime('%Y%m%d')
            sigs = constituent_signal_lookup.get(date_str, {})
            if sigs:
                for j in off_idx:
                    etf = etf_names[j]
                    if etf in sigs and not np.isnan(scores_vec[j]):
                        cwm_val = sigs[etf].get('cwm', 0.0)
                        conc_val = sigs[etf].get('conc', 0.0)
                        scores_vec[j] += config.cwm_weight * cwm_val + config.conc_weight * conc_val

        # --- 选 top_n (or softmax all-offensive) ---
        off_scores = [(scores_vec[j], j) for j in off_idx if not np.isnan(scores_vec[j])]
        off_scores.sort(key=lambda x: x[0], reverse=True)

        # D5: Softmax initial selection — ALL offensive ETFs
        if eff_softmax_enabled:
            selected_off = [j for _, j in off_scores]
        else:
            selected_off = [j for _, j in off_scores[:eff_top_n]]

        # --- Score Margin: 防噪声换仓（仅非 softmax 模式）---
        # 使用共享函数 (engine_core.compute_score_margin)
        if not eff_softmax_enabled and last_selected is not None:
            if len(off_scores) > eff_top_n:
                gap = off_scores[eff_top_n - 1][0] - off_scores[eff_top_n][0]
                eff_margin, gap_history = compute_score_margin(gap, gap_history, config)

                if config.score_margin > 0 or config.dynamic_margin_sensitivity > 0:
                    if gap < eff_margin:
                        valid_last = [j for j in last_selected if j in off_idx and not np.isnan(scores_vec[j])]
                        if len(valid_last) == eff_top_n:
                            selected_off = valid_last
            else:
                # off_scores <= eff_top_n: no gap to track, keep gap_history fresh
                pass

        # --- Trend Confirmation: 趋势确认（候选对连续N周一致才切换）---
        # 使用共享函数 (engine_core.apply_trend_confirmation)
        if config.trend_confirm_weeks > 0 and not eff_softmax_enabled and last_selected is not None:
            selected_off, pending_selected, pending_count = apply_trend_confirmation(
                selected_off, last_selected, pending_selected, pending_count, config
            )

        # --- D4: 单ETF动量过滤器 (P0, 默认关闭, 纯无状态) ---
        # D4 runs before softmax — filters weak ETFs before weights are computed
        extra_defense = 0.0
        if config.d4_enabled:
            selected_off, extra_defense = apply_individual_momentum_filter(
                selected_off, w_rets, i, config, scores_vec, off_idx=off_idx
            )

        # --- D5: Softmax-Weighted Allocation (computed AFTER D4 filter) ---
        sm_weights = None
        if eff_softmax_enabled and selected_off:
            # T26b: Clamp to hard_top_n before softmax — only top N ETFs get weight
            selected_top = selected_off[:eff_softmax_hard_top_n]
            off_scores_dict = {etf_names[j]: float(scores_vec[j]) for j in selected_top
                               if not np.isnan(scores_vec[j])}
            sm_weights = compute_softmax_allocation(off_scores_dict, eff_softmax_temperature)

        # --- vol 三段式防御（纳指 vol 原始逻辑）---
        # 使用共享函数 (strategy.calculate_defense_ratio)
        nasdaq_vol = vol_values[i, NASDAQ_IDX]
        def_ratio = calculate_defense_ratio(nasdaq_vol, config, base_def_alloc=eff_def_alloc)

        # --- Layer 3.5: 危机相关性收敛保护 ---
        # 使用共享函数 (engine_core.compute_crisis_boost)
        crisis_boost = compute_crisis_boost(w_rets, i, off_idx, config)
        if crisis_boost > 0:
            def_ratio = min(def_ratio + crisis_boost, 1.0)

        # --- 市场状态感知止损（P1 Fix #1, 替代三层止损）---
        if config.stateful_stop_loss:
            # 计算状态判定所需信号
            # nasdaq_12w_ret: 从 w_rets 历史计算
            if i >= 12:
                nasdaq_12w_ret = np.prod(1 + w_rets[i-12:i, NASDAQ_IDX]) - 1
            else:
                nasdaq_12w_ret = 0.0

            # nasdaq_vol_pct: 纳指20w波动率在2年(104周)窗口内的百分位
            if i >= 20:
                current_vol = np.std(w_rets[i-20:i, NASDAQ_IDX], ddof=config.vol_ddof) * np.sqrt(52)
                vol_history = [np.std(w_rets[max(0, j-20):j, NASDAQ_IDX], ddof=config.vol_ddof) * np.sqrt(52)
                               for j in range(max(20, i-104), i+1)]
                nasdaq_vol_pct = sum(1 for v in vol_history if v < current_vol) / len(vol_history)
            else:
                nasdaq_vol_pct = 0.5  # 默认中位

            # 当前回撤
            dd_current = (peak - nav) / peak if peak > 0 else 0.0

            market_state = detect_market_state(nasdaq_12w_ret, nasdaq_vol_pct, dd_current, config)

            ss_def, in_recovery, recovery_ctr = check_stop_loss_stateful(
                nav, peak, market_state, config, previous_def, recovery_ctr, in_recovery
            )
            if ss_def > 0:
                def_ratio = max(def_ratio, ss_def)
            previous_def = ss_def

        # --- 止损兜底（Phase A-2 分层 vs D1 三层 vs 原始单层）---
        if config.ptiered_stop_loss:
            level, forced_def = check_stop_loss_ptiered(nav, peak, config)
            def_ratio = max(def_ratio, forced_def)
        elif config.tiered_stop_loss:
            weekly_ret_realized = recent_weekly_rets[-1] if recent_weekly_rets else 0.0

            level, forced_def = check_stop_loss_tiered(
                nav, peak, weekly_ret_realized, recent_weekly_rets, config
            )
            if level >= 2:
                # L2/L3: 强制高防御 + 恢复期计数
                def_ratio = forced_def
                if stop_loss_level == 0:
                    stop_loss_level = level
                    tiered_recovery_weeks = 0
            elif level == 1:
                # L1: 预警 — 防御比例至少 50%
                def_ratio = max(def_ratio, forced_def)
                stop_loss_level = 0
                tiered_recovery_weeks = 0
            else:
                # 无触发
                if stop_loss_level >= 2:
                    # 恢复期内保持高防御
                    def_ratio = config.l2_defense
                    tiered_recovery_weeks += 1
                    recovery_needed = (
                        config.l2_recovery_weeks if stop_loss_level == 2
                        else config.l3_recovery_weeks
                    )
                    if tiered_recovery_weeks >= recovery_needed:
                        stop_loss_level = 0
                        tiered_recovery_weeks = 0
                else:
                    stop_loss_level = 0
                    tiered_recovery_weeks = 0
        elif not config.stateful_stop_loss:
            # 原始单层止损逻辑（保持不变）
            if not in_stop_loss and check_stop_loss(nav, peak, eff_stop_loss):
                in_stop_loss = True
                stop_loss_weeks = 0

            if in_stop_loss:
                def_ratio = max(def_ratio, 0.95)
                stop_loss_weeks += 1
                if stop_loss_weeks >= config.recovery_weeks:
                    in_stop_loss = False

        # --- D4 extra_defense: max(vol阶段防御, D4额外防御) ---
        def_ratio = max(def_ratio, extra_defense)

        # --- 构建仓位 (使用增强的 allocate()) ---
        # 动态hongli_ratio：使用共享函数 (engine_core.compute_dynamic_hongli)
        if def_idx and len(def_idx) >= 2:
            hl_vol = vol_values[i, def_idx[0]]
            eff_hl_ratio = compute_dynamic_hongli(hl_vol, config)
        else:
            eff_hl_ratio = config.hongli_ratio

        # D5: Softmax-Weighted Allocation (特殊路径, DISABLED in v3.0)
        if sm_weights is not None:
            alloc = allocate(
                [], def_ratio, config,
                etf_names=etf_names, off_idx=off_idx, def_idx=def_idx,
                eff_hl_ratio=eff_hl_ratio,
            )
            for j in selected_off:
                etf_name = etf_names[j]
                alloc[j] = (1 - def_ratio) * sm_weights.get(etf_name, 0.0)
        else:
            # 常规路径: inv-vol or equal-weight, 使用增强 allocate()
            off_weights = None
            if config.inv_vol_enabled and selected_off:
                off_weights = compute_inv_vol_weights(
                    w_rets, selected_off, i, config.inv_vol_window, config.vol_ddof
                )

            selected_names = [etf_names[j] for j in selected_off] if selected_off else []
            use_static_cap = config.max_single_alloc < 1.0 and not config.dynamic_weight_cap
            alloc = allocate(
                selected_names, def_ratio, config,
                etf_names=etf_names, off_idx=off_idx, def_idx=def_idx,
                eff_hl_ratio=eff_hl_ratio,
                off_weights=off_weights,
                apply_cap=use_static_cap,
            )

        # --- 权重上限（D2B）— 仅 dynamic_cap 路径需要单独处理 ---
        if config.max_single_alloc < 1.0 and config.dynamic_weight_cap:
            alloc = apply_max_alloc_cap(
                alloc, config.max_single_alloc, off_idx,
                overflow_to_defense_only=config.overflow_to_defense_only,
                dynamic_cap=True,
                market_state=market_state if config.stateful_stop_loss else None,
                config=config,
                def_idx=def_idx
            )

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
            for j in range(n_etfs)
            if not np.isnan(w_rets[i, j])
        )
        # 汇率对冲成本（仅当配置启用）：按纳指仓位比例计提周度成本
        if config.hedge_cost_weekly > 0:
            wret -= alloc[NASDAQ_IDX] * config.hedge_cost_weekly
        nav *= (1 + wret - fee_cost)
        peak = max(peak, nav)

        # 追踪最近几周策略收益率（用于 L3 熔断判断）
        recent_weekly_rets.append(wret - fee_cost)

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
            'stop_loss_level': stop_loss_level,
            'market_state': str(market_state),
            'regime': str(current_regime.value),
            'nasdaq_vol': nasdaq_vol,
            'turnover': turnover,
            'fee_cost': fee_cost,
        }
        # 记录仓位
        for k, etf in enumerate(etf_names):
            record[f'weight_{etf}'] = alloc[k]

        weekly_records.append(record)
        last_alloc = alloc.copy()
        last_selected = selected_off.copy()

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
