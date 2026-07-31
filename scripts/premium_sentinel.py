#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调仓日溢价哨兵 (任务22) — 只提示、不自动切换。

被 scripts/rebalance_live.py --premium-check 惰性导入调用；模块导入期不发起任何网络请求
(CI 的 --verify 依赖 rebalance_live 导入期零网络)。

数据源优先级:
  1. tushare fund_daily(最新收盘) + fund_nav(最新单位净值)，token 读项目 .env
     (口径同 scripts/_exp_fetch_premium_data.py: premium = close / unit_nav - 1)
  2. 东方财富公开接口兜底: K线未复权收盘 + pingzhongdata 单位净值
     (与 scripts/_exp_sp500_swap.py em_kline / fetch_513100_nav 已验证方式同口径)
  3. 全部失败时返回带 error 标记的结果，绝不抛异常中断调仓主流程。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# 主标的: 策略"纳指ETF"列对应 513100
TARGET_CODE = '513100'

# 候选表 code -> (名称, 标的类型)。"纳指"=同标的替代, "标普"=跨标的替代
# (清单与任务17 scripts/_exp_fetch_premium_data.py 的 QDII 候选一致)
CANDIDATES = {
    '513300': ('华夏纳斯达克ETF', '纳指'),
    '159941': ('广发纳指ETF', '纳指'),
    '513390': ('华泰柏瑞纳斯达克ETF', '纳指'),
    '159632': ('嘉实纳斯达克ETF', '纳指'),
    '513500': ('博时标普500ETF', '标普'),
    '513650': ('易方达标普500ETF', '标普'),
}
NAMES = {TARGET_CODE: '国泰纳指ETF', **{k: v[0] for k, v in CANDIDATES.items()}}

# 阈值来源: 任务16 溢价盈亏平衡 p*≈2.1% (output/experiments/exp_sp500_swap.md §4.3,
# 回测口径 2.12%)。high=2.5%: 显著超过 p* 才红色告警; low=2.0%: 进入观察区,
# 两档间留滞回带防抖，与既有 p*≈2.1% 框架一致。
THRESHOLD_HIGH = 0.025
THRESHOLD_LOW = 0.020


def _load_env():
    """读项目 .env 注入环境变量 (口径同 _exp_fetch_premium_data.py, 不打印明文)。"""
    env = PROJECT / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def _ts_code(code: str) -> str:
    return f"{code}.{'SZ' if code.startswith('15') else 'SH'}"


def _em_secid(code: str) -> str:
    return f"{'0' if code.startswith('15') else '1'}.{code}"


def _fetch_tushare_one(pro, code: str) -> dict:
    """tushare: fund_daily 最近交易日收盘 + fund_nav 最近单位净值 → 溢价。"""
    from datetime import datetime, timedelta
    beg = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    daily = pro.fund_daily(ts_code=_ts_code(code), start_date=beg)
    nav = pro.fund_nav(ts_code=_ts_code(code), start_date=beg)
    if daily is None or not len(daily) or nav is None or not len(nav):
        raise RuntimeError('fund_daily/fund_nav 返回空')
    d = daily.sort_values('trade_date').iloc[-1]
    n = nav.dropna(subset=['unit_nav']).sort_values('nav_date').iloc[-1]
    close, unav = float(d['close']), float(n['unit_nav'])
    return {'close': close, 'unit_nav': unav, 'premium': close / unav - 1,
            'close_date': str(d['trade_date']), 'nav_date': str(n['nav_date']),
            'source': 'tushare'}


def _fetch_em_one(code: str) -> dict:
    """东财兜底: K线(fqt=0 未复权)最近收盘 + pingzhongdata 最新单位净值。"""
    import requests
    from datetime import datetime, timedelta
    beg = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    r = requests.get(
        'https://push2his.eastmoney.com/api/qt/stock/kline/get',
        params=dict(secid=_em_secid(code), fields1='f1,f2,f3',
                    fields2='f51,f52,f53,f54,f55', klt='101', fqt='0',
                    beg=beg, end='20500101'),
        timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
    d = r.json().get('data')
    if not d or not d.get('klines'):
        raise RuntimeError(f'eastmoney kline 无数据: {code}')
    last = d['klines'][-1].split(',')
    close_date, close = last[0], float(last[2])
    r2 = requests.get(f'https://fund.eastmoney.com/pingzhongdata/{code}.js',
                      timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
    m = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\])\s*;', r2.text)
    if not m:
        raise RuntimeError(f'pingzhongdata 解析失败: {code}')
    arr = json.loads(m.group(1))
    if not arr:
        raise RuntimeError(f'pingzhongdata 净值为空: {code}')
    # x 为毫秒时间戳(UTC), +8h 转北京时间取日期
    nav_date = datetime.utcfromtimestamp(arr[-1]['x'] / 1000 + 8 * 3600).strftime('%Y-%m-%d')
    unav = float(arr[-1]['y'])
    return {'close': close, 'unit_nav': unav, 'premium': close / unav - 1,
            'close_date': close_date, 'nav_date': nav_date, 'source': 'eastmoney'}


def fetch_premiums(codes=None) -> dict:
    """拉取各代码最新 close 与 unit_nav 计算溢价。

    返回 {code: {'close','unit_nav','premium','close_date','nav_date','source'}}；
    失败项为 {'error': 原因, 'premium': None, 'source': 'none'}。绝不抛异常。
    """
    if codes is None:
        codes = [TARGET_CODE] + list(CANDIDATES)
    pro = None
    try:
        _load_env()
        token = os.environ.get('TUSHARE_TOKEN', '')
        if token:
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()
    except Exception:
        pro = None
    out = {}
    for code in codes:
        rec = None
        if pro is not None:
            try:
                rec = _fetch_tushare_one(pro, code)
            except Exception:
                rec = None  # token失效/无权限/无数据 → 该代码走东财兜底
        if rec is None:
            try:
                rec = _fetch_em_one(code)
            except Exception as e:
                rec = {'error': str(e)[:120], 'premium': None, 'source': 'none'}
        out[code] = rec
    return out


def advise(premiums: dict, threshold_high: float = THRESHOLD_HIGH,
           threshold_low: float = THRESHOLD_LOW) -> str:
    """基于溢价结果生成建议文本 (只提示, 不自动切换)。"""
    lines = [f"-- 溢价哨兵 (阈值 {threshold_low*100:.1f}%/{threshold_high*100:.1f}%, 仅提示不自动切换) --"]
    tgt = premiums.get(TARGET_CODE) or {}
    p = tgt.get('premium')
    if p is None:
        lines.append(f"  ⚠️ 降级: 无法获取 {TARGET_CODE} 溢价 ({tgt.get('error', '无数据')}), "
                     f"跳过哨兵判定, 不影响调仓建议")
        return '\n'.join(lines)
    lines.append(f"  {TARGET_CODE}({NAMES[TARGET_CODE]}) 溢价 {p*100:.2f}%  "
                 f"[close={tgt.get('close')} nav={tgt.get('unit_nav')} "
                 f"@{tgt.get('close_date', '?')}, 源:{tgt.get('source', '?')}]")
    if p < threshold_low:
        lines.append(f"  ✅ 低于观察阈值 {threshold_low*100:.1f}%: 溢价正常, 按原计划执行 {TARGET_CODE}")
        return '\n'.join(lines)
    if p < threshold_high:
        lines.append(f"  🟡 观察区 [{threshold_low*100:.1f}%, {threshold_high*100:.1f}%): "
                     f"接近盈亏平衡 p*≈2.1% (任务16), 建议盘中复核实时溢价后再执行")
        return '\n'.join(lines)
    lines.append(f"  🔴 溢价 ≥ 告警阈值 {threshold_high*100:.1f}% (盈亏平衡 p*≈2.1%, 任务16), 候选对比:")
    lines.append(f"    {'代码':<8s} {'名称':<12s} {'标的':<4s} {'溢价':>8s}")
    valid = []
    for code, (name, kind) in CANDIDATES.items():
        rec = premiums.get(code) or {}
        cp = rec.get('premium')
        if cp is None:
            lines.append(f"    {code:<8s} {name:<12s} {kind:<4s} {'获取失败':>8s}")
        else:
            lines.append(f"    {code:<8s} {name:<12s} {kind:<4s} {cp*100:>7.2f}%")
            valid.append((code, name, kind, cp))
    # 同标的(纳指)优先、溢价最低; 纳指候选全失败才退到标普
    nas = sorted((v for v in valid if v[2] == '纳指'), key=lambda x: x[3])
    spx = sorted((v for v in valid if v[2] == '标普'), key=lambda x: x[3])
    best = nas[0] if nas else (spx[0] if spx else None)
    if best is None:
        lines.append('  ⚠️ 候选溢价全部获取失败, 无法给出替代建议')
    elif best[3] >= p:
        lines.append(f"  👉 候选中最低溢价 {best[0]}({best[1]}) 为 {best[3]*100:.2f}%, "
                     f"不低于 {TARGET_CODE}, 建议维持原标的并人工复核")
    else:
        tag = ('同标的(纳指)' if best[2] == '纳指'
               else '跨标的(标普, 注意与策略纳指列的收益差, 见任务16报告)')
        lines.append(f"  👉 建议执行标的: {best[0]}({best[1]}) 溢价 {best[3]*100:.2f}%, {tag}")
    lines.append('  ⚠️ 风险提示: 建议仅供参考, QDII 净值 T+1 存在时差噪声, '
                 '请人工确认实时溢价后再决定, 本工具不自动切换。')
    return '\n'.join(lines)


if __name__ == '__main__':
    # 手工诊断入口 (仅直接运行时联网)
    print(advise(fetch_premiums()))
