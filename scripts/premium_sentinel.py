#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调仓日溢价哨兵 (任务22; v2 回落防线—任务28) — 只提示、不自动切换。

被 scripts/rebalance_live.py --premium-check 惰性导入调用；模块导入期不发起任何网络请求
(CI 的 --verify 依赖 rebalance_live 导入期零网络)。

最新溢价数据源优先级:
  1. tushare fund_daily(最新收盘) + fund_nav(最新单位净值)，token 读项目 .env
     (口径同 scripts/_exp_fetch_premium_data.py: premium = close / unit_nav - 1)
  2. 公开接口兜底: 日K未复权收盘 (东财→新浪→腾讯三级链, 任务28 实测东财 push2his
     可能被限流拒连, 新浪/腾讯 close 与 tushare 缓存逐日一致) + 东财 pingzhongdata 单位净值。
  3. 全部失败时返回带 error 标记的结果，绝不抛异常中断调仓主流程。

v2 回落防线 (E4 实证, output/experiments/premium_e4_collapse.md / SOP §6.2):
  - R1 溢价峰值回撤: p5 (5日平滑溢价) 距 20 日峰值回撤 dd20≥2pp 且 p5>1% → 触发
    (附 X=1.5pp 备选口径); 溢价历史 = 缓存 CSV 打底 + 公开源增量补齐。
  - R2 份额扩张预警: 份额 5 日扩张 ≥5% (缓存打底 + 上交所公开接口增量, 仅沪市);
    数据不可得时明确降级提示人工核查。
  入口: collapse_report() / CLI --collapse-check。任何失败只降级不抛异常。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT / 'data' / 'experiments' / 'tushare_cache'
_UA = {'User-Agent': 'Mozilla/5.0'}

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

# 阈值口径 (任务28 修订): low=1.5% 对齐 E3 运营纪律"溢价>1.5% 新增走场外"
# (output/experiments/premium_e3_switch.md 结论5 / SOP §3 纪律表); high=2.5% 红色告警
# 并列示候选对比, 两档间留滞回带防抖。旧"p*≈2.1% 盈亏平衡"框架 (任务16) 已被
# E3 证伪 (溢价高度持续, p* 前提假设不成立), 不再引用。
THRESHOLD_HIGH = 0.025
THRESHOLD_LOW = 0.015

# E4 回落防线参数 (premium_e4_collapse.md §3 判读2 / SOP §6.2 溢价回落防线)
R1_X_MAIN = 0.02      # 主推荐: dd20 ≥ 2pp 触发
R1_X_ALT = 0.015      # 备选稳健口径 (三标的皆正)
R1_P5_FLOOR = 0.01    # 且 p5 > 1% 才有防守意义
R2_SHARE_5D = 0.05    # 份额 5 日扩张 ≥ +5% → 战略预警
P5_WIN, DD20_WIN = 5, 20


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
    """公开源兜底: 日K未复权收盘 (东财→新浪→腾讯) + pingzhongdata 最新单位净值。"""
    closes, src = _fetch_close_hist(code, days=30)
    if not closes:
        raise RuntimeError(f'公开源收盘全部失败: {code}')
    close_date = max(closes)
    close = closes[close_date]
    navs = _fetch_nav_hist(code)
    if not navs:
        raise RuntimeError(f'pingzhongdata 净值为空: {code}')
    nav_date, unav = navs[-1]
    return {'close': close, 'unit_nav': unav, 'premium': close / unav - 1,
            'close_date': close_date, 'nav_date': nav_date, 'source': src}


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
                     f"新增应走场外联接申购 (E3 纪律: 溢价>1.5% 新增走场外, SOP §3), "
                     f"建议盘中复核实时溢价后再执行")
        return '\n'.join(lines)
    lines.append(f"  🔴 溢价 ≥ 告警阈值 {threshold_high*100:.1f}% (新增必走场外, 存量关注回落防线, "
                 f"SOP §3/§6.2), 候选对比:")
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


# ==================================================================
# v2 回落防线 (任务28): R1 溢价峰值回撤 + R2 份额扩张预警
# 口径与 scripts/_exp_premium_e4.py 一致: p5 = 溢价 5 日均值 (min_periods=3),
# dd20 = p5 近 20 日(含当日)峰值 − 当日 p5。历史数据 = 缓存 CSV 打底 + 公开源增量。
# ==================================================================

def _cache_tag(code: str) -> str:
    return f"{code}{'SZ' if code.startswith('15') else 'SH'}"


def _load_premium_cache(code: str) -> list:
    """读缓存 premium_{tag}.csv → [(date, premium)] 升序; 无文件/坏行返回能读到的部分。"""
    import csv
    path = CACHE_DIR / f'premium_{_cache_tag(code)}.csv'
    if not path.exists():
        return []
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                rows.append((r['date'], float(r['premium'])))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort()
    return rows


def _fetch_close_hist(code: str, days: int = 45):
    """近 days 日的日频未复权收盘 {date: close}; 东财→新浪→腾讯三级链。
    返回 (dict, 源名); 全部失败返回 ({}, 'none')。新浪/腾讯 close 已与 tushare
    缓存逐日核验一致 (任务28 验证)。"""
    import requests
    from datetime import datetime, timedelta
    beg = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    try:  # 1) 东财 kline (可能被限流拒连)
        r = requests.get(
            'https://push2his.eastmoney.com/api/qt/stock/kline/get',
            params=dict(secid=_em_secid(code), fields1='f1,f2,f3',
                        fields2='f51,f52,f53,f54,f55', klt='101', fqt='0',
                        beg=beg, end='20500101'),
            timeout=15, headers=_UA)
        d = r.json().get('data') or {}
        out = {}
        for k in d.get('klines') or []:
            p = k.split(',')
            out[p[0]] = float(p[2])
        if out:
            return out, 'eastmoney'
    except Exception:
        pass
    sym = f"{'sz' if code.startswith('15') else 'sh'}{code}"
    n = max(days, 30)
    try:  # 2) 新浪日K
        r = requests.get(
            'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData',
            params={'symbol': sym, 'scale': '240', 'ma': 'no', 'datalen': str(n)},
            timeout=15, headers=_UA)
        out = {x['day']: float(x['close']) for x in (r.json() or [])}
        if out:
            return out, 'sina'
    except Exception:
        pass
    try:  # 3) 腾讯日K (不复权 day 字段)
        r = requests.get('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get',
                         params={'param': f'{sym},day,,,{n},'}, timeout=15, headers=_UA)
        d = (r.json().get('data') or {}).get(sym) or {}
        ks = d.get('day') or d.get('qfqday') or []
        out = {k[0]: float(k[2]) for k in ks}
        if out:
            return out, 'tencent'
    except Exception:
        pass
    return {}, 'none'


def _fetch_nav_hist(code: str) -> list:
    """东财 pingzhongdata 全历史单位净值 [(date, nav)] 升序; 失败抛异常 (由上层兜住)。"""
    import requests
    from datetime import datetime, timezone
    r = requests.get(f'https://fund.eastmoney.com/pingzhongdata/{code}.js',
                     timeout=25, headers=_UA)
    m = re.search(r'Data_netWorthTrend\s*=\s*(\[.*?\])\s*;', r.text)
    if not m:
        raise RuntimeError(f'pingzhongdata 解析失败: {code}')
    arr = json.loads(m.group(1))
    # x 为毫秒时间戳(UTC), +8h 转北京时间取日期
    return [(datetime.fromtimestamp(a['x'] / 1000 + 8 * 3600, tz=timezone.utc).strftime('%Y-%m-%d'),
             float(a['y'])) for a in arr if a.get('y') is not None]


def fetch_premium_history(code: str = TARGET_CODE, online: bool = True):
    """溢价日序列: 缓存打底 + 公开源增量补齐 (只补缓存末日之后, 不回改历史)。
    增量口径同缓存: premium = close / 最近可得 nav (asof ≤7 自然日)。
    返回 (rows, note); 任何网络失败只退化为纯缓存。"""
    rows = _load_premium_cache(code)
    note = f"缓存{len(rows)}行" + (f"@{rows[-1][0]}" if rows else '')
    if not online:
        return rows, note + ', 离线'
    try:
        from datetime import datetime, timedelta
        last = rows[-1][0] if rows else '1900-01-01'
        closes, csrc = _fetch_close_hist(code)
        new_dates = sorted(d for d in closes if d > last)
        if not new_dates:
            return rows, note + ', 无增量'
        navs = _fetch_nav_hist(code)
        added = 0
        for d in new_dates:
            cand = [(nd, nv) for nd, nv in navs if nd <= d]
            if not cand:
                continue
            nd, nv = cand[-1]
            lag = (datetime.strptime(d, '%Y-%m-%d') - datetime.strptime(nd, '%Y-%m-%d')).days
            if lag > 7 or nv <= 0:
                continue
            rows.append((d, closes[d] / nv - 1))
            added += 1
        rows.sort()
        note += f", 增量{added}日(源:{csrc})"
    except Exception as e:
        note += f", 增量失败({str(e)[:60]})"
    return rows, note


def collapse_metrics(prem_rows: list) -> dict:
    """由 [(date, premium)] 计算 p5/dd20 与 R1 判定 (纯函数, 供构造数据离线测试)。"""
    prem_rows = sorted(prem_rows)
    if len(prem_rows) < P5_WIN:
        return {'ok': False, 'msg': f'溢价历史不足 {P5_WIN} 行 (仅 {len(prem_rows)})'}
    prems = [p for _, p in prem_rows]
    p5s = []
    for i in range(len(prems)):
        w = prems[max(0, i - P5_WIN + 1):i + 1]
        p5s.append(sum(w) / len(w) if len(w) >= 3 else None)
    valid = [(prem_rows[i][0], v) for i, v in enumerate(p5s) if v is not None]
    if len(valid) < 5:  # dd20 滞后窗口 min_periods=5 (同 E4 脚本)
        return {'ok': False, 'msg': f'p5 有效点不足 5 个 (仅 {len(valid)})'}
    p5 = valid[-1][1]
    win = valid[-DD20_WIN:]
    peak = max(v for _, v in win)
    peak_date = next(d for d, v in win if v == peak)
    dd20 = peak - p5

    def _fired(x):
        return dd20 >= x and p5 > R1_P5_FLOOR

    return {'ok': True, 'date': prem_rows[-1][0], 'premium': prems[-1],
            'p5': p5, 'p5_peak20': peak, 'p5_peak20_date': peak_date, 'dd20': dd20,
            'r1_main': _fired(R1_X_MAIN), 'r1_alt': _fired(R1_X_ALT),
            'gap_main_pp': (R1_X_MAIN - dd20) * 100,
            'gap_alt_pp': (R1_X_ALT - dd20) * 100, 'n_rows': len(prem_rows)}


def _load_share_cache(code: str) -> list:
    """读缓存 fund_share_{tag}.csv → [(date, 万份)] 升序。"""
    import csv
    path = CACHE_DIR / f'fund_share_{_cache_tag(code)}.csv'
    if not path.exists():
        return []
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            td = (r.get('trade_date') or '').strip()
            if len(td) != 8:
                continue
            try:
                rows[f'{td[:4]}-{td[4:6]}-{td[6:]}'] = float(r['fd_share'])
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows.items())


def _fetch_sse_share(code: str, date: str):
    """上交所 ETF 每日规模公开查询 (任务28 验证: TOT_VOL 万份, 与 tushare fund_share
    一致)。仅沪市; 非交易日/未公布返回 None。"""
    import requests
    r = requests.get(
        'https://query.sse.com.cn/commonQuery.do',
        params={'sqlId': 'COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L',
                'STAT_DATE': date, 'pageHelp.pageSize': 10000},
        timeout=15, headers={**_UA, 'Referer': 'https://www.sse.com.cn/'})
    for row in r.json().get('result') or []:
        if row.get('SEC_CODE') == code:
            return float(row['TOT_VOL'])
    return None


def fetch_share_history(code: str = TARGET_CODE, online: bool = True,
                        trade_dates: list = None):
    """份额日序列: 缓存打底 + 上交所增量 (仅沪市 51xxxx; 深市无公开日频源, 纯缓存)。
    trade_dates 用于确定需补的交易日; 最多补 6 日且逐次 sleep 礼貌限频。"""
    rows = _load_share_cache(code)
    note = f"缓存{len(rows)}行" + (f"@{rows[-1][0]}" if rows else '')
    if not online:
        return rows, note + ', 离线'
    if code.startswith('15'):
        return rows, note + ', 深市无增量源'
    try:
        import time
        last = rows[-1][0] if rows else '1900-01-01'
        todo = [d for d in (trade_dates or []) if d > last][-6:]
        added = 0
        for d in todo:
            v = _fetch_sse_share(code, d)
            if v is not None:
                rows.append((d, v))
                added += 1
            time.sleep(0.6)
        rows.sort()
        note += f', SSE增量{added}日' if todo else ', 无增量'
    except Exception as e:
        note += f', SSE失败({str(e)[:60]})'
    return rows, note


def share_metrics(share_rows: list) -> dict:
    """份额 5 日扩张率; 单日比率>1.5或<0.5 视为拆分等公司行为折算连续 (口径同 E4)。"""
    share_rows = sorted(share_rows)
    if len(share_rows) < 6:
        return {'ok': False, 'msg': f'份额历史不足 6 行 (仅 {len(share_rows)})'}
    vals = [v for _, v in share_rows[-30:]]
    adj = list(vals)
    for i in range(1, len(adj)):
        ratio = adj[i] / adj[i - 1] if adj[i - 1] else 1.0
        if ratio > 1.5 or ratio < 0.5:
            for j in range(i, len(adj)):
                adj[j] /= ratio
    chg5 = adj[-1] / adj[-6] - 1 if adj[-6] else 0.0
    return {'ok': True, 'date': share_rows[-1][0], 'chg5': chg5,
            'r2': chg5 >= R2_SHARE_5D}


def collapse_check(code: str = TARGET_CODE, online: bool = True) -> dict:
    """R1/R2 汇总判定; 绝不抛异常 (失败项带 ok=False + msg)。"""
    try:
        prem_rows, pnote = fetch_premium_history(code, online=online)
        m = collapse_metrics(prem_rows)
        m['note'] = pnote
    except Exception as e:
        m = {'ok': False, 'msg': str(e)[:120], 'note': ''}
    try:
        tdates = [d for d, _ in prem_rows] if m.get('ok') else []
        share_rows, snote = fetch_share_history(code, online=online, trade_dates=tdates)
        s = share_metrics(share_rows)
        s['note'] = snote
    except Exception as e:
        s = {'ok': False, 'msg': str(e)[:80]}
    m['share'] = s
    return m


def collapse_advise(chk: dict) -> str:
    """回落防线判定 → 提示文本 (只提示, 不自动切换; 纯函数供离线测试)。"""
    lines = ['-- 溢价回落防线 (E4/SOP §6.2: R1 峰值回撤 + R2 份额预警; 仅提示不自动切换) --']
    if not chk.get('ok'):
        lines.append(f"  ⚠️ R1 降级: {chk.get('msg', '无数据')} — 请按 SOP §6.2 人工核查溢价回撤, "
                     f"不影响调仓建议")
    else:
        lines.append(f"  数据: {chk.get('note', '')}; 末日 {chk['date']} 溢价 {chk['premium']*100:.2f}%")
        lines.append(f"  p5={chk['p5']*100:.2f}%  20日峰值p5={chk['p5_peak20']*100:.2f}% "
                     f"({chk['p5_peak20_date']})  dd20={chk['dd20']*100:.2f}pp")
        if chk['r1_main']:
            lines.append(f"  🔴 R1 触发 (dd20≥2.0pp 且 p5>1%): 按 SOP §6.2 回落防线执行 — "
                         f"存量纳指腿转零/低溢价通道, 周频调仓日执行勿盘中抢跑")
        elif chk['p5'] <= R1_P5_FLOOR:
            lines.append(f"  ✅ R1 不适用: p5≤1% (已低溢价, 无防守必要)")
        else:
            lines.append(f"  ✅ R1(X=2pp) 未触发: 距触发还差 {chk['gap_main_pp']:.2f}pp 回撤")
        alt = ('🔴 触发' if chk['r1_alt'] else
               (f"未触发, 差 {chk['gap_alt_pp']:.2f}pp" if chk['p5'] > R1_P5_FLOOR
                else '不适用 (p5≤1%)'))
        lines.append(f"     备选口径 X=1.5pp: {alt}")
    s = chk.get('share') or {}
    if not s.get('ok'):
        lines.append(f"  ⚠️ R2 降级: 份额数据不可用 ({s.get('msg', '无数据')}), 请人工核查 "
                     f"(上交所ETF规模公告/基金公司官网, SOP §6.2)")
    elif s['r2']:
        lines.append(f"  🔴 R2 份额预警: 5日扩张 {s['chg5']*100:+.1f}% ≥ +5% → 红色警戒, "
                     f"当周 R1 改每日盘后核查并复核 QDII 额度公告 (SOP §6.2)")
    else:
        lines.append(f"  ✅ R2 未触发: 份额5日变动 {s['chg5']*100:+.2f}% (<+5%) [{s.get('note', '')}]")
    return '\n'.join(lines)


def collapse_report(code: str = TARGET_CODE, online: bool = True) -> str:
    """一键入口: 取数+判定+文本; 任何异常只降级不上抛 (调仓主流程铁律)。"""
    try:
        return collapse_advise(collapse_check(code, online=online))
    except Exception as e:  # 双重兜底
        return f"  ⚠️ 溢价回落防线降级: {str(e)[:80]}, 不影响调仓建议"


if __name__ == '__main__':
    # 手工诊断入口 (仅直接运行时联网); --collapse-check 只跑回落防线
    import sys as _sys
    if '--collapse-check' in _sys.argv:
        print(collapse_report())
    else:
        print(advise(fetch_premiums()))
        print(collapse_report())
