#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一入口周度刷新脚本（任务 #72）：数据更新 → 校验 → 看板生成 → git 提交推送，全链路无人值守。

供 hermes 定期外部触发（推荐周六，A股周五收盘数据已就绪）。

用法（hermes 调用）:
    /home/ubuntu/claw_etf_strategy/.venv/bin/python \\
        /home/ubuntu/claw_etf_strategy/scripts/weekly_refresh.py
    # 演练模式：全链路执行（含数据更新与看板生成），但最后只打印将提交的文件，
    # 不执行 git add/commit/push：
    ... scripts/weekly_refresh.py --dry-run

步骤链:
    1. 周频 NAV 更新（子进程调用 update_etf_data_tushare.py；增量截止日截断到最近
       完整 ISO 周，本周数据已最新则幂等跳过；更新后把软链接实体化为普通文件并
       清理临时产物，保持 git 干净）
    2. tushare_cache 日频 OHLC+amount 增量补齐（PVD 因子依赖 amount；追加写入，
       不重写既有行）
    3. 数据质量校验（NaN / 新增行收益率跳变 >20% / 日期合法性），失败即中止，
       不进入看板生成与提交
    4. 看板生成（gen_dashboard.py → 覆写 index.html + dashboard/data.json）
    5. git add 精确文件集 + commit + push（无实际变化跳过 commit；push 失败重试 1 次）
    6. 执行摘要（stdout）

退出码语义:
    0 = 成功（含"数据已最新、无变化"的幂等无操作）
    1 = 未预期异常
    2 = 数据更新失败（含 tushare token 失效 —— 明确报错退出，绝不静默用旧数据）
    3 = 已有实例在运行（锁 /tmp/claw_refresh.lock 存在且 < 2 小时）
    4 = 数据质量校验失败
    5 = 看板生成失败
    6 = git 失败（commit 成功但 push 失败时，提交保留在本地，下次运行自动补推）

日志: stdout 带时间戳（hermes 直接捕获），同时追加写 output/refresh_log_YYYYMMDD.log
（*.log 已被 .gitignore 覆盖）。
安全: TUSHARE_TOKEN 只从环境变量 / .env 读取，所有日志输出统一掩码，绝不打印明文。
失败容错: git add 只在全链路成功后进行，任何一步失败均不产生部分提交；数据文件如已
被修改，可手动 `git checkout -- data/ index.html dashboard/` 恢复。
调试钩子: CLAW_REFRESH_FORCE_UPDATE=1 跳过步骤 1 的幂等预检，强制运行更新子进程。
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import _ETF_CACHE_CODE_MAP  # noqa: E402  ETF名 -> '513100SH'

DATA_DIR = ROOT / 'data'
NAV_FILE = DATA_DIR / 'all_etfs_nav_latest.csv'
CACHE_DIR = DATA_DIR / 'experiments' / 'tushare_cache'
DASHBOARD_CFG = ROOT / 'config' / 'strategy_v4_5_pvd.yaml'
LOCK_FILE = Path('/tmp/claw_refresh.lock')
LOCK_MAX_AGE_SEC = 2 * 3600
LOG_FILE = ROOT / 'output' / f"refresh_log_{datetime.now().strftime('%Y%m%d')}.log"

# 步骤 5 的精确提交文件集（目录路径覆盖其下 *.csv）
GIT_PATHS = [
    'data/all_etfs_nav_latest.csv',
    'data/experiments/tushare_cache',
    'index.html',
    'dashboard/data.json',
]

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_DATA = 2
EXIT_LOCK = 3
EXIT_VALIDATE = 4
EXIT_DASHBOARD = 5
EXIT_GIT = 6

JUMP_THRESHOLD = 0.20   # 收益率异常跳变阈值（新增行/尾部行）
_TOKEN = ''             # 加载 .env 后填充，仅用于日志掩码


class StepError(Exception):
    """步骤失败：携带退出码与明确错误信息，中止后续步骤。"""

    def __init__(self, exit_code: int, msg: str):
        super().__init__(msg)
        self.exit_code = exit_code


# ---------- 日志（stdout + 文件，统一掩码 token） ----------
def _mask(text: str) -> str:
    if _TOKEN and _TOKEN in text:
        text = text.replace(_TOKEN, '***TOKEN***')
    return text


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {_mask(str(msg))}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass  # 日志文件写不进不影响主流程


def run_cmd(args, extra_env=None, echo=True):
    """运行子进程，合并 stdout/stderr，逐行掩码后转发到日志。"""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    cp = subprocess.run(args, cwd=str(ROOT), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if echo and cp.stdout:
        for line in cp.stdout.rstrip('\n').split('\n'):
            log(f"    | {line}")
    return cp


# ---------- .env 与 tushare ----------
def load_env():
    """读取项目 .env（与 update_etf_data_tushare.py 同口径），环境变量优先。"""
    global _TOKEN
    env_file = ROOT / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())
    _TOKEN = os.environ.get('TUSHARE_TOKEN', '')


def init_tushare():
    if not os.environ.get('TUSHARE_TOKEN'):
        raise StepError(EXIT_DATA, '未在环境变量 / .env 中找到 TUSHARE_TOKEN，中止（不静默用旧数据）')
    import tushare as ts
    ts.set_token(os.environ['TUSHARE_TOKEN'])
    return ts.pro_api()


def fetch_fund_daily(pro, ts_code: str, start: str, end: str):
    """带 1 次重试的 fund_daily 增量拉取；频率超限等满窗口再试。"""
    last_err = None
    for attempt in (1, 2):
        try:
            df = pro.fund_daily(ts_code=ts_code, start_date=start, end_date=end)
            time.sleep(0.35)  # 频控间隔
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 61.0 if '频率超限' in str(e) else 3.0
            log(f"    ⚠️ fund_daily({ts_code}) 第{attempt}次失败: {_mask(str(e))[:120]}，{wait:.0f}s 后重试")
            time.sleep(wait)
    raise StepError(EXIT_DATA, f'fund_daily({ts_code}) 拉取失败（已重试）: {_mask(str(last_err))[:160]}')


# ---------- 锁 ----------
def acquire_lock() -> bool:
    for _ in range(2):
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - LOCK_FILE.stat().st_mtime
            except FileNotFoundError:
                continue  # 对方刚释放，重试创建
            if age < LOCK_MAX_AGE_SEC:
                return False
            log(f"⚠️ 发现陈旧锁（{age / 60:.0f} 分钟前），清除后接管")
            LOCK_FILE.unlink(missing_ok=True)
    return False


def release_lock():
    LOCK_FILE.unlink(missing_ok=True)


# ---------- 步骤实现 ----------
def end_cap() -> datetime:
    """增量截止日：周六/周日运行取当天（本周五数据已完结）；周一~周五运行
    截断到上个周日，只纳入已完整结束的 ISO 周，避免写入本周未完结快照。"""
    now = datetime.now()
    if now.weekday() >= 5:
        return now
    return now - timedelta(days=now.weekday() + 1)


def _same_or_newer_iso_week(d1: pd.Timestamp, d2: datetime) -> bool:
    a, b = d1.isocalendar(), d2.isocalendar()
    return (a[0], a[1]) >= (b[0], b[1])


def materialize_nav_symlink():
    """update 脚本会把 all_etfs_nav_latest.csv 换成指向带日期新文件的软链接。
    仓库里它是普通文件（带日期文件被 .gitignore 忽略），这里实体化并清理临时产物。"""
    if not NAV_FILE.is_symlink():
        return
    target = (DATA_DIR / os.readlink(NAV_FILE)).resolve()
    content = target.read_bytes()
    NAV_FILE.unlink()
    NAV_FILE.write_bytes(content)
    target.unlink(missing_ok=True)
    (DATA_DIR / '.last_backup_target').unlink(missing_ok=True)
    log(f"  软链接已实体化为普通文件（并清理 {target.name}）")


def step1_update_nav(force: bool):
    """返回 (prev_last_date, n_new_rows, detail)。"""
    if not NAV_FILE.exists():
        raise StepError(EXIT_DATA, f'生产 NAV 文件不存在: {NAV_FILE}')
    prev = pd.read_csv(NAV_FILE)
    prev_last = pd.to_datetime(prev['日期']).max()
    cap = end_cap()
    if _same_or_newer_iso_week(prev_last, cap) and not force:
        return prev_last, 0, f"幂等跳过: NAV 最新行 {prev_last:%Y-%m-%d} 已覆盖截止周（cap={cap:%Y-%m-%d}）"

    cp = run_cmd([sys.executable, str(ROOT / 'scripts' / 'update_etf_data_tushare.py')],
                 extra_env={'ETF_UPDATE_END_DATE': cap.strftime('%Y%m%d')})
    if cp.returncode != 0:
        raise StepError(EXIT_DATA,
                        f'NAV 更新子进程失败 (exit={cp.returncode})。常见原因: tushare token 失效 / '
                        f'接口无权限 / 网络异常。已中止，未使用旧数据继续。')
    materialize_nav_symlink()
    cur = pd.read_csv(NAV_FILE)
    n_new = len(cur) - len(prev)
    last = pd.to_datetime(cur['日期']).max()
    return prev_last, n_new, f"更新完成: 新增 {n_new} 行，最新 {last:%Y-%m-%d}"


def step2_update_cache():
    """5 只资产池 ETF 的 fund_daily 日频缓存增量补齐（追加写，不重写既有行）。
    返回 (added_dict, detail)。"""
    cap_str = end_cap().strftime('%Y%m%d')
    added = {}
    pro = None
    for name, tag in _ETF_CACHE_CODE_MAP.items():
        path = CACHE_DIR / f'fund_daily_{tag}.csv'
        if not path.exists():
            raise StepError(EXIT_DATA, f'缓存文件缺失: {path}，请先跑 scripts/_exp_fetch_premium_data.py 全量重建')
        df = pd.read_csv(path, dtype={'trade_date': str})
        last = str(df['trade_date'].max())
        start = (datetime.strptime(last, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        if start > cap_str:
            log(f"  {name}({tag}): 已最新（{last}），跳过")
            added[path.name] = 0
            continue
        if pro is None:
            pro = init_tushare()
        new = fetch_fund_daily(pro, f'{tag[:6]}.{tag[6:]}', start, cap_str)
        if new is None or len(new) == 0:
            log(f"  {name}({tag}): 区间 {start}~{cap_str} 无新交易日数据")
            added[path.name] = 0
            continue
        new['trade_date'] = new['trade_date'].astype(str)
        new = new[new['trade_date'] > last].sort_values('trade_date')
        new = new.reindex(columns=list(df.columns))  # 与既有文件严格同列序
        if len(new) == 0:
            added[path.name] = 0
            continue
        with open(path, 'a', encoding='utf-8') as f:
            new.to_csv(f, header=False, index=False)
        added[path.name] = len(new)
        log(f"  {name}({tag}): 追加 {len(new)} 行 → {new['trade_date'].max()}")
    total = sum(added.values())
    return added, f"共追加 {total} 行（{sum(1 for v in added.values() if v > 0)}/{len(added)} 个文件有增量）"


def step3_validate(prev_nav_last: pd.Timestamp, cache_added: dict):
    """新数据基本合理性：NaN / 日期合法性 / 新增+尾部行收益率跳变。失败列出全部问题后中止。"""
    problems = []
    today = pd.Timestamp.now().normalize()

    # --- NAV 周频文件 ---
    nav = pd.read_csv(NAV_FILE)
    dates = pd.to_datetime(nav['日期'], errors='coerce')
    if dates.isna().any():
        problems.append('NAV: 存在无法解析的日期')
    else:
        if not dates.is_monotonic_increasing:
            problems.append('NAV: 日期非单调递增')
        if dates.duplicated().any():
            problems.append(f'NAV: 日期重复 {dates[dates.duplicated()].dt.strftime("%Y-%m-%d").tolist()}')
        if dates.max() > today:
            problems.append(f'NAV: 末行日期在未来 {dates.max():%Y-%m-%d}')
    if nav.isnull().any().any():
        bad = nav.columns[nav.isnull().any()].tolist()
        problems.append(f'NAV: 存在 NaN，列 {bad}')
    etf_cols = [c for c in nav.columns if c != '日期']
    n_new = int((dates > prev_nav_last).sum()) if not dates.isna().any() else 0
    tail_n = max(n_new + 1, 6)  # 新增行 + 至少覆盖尾部 5 个周收益
    rets = nav[etf_cols].tail(tail_n).pct_change().abs()
    jumps = rets[rets > JUMP_THRESHOLD].stack()
    for (idx, col), v in jumps.items():
        problems.append(f'NAV: {col} 在 {nav["日期"].iloc[idx]} 周收益跳变 {v * 100:.1f}% > {JUMP_THRESHOLD * 100:.0f}%')

    # --- 日频缓存 ---
    for name, tag in _ETF_CACHE_CODE_MAP.items():
        fname = f'fund_daily_{tag}.csv'
        df = pd.read_csv(CACHE_DIR / fname, dtype={'trade_date': str})
        d = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
        if d.isna().any():
            problems.append(f'{fname}: 存在无法解析的 trade_date')
            continue
        if not d.is_monotonic_increasing or d.duplicated().any():
            problems.append(f'{fname}: trade_date 非单调递增或重复')
        if d.max() > today:
            problems.append(f'{fname}: 末行日期在未来 {d.max():%Y-%m-%d}')
        tail_k = max(cache_added.get(fname, 0) + 1, 6)
        tail = df.tail(tail_k)
        if tail[['close', 'amount']].isnull().any().any():
            problems.append(f'{fname}: 尾部 {tail_k} 行 close/amount 存在 NaN')
        if (tail['close'] <= 0).any() or (tail['amount'] < 0).any():
            problems.append(f'{fname}: 尾部行 close<=0 或 amount<0')
        cjump = tail['close'].pct_change().abs()
        if (cjump > JUMP_THRESHOLD).any():
            bad_dt = tail.loc[cjump.idxmax(), 'trade_date']
            problems.append(f'{fname}: {bad_dt} 日收益跳变 {cjump.max() * 100:.1f}% > {JUMP_THRESHOLD * 100:.0f}%')

    if problems:
        for p in problems:
            log(f"  ❌ {p}")
        raise StepError(EXIT_VALIDATE, f'数据质量校验失败（{len(problems)} 项），已中止，不进入看板生成与提交')
    return f"通过（NAV {len(nav)} 行含新增 {n_new} 行；5 个日频缓存尾部检查通过）"


def step4_dashboard():
    if not DASHBOARD_CFG.exists():
        raise StepError(EXIT_DASHBOARD, f'看板依赖的配置不存在: {DASHBOARD_CFG}')
    cp = run_cmd([sys.executable, str(ROOT / 'scripts' / 'gen_dashboard.py')])
    if cp.returncode != 0:
        raise StepError(EXIT_DASHBOARD, f'gen_dashboard.py 失败 (exit={cp.returncode})')
    return '看板已生成（index.html + dashboard/data.json）'


def step5_git(dry_run: bool):
    cp = run_cmd(['git', 'status', '--porcelain', '--'] + GIT_PATHS, echo=False)
    if cp.returncode != 0:
        raise StepError(EXIT_GIT, 'git status 失败')
    changes = [l for l in cp.stdout.splitlines() if l.strip()]
    if not changes:
        return '目标文件集无实际变化，跳过 commit/push（幂等）'

    msg = f"chore: weekly refresh {datetime.now().strftime('%Y-%m-%d')}"
    log(f"  变更文件 {len(changes)} 个:")
    for line in changes:
        log(f"    {line}")
    if dry_run:
        return f'[dry-run] 将 add 上述 {len(changes)} 个文件并 commit "{msg}" + push，本次未执行'

    if run_cmd(['git', 'add', '--'] + GIT_PATHS).returncode != 0:
        raise StepError(EXIT_GIT, 'git add 失败')
    if run_cmd(['git', 'commit', '-m', msg]).returncode != 0:
        raise StepError(EXIT_GIT, 'git commit 失败')
    for attempt in (1, 2):
        if run_cmd(['git', 'push', 'origin', 'HEAD']).returncode == 0:
            return f'已 commit + push："{msg}"（{len(changes)} 个文件）'
        if attempt == 1:
            log('  ⚠️ push 失败，15s 后重试 1 次')
            time.sleep(15)
    raise StepError(EXIT_GIT, f'push 失败（已重试 1 次）。commit "{msg}" 已保留在本地，'
                              f'网络恢复后手动 push 或等待下次运行自动补推')


# ---------- 主流程 ----------
def main() -> int:
    ap = argparse.ArgumentParser(description='周度全链路刷新（数据→校验→看板→git）')
    ap.add_argument('--dry-run', action='store_true',
                    help='全链路执行但不 git add/commit/push，只打印将提交的文件')
    args = ap.parse_args()

    load_env()

    if not acquire_lock():
        log(f'❌ 另一实例正在运行（{LOCK_FILE} 存在且未过期），退出。exit={EXIT_LOCK}')
        return EXIT_LOCK

    results = []  # (步骤名, 结果描述)
    code = EXIT_OK
    try:
        log('=' * 62)
        log(f"周度刷新开始 {'[dry-run]' if args.dry_run else ''} （项目: {ROOT}）")
        log('=' * 62)

        force = os.environ.get('CLAW_REFRESH_FORCE_UPDATE') == '1'
        log('Step 1/5: 周频 NAV 数据更新')
        prev_last, n_new, detail = step1_update_nav(force)
        results.append(('1 NAV更新', detail)); log(f"  ✅ {detail}")

        log('Step 2/5: tushare_cache 日频增量补齐')
        cache_added, detail = step2_update_cache()
        results.append(('2 缓存增量', detail)); log(f"  ✅ {detail}")

        log('Step 3/5: 数据质量校验')
        detail = step3_validate(prev_last, cache_added)
        results.append(('3 数据校验', detail)); log(f"  ✅ {detail}")

        log('Step 4/5: 看板生成')
        detail = step4_dashboard()
        results.append(('4 看板生成', detail)); log(f"  ✅ {detail}")

        log('Step 5/5: git 提交与推送')
        detail = step5_git(args.dry_run)
        results.append(('5 git提交', detail)); log(f"  ✅ {detail}")

    except StepError as e:
        code = e.exit_code
        results.append(('中止', str(e)))
        log(f"❌ {e}")
    except Exception as e:  # noqa: BLE001
        code = EXIT_UNEXPECTED
        results.append(('中止', f'未预期异常: {e}'))
        import traceback
        log(f"❌ 未预期异常: {_mask(traceback.format_exc())}")
    finally:
        release_lock()

    log('-' * 62)
    log('执行摘要:')
    for step, detail in results:
        log(f"  [{step}] {detail}")
    log(f"{'✅ 全链路成功' if code == EXIT_OK else f'❌ 刷新失败'}，退出码 {code}")
    return code


if __name__ == '__main__':
    sys.exit(main())
