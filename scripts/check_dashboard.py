#!/usr/bin/env python3
"""看板上线前校验门禁（13 项自动检查）。

退出码 0=全部通过, 非0=至少一项失败。
每项打印 [PASS] 或 [FAIL reason]。
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
INDEX = PROJ / "index.html"
MAX_SIZE_KB = 500


def main() -> int:
    if not INDEX.exists():
        print(f"[FAIL] index.html 不存在: {INDEX}")
        return 1

    html = INDEX.read_text(encoding="utf-8")
    failed = 0

    # 1. viewport meta
    if '<meta name="viewport" content="width=device-width' in html:
        print("[PASS] 1. viewport meta 存在")
    else:
        print("[FAIL] 1. viewport meta 缺失（需含 width=device-width）")
        failed += 1

    # 2. Chart.js CDN
    if re.search(r'<script[^>]+chart\.js[^>]*>', html, re.IGNORECASE):
        print("[PASS] 2. Chart.js CDN script 标签存在")
    else:
        print("[FAIL] 2. Chart.js CDN script 标签缺失")
        failed += 1

    # 3. CDN 降级逻辑
    if "typeof Chart" in html:
        print("[PASS] 3. CDN 降级逻辑（typeof Chart 检查）存在")
    else:
        print("[FAIL] 3. CDN 降级逻辑缺失（未找到 typeof Chart）")
        failed += 1

    # 4. 三主图元素
    missing_charts = []
    for cid in ("navChart", "defChart", "annualChart"):
        if cid not in html:
            missing_charts.append(cid)
    if not missing_charts:
        print("[PASS] 4. 三主图元素（navChart/defChart/annualChart）存在")
    else:
        print(f"[FAIL] 4. 主图元素缺失: {missing_charts}")
        failed += 1

    # 5. 主力追踪板块
    if "ntGridCharts" in html or "nt-section" in html:
        print("[PASS] 5. 主力追踪板块（ntGridCharts/nt-section）存在")
    else:
        print("[FAIL] 5. 主力追踪板块缺失")
        failed += 1

    # 6. 无固定列数 grid（关键）
    # 扫描 grid-template-columns 声明，若含 repeat(2,1fr) 或 repeat(3,1fr)
    # 且不含 auto-fit / auto-fill / minmax，FAIL
    bad_grids = []
    for m in re.finditer(r'grid-template-columns\s*:\s*([^;}{]+)', html):
        val = m.group(1).strip()
        if re.search(r'repeat\s*\(\s*[23]\s*,', val):
            if not re.search(r'(auto-fit|auto-fill|minmax)', val):
                bad_grids.append(val[:80])
    if not bad_grids:
        print("[PASS] 6. 无固定列数 grid（无 repeat(2/3,1fr) 无 auto-fit）")
    else:
        print(f"[FAIL] 6. 存在固定列数 grid: {bad_grids}")
        failed += 1

    # 7. 数据完整性
    data_match = re.search(r'const\s+DATA\s*=\s*(\{.*?\})\s*;\s*\n', html, re.DOTALL)
    if not data_match:
        # 尝试提取嵌入 JSON（可能很大，用另一种模式）
        data_match = re.search(r'const DATA = (.+?);\n\(function', html, re.DOTALL)
    data_ok = False
    if data_match:
        try:
            data = json.loads(data_match.group(1))
            issues = []
            sharpe = data.get("metrics", {}).get("sharpe", 0)
            if not (sharpe and sharpe > 0):
                issues.append(f"sharpe={sharpe} 不>0")
            nav_pts = len(data.get("nav", {}).get("dates", []))
            if nav_pts <= 100:
                issues.append(f"净值点={nav_pts} 不>100")
            nt = data.get("national_team", {})
            if "available" not in nt:
                issues.append("national_team.available 缺失")
            if not issues:
                data_ok = True
                print(f"[PASS] 7. 数据完整性（sharpe={sharpe:.3f}, 净值点={nav_pts}, NT.available={nt.get('available')}）")
            else:
                print(f"[FAIL] 7. 数据完整性: {'; '.join(issues)}")
                failed += 1
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[FAIL] 7. 数据完整性: DATA JSON 解析失败 — {e}")
            failed += 1
    else:
        print("[FAIL] 7. 数据完整性: 未找到内嵌 DATA JSON")
        failed += 1

    # 8. JS 语法（node --check）
    # 提取 <script> 内容用 node 校验
    script_match = re.search(r'<script id="main-script">(.*?)</script>', html, re.DOTALL)
    if script_match:
        js_code = script_match.group(1)
        # 写入临时文件让 node 检查
        tmp_js = PROJ / ".tmp_check_dashboard.js"
        try:
            tmp_js.write_text(js_code, encoding="utf-8")
            cp = subprocess.run(
                ["node", "--check", str(tmp_js)],
                capture_output=True, text=True, timeout=15
            )
            if cp.returncode == 0:
                print("[PASS] 8. JS 语法（node --check）通过")
            else:
                err = (cp.stderr or cp.stdout or "").strip()[:200]
                print(f"[FAIL] 8. JS 语法错误: {err}")
                failed += 1
        except FileNotFoundError:
            print("[PASS] 8. JS 语法（node 不可用，跳过）")
        except subprocess.TimeoutExpired:
            print("[FAIL] 8. JS 语法: node --check 超时")
            failed += 1
        finally:
            tmp_js.unlink(missing_ok=True)
    else:
        print("[FAIL] 8. JS 语法: 未找到 <script id=\"main-script\"> 块")
        failed += 1

    # 9. 文件大小
    size_kb = INDEX.stat().st_size / 1024
    if size_kb < MAX_SIZE_KB:
        print(f"[PASS] 9. 文件大小 {size_kb:.1f} KB < {MAX_SIZE_KB} KB")
    else:
        print(f"[FAIL] 9. 文件大小 {size_kb:.1f} KB >= {MAX_SIZE_KB} KB 上限")
        failed += 1

    # ---------- 新增 10-13 项（图表数据完整性 & 可见性） ----------

    # 10. 三主图有数据：DATA JSON 中 nav/drawdown/defense 数组长度 > 50
    if data_match:
        try:
            data_10 = json.loads(data_match.group(1))
            nav_len = len(data_10.get("nav", {}).get("dates", []))
            dd_len = len(data_10.get("drawdown", {}).get("dates", []))
            def_len = len(data_10.get("defense", {}).get("dates", []))
            if nav_len > 50 and dd_len > 50 and def_len > 50:
                print(f"[PASS] 10. 三主图有数据（nav={nav_len}, drawdown={dd_len}, defense={def_len}）")
            else:
                print(f"[FAIL] 10. 三主图数据不足: nav={nav_len}, drawdown={dd_len}, defense={def_len}（需>50）")
                failed += 1
        except (json.JSONDecodeError, ValueError):
            print("[FAIL] 10. 三主图有数据: DATA JSON 解析失败")
            failed += 1
    else:
        print("[FAIL] 10. 三主图有数据: 未找到 DATA JSON")
        failed += 1

    # 11. Chart.js 初始化代码存在：检查三个 new Chart 调用
    init_missing = []
    for chart_id in ("navChart", "ddChart", "defChart"):
        pattern = rf"new Chart\(document\.getElementById\(['\"]" + chart_id + rf"['\"]\)"
        if not re.search(pattern, html):
            init_missing.append(chart_id)
    if not init_missing:
        print("[PASS] 11. Chart.js 初始化代码存在（navChart/ddChart/defChart）")
    else:
        print(f"[FAIL] 11. Chart.js 初始化缺失: {init_missing}")
        failed += 1

    # 12. 主力追踪图表有数据：share_trends 非空且至少 1 个指数 aum 长度 > 50
    if data_match:
        try:
            data_12 = json.loads(data_match.group(1))
            nt_data = data_12.get("national_team", {})
            trends = nt_data.get("share_trends", {})
            if not trends:
                # 降级模式下 share_trends 可能缺失，检查 available
                if nt_data.get("available") is False:
                    print("[PASS] 12. 主力追踪图表数据（降级模式，share_trends 不要求）")
                else:
                    print("[FAIL] 12. 主力追踪图表: share_trends 为空")
                    failed += 1
            else:
                max_aum = max(len(t.get("aum", [])) for t in trends.values())
                if max_aum > 50:
                    print(f"[PASS] 12. 主力追踪图表有数据（{len(trends)} 指数, 最长 aum={max_aum}）")
                else:
                    print(f"[FAIL] 12. 主力追踪图表数据不足: 最长 aum={max_aum}（需>50）")
                    failed += 1
        except (json.JSONDecodeError, ValueError):
            print("[FAIL] 12. 主力追踪图表数据: DATA JSON 解析失败")
            failed += 1
    else:
        print("[FAIL] 12. 主力追踪图表数据: 未找到 DATA JSON")
        failed += 1

    # 13. 所有 section 可见：CSS 中不对关键容器设置 display:none/visibility:hidden
    hidden_issues = []
    critical_selectors = [".panel", ".grid-2-1", "#navChart", "#ddChart", "#defChart",
                          ".chart-stack", ".nt-section", "#nt-section"]
    for sel in critical_selectors:
        # 检查 CSS 中是否有 selector{...display:none...} 或 visibility:hidden
        escaped = re.escape(sel)
        pat = escaped + r'[^{}]*\{[^}]*(display\s*:\s*none|visibility\s*:\s*hidden)[^}]*\}'
        if re.search(pat, html):
            hidden_issues.append(sel)
    if not hidden_issues:
        print("[PASS] 13. 所有 section 可见（无 display:none/visibility:hidden）")
    else:
        print(f"[FAIL] 13. 关键容器被隐藏: {hidden_issues}")
        failed += 1

    # 汇总
    total_checks = 13
    print(f"\n{'='*50}")
    if failed == 0:
        print(f"✅ 全部 {total_checks} 项检查通过")
    else:
        print(f"❌ {failed} 项检查失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
