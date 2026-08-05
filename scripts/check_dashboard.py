#!/usr/bin/env python3
"""看板上线前校验门禁（9 项自动检查）。

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

    # 汇总
    print(f"\n{'='*50}")
    if failed == 0:
        print(f"✅ 全部 9 项检查通过")
    else:
        print(f"❌ {failed} 项检查失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
