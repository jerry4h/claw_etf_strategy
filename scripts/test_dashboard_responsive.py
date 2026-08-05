#!/usr/bin/env python3
"""多分辨率看板验证辅助脚本。

使用方法：
1. 修改 gen_dashboard.py 模板后：
   python scripts/gen_dashboard.py --preview
2. 打开 http://localhost:8000 目视检查
3. 多分辨率验证：
   python scripts/test_dashboard_responsive.py
4. 确认无问题后 Ctrl+C 停止服务，执行正常 commit+push
"""

import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent

# 标准测试分辨率清单
VIEWPORTS = [
    ("iPhone SE", 375, 667),
    ("iPhone 14", 390, 844),
    ("iPad Mini", 768, 1024),
    ("Desktop", 1280, 800),
    ("Desktop Wide", 1920, 1080),
]

# 检查清单
CHECKLIST = [
    "指标卡片不溢出",
    "净值/回撤/防御三图正常渲染",
    "主力追踪量价对比图 grid 自适应列数",
    "tooltip 无抖动",
    "对数/线性切换按钮可点击",
    "事件表不横向溢出",
    "参数网格小屏两列",
]


def _has_playwright() -> bool:
    """检测是否安装了 playwright"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "playwright"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_playwright_screenshots():
    """使用 playwright 对每个分辨率截图"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  ❌ playwright 导入失败，跳过截图。")
        return False

    html_path = PROJ / "index.html"
    if not html_path.exists():
        print("  ❌ index.html 不存在，请先运行 python scripts/gen_dashboard.py")
        return False

    out_dir = PROJ / "output" / "dashboard_screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"file://{html_path}"

    print(f"\n📸 正在截图（保存至 {out_dir.relative_to(PROJ)}/）...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for name, width, height in VIEWPORTS:
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            # 等待图表渲染完成
            page.wait_for_timeout(2000)
            screenshot_path = out_dir / f"{name.replace(' ', '_')}.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  ✅ {name} ({width}×{height}) → {screenshot_path.name}")
            context.close()
        browser.close()

    print(f"\n🎉 所有截图已保存至: {out_dir}")
    return True


def _print_manual_guide():
    """打印手动测试指引"""
    print("\n" + "=" * 60)
    print("📋 手动多分辨率测试指引")
    print("=" * 60)
    print("\n未检测到 playwright，请按以下步骤手动验证：")
    print("\n安装 playwright（可选，推荐）：")
    print("  pip install playwright && python -m playwright install chromium\n")
    print("─" * 60)
    print("\n使用 Chrome DevTools 手动测试：")
    print("1. 打开 http://localhost:8000（需先运行 gen_dashboard.py --preview）")
    print("2. 按 F12 打开 DevTools → 点击设备切换按钮（Ctrl+Shift+M）")
    print("3. 依次设置以下分辨率并检查效果：\n")

    for name, width, height in VIEWPORTS:
        print(f"   📱 {name:15s} → 设置为 {width} × {height}")

    print("\n" + "\u2500" * 60)


def _print_checklist():
    """打印检查清单"""
    print("\n" + "=" * 60)
    print("✅ 视觉检查清单（每个分辨率逐项确认）")
    print("=" * 60 + "\n")
    for item in CHECKLIST:
        print(f"  [ ] {item}")
    print()


def main():
    print("🔍 虾池ETF看板 — 多分辨率响应式验证")
    print(f"   项目根: {PROJ}")

    # 检查 index.html 是否存在
    html_path = PROJ / "index.html"
    if not html_path.exists():
        print("\n⚠️  index.html 不存在！请先运行：")
        print("   python scripts/gen_dashboard.py")
        sys.exit(1)

    html_size = html_path.stat().st_size / 1024
    print(f"   index.html: {html_size:.1f} KB")

    # 打印检查清单
    _print_checklist()

    # 检测 playwright 并决定执行路径
    if _has_playwright():
        print("✅ 检测到 playwright，自动截图模式...")
        success = _run_playwright_screenshots()
        if success:
            print("\n💡 请人工比对截图，确认各分辨率下无排版问题。")
    else:
        _print_manual_guide()

    print("\n" + "\u2500" * 60)
    print("📝 完整工作流：")
    print("   1. python scripts/gen_dashboard.py --preview")
    print("   2. 浏览器打开 http://localhost:8000 目视检查")
    print("   3. python scripts/test_dashboard_responsive.py")
    print("   4. 确认无问题后 Ctrl+C 停止服务 → commit + push")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
