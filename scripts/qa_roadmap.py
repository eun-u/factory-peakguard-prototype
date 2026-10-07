"""Render roadmap and measure viewport overflow (optional Playwright dependency)."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
results = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for width in (390, 1024, 1440):
        page = browser.new_page(viewport={"width": width, "height": 900}, device_scale_factor=1)
        page.goto((ROOT/"docs/roadmap.html").as_uri())
        page.screenshot(path=str(ROOT/f"outputs/logs/ui_{width}.png"), full_page=True)
        overflow = page.evaluate("document.documentElement.scrollWidth-document.documentElement.clientWidth")
        if overflow:
            print(page.evaluate("Array.from(document.querySelectorAll('body *')).map(e=>({tag:e.tagName,c:e.className,r:e.getBoundingClientRect().right})).filter(e=>e.r>innerWidth+1).slice(0,20)"))
        assert overflow == 0, (width, overflow)
        results.append({"width": width, "horizontal_overflow": overflow, "title": page.title()})
        page.close()
    browser.close()
(ROOT/"outputs/logs/ui_validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False))
