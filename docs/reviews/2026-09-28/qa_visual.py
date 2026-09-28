"""Render and check the standalone review, without starting the prototype app."""
from pathlib import Path
import json
from urllib.parse import unquote, urlparse
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/reviews/2026-09-28"
HTML = OUT / "service_domain_map.html"
AUDIT = Path("C:/Users/User/.agents/skills/design-spatial/scripts/layout-audit.js")


def main():
    results = {"viewports": [], "errors": [], "external_requests": []}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in (390, 800, 1024, 1440):
            page = browser.new_page(viewport={"width": width, "height": 960}, device_scale_factor=1)
            page.on("pageerror", lambda error: results["errors"].append(str(error)))
            page.on("request", lambda req: results["external_requests"].append(req.url)
                    if req.url.startswith(("http://", "https://")) else None)
            page.goto(HTML.as_uri())
            page.evaluate("document.fonts.ready")
            overflow = page.evaluate("document.documentElement.scrollWidth-document.documentElement.clientWidth")
            assert overflow == 0, (width, overflow)
            page.screenshot(path=str(OUT / f"visual_{width}_top.png"))
            page.screenshot(path=str(OUT / f"visual_{width}_full.png"), full_page=True)
            filters = page.locator("[data-filter]")
            interactions = []
            for i in range(filters.count()):
                button = filters.nth(i)
                value = button.get_attribute("data-filter")
                button.click()
                states = page.locator("[data-priority]").evaluate_all(
                    "els=>els.map(e=>({priority:e.dataset.priority,hidden:e.hidden,shown:getComputedStyle(e).display!=='none'}))")
                assert all(s["hidden"] == (value != "all" and s["priority"] != value) for s in states)
                assert all(s["shown"] == (not s["hidden"]) for s in states)
                interactions.append({"filter": value, "visible": sum(s["shown"] for s in states)})
                assert button.get_attribute("aria-pressed") == "true"
            if filters.count():
                page.locator('[data-filter="all"]').click()
            details = page.locator("details")
            for i in range(details.count()):
                details.nth(i).locator("summary").click()
                assert details.nth(i).get_attribute("open") is not None
                details.nth(i).locator("summary").click()
                assert details.nth(i).get_attribute("open") is None
            assert page.evaluate("document.documentElement.scrollWidth-document.documentElement.clientWidth") == 0
            sections = page.locator("section[id]")
            if width in (390, 800, 1440):
                for i in range(sections.count()):
                    section = sections.nth(i)
                    section.screenshot(path=str(OUT / f"visual_{width}_{section.get_attribute('id')}.png"))
            scroll_checks = []
            wraps = page.locator(".table-wrap")
            for i in range(wraps.count()):
                wrap = wraps.nth(i)
                state = wrap.evaluate("e=>({client:e.clientWidth,scroll:e.scrollWidth,tabindex:e.getAttribute('tabindex')})")
                if state["scroll"] > state["client"]:
                    assert state["tabindex"] == "0", state
                    wrap.evaluate("e=>{e.scrollLeft=e.scrollWidth}")
                    state["scrolled_to"] = wrap.evaluate("e=>e.scrollLeft")
                    assert state["scrolled_to"] > 0
                scroll_checks.append(state)
            if any(x.get("scrolled_to", 0) > 0 for x in scroll_checks):
                page.locator("#capability").screenshot(path=str(OUT / f"visual_{width}_capability_scrolled.png"))
            wraps.evaluate_all("els=>els.forEach(e=>{e.scrollLeft=0})")
            links = page.locator("a[href]").evaluate_all("els=>els.map(e=>({href:e.href,text:e.textContent}))")
            for link in links:
                url = urlparse(link["href"])
                if url.scheme == "file":
                    target = Path(unquote(url.path).lstrip("/"))
                    assert target.exists(), (link["text"], target)
                    if target == HTML and url.fragment:
                        assert page.locator(f'[id="{url.fragment}"]').count() == 1
            page.evaluate("window.scrollTo(0,0)")
            audit = None
            if AUDIT.exists():
                audit_code = AUDIT.read_text(encoding="utf-8")
                audit_code = audit_code.replace("ratio:Math.round(ra*10)/10,large", "label:el.textContent.trim(),ratio:Math.round(ra*10)/10,large")
                audit_code = audit_code.replace("ratio:c.ratio, large:c.large", "label:c.label, ratio:c.ratio, large:c.large")
                audit_code = audit_code.replace("collisions: collisions.length,", "collisions: collisions.length, collision_labels:collisions.map(([a,b])=>[a.el.textContent.trim(),b.el.textContent.trim()]),")
                page.add_script_tag(content=audit_code)
                audit = page.evaluate("__audit({contentSelector:'h1,h2,h3,p,button,summary,nav a'})")
                page.screenshot(path=str(OUT / f"visual_{width}_annotated.png"))
                page.evaluate("__clearAudit()")
            actions = page.locator("button,nav a,summary").evaluate_all(
                "els=>els.map(e=>({label:e.textContent.trim(),w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height}))")
            results["viewports"].append({"width": width, "horizontal_overflow": overflow,
                                           "filter_results": interactions, "details_checked": details.count(),
                                           "link_count": len(links), "layout_heuristics": audit,
                                           "action_sizes": actions, "table_scroll_checks": scroll_checks})
            page.close()
        browser.close()
    assert not results["errors"] and not results["external_requests"]
    results["hard_check_failures"] = []
    for viewport in results["viewports"]:
        audit = viewport["layout_heuristics"]
        if audit and not audit["gates_pass"]:
            results["hard_check_failures"].append({"width": viewport["width"], "gates": audit["gates"]})
        small = [x for x in viewport["action_sizes"] if x["w"] < 44 or x["h"] < 44]
        if small:
            results["hard_check_failures"].append({"width": viewport["width"], "small_controls": small})
    (OUT / "visual_validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"widths": [x["width"] for x in results["viewports"]],
                      "overflow": [x["horizontal_overflow"] for x in results["viewports"]],
                      "errors": results["errors"], "external_requests": results["external_requests"]}))
    assert not results["hard_check_failures"], results["hard_check_failures"]


if __name__ == "__main__":
    main()
