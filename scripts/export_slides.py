"""Optional HTML presentation PDF export and page/overflow verification."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz
from playwright.sync_api import sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/"slides/development_deck.html"
    target = root/"slides/development_deck.pdf"
    qa = root/"outputs/logs/slides_qa"
    qa.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
        page.goto(source.as_uri())
        page.evaluate("document.fonts.ready")
        slides = page.locator(".slide")
        count = slides.count()
        if count != 14:
            raise AssertionError(f"Expected 14 slides, found {count}")
        bounds = slides.evaluate_all("""slides => slides.map((s,i) => {
            const box=s.getBoundingClientRect();
            const bad=[...s.querySelectorAll('h1,h2,h3,p,table,svg,img,footer,li')].filter(e=>{
                const r=e.getBoundingClientRect(); const css=getComputedStyle(e);
                return css.display!=='none' && (r.left<box.left-1 || r.right>box.right+1 || r.top<box.top-1 || r.bottom>box.bottom+1);
            }).map(e=>e.tagName+':'+e.textContent.slice(0,60));
            const title=s.querySelector('header h2')?.getBoundingClientRect();
            const number=s.querySelector('.page')?.getBoundingClientRect();
            const body=s.querySelector('.slide-body')?.getBoundingClientRect();
            const takeaway=s.querySelector('.takeaway')?.getBoundingClientRect();
            const footer=s.querySelector('footer')?.getBoundingClientRect();
            if(title && number && title.right>number.left-8)bad.push('title overlaps page number');
            if(body && takeaway && body.bottom>takeaway.top-8)bad.push('body overlaps takeaway');
            if(takeaway && footer && takeaway.bottom>footer.top-8)bad.push('takeaway overlaps footer');
            return {page:i+1,title:s.dataset.title||'',overflow:bad};
        })""")
        if any(row["overflow"] for row in bounds):
            raise AssertionError(bounds)
        page.pdf(path=str(target), prefer_css_page_size=True, print_background=True,
                 margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        browser.close()
    with fitz.open(target) as document:
        if len(document) != count:
            raise AssertionError(f"PDF page mismatch: {len(document)} != {count}")
        for index, page in enumerate(document):
            bounds[index]["pdf_text_characters"] = len(page.get_text().strip())
            if bounds[index]["pdf_text_characters"] < 20:
                raise AssertionError(f"Missing editable text on PDF page {index+1}")
            if abs(page.rect.width/page.rect.height - 16/9) > .005:
                raise AssertionError(f"Unexpected slide aspect on page {index+1}")
            page.get_pixmap(matrix=fitz.Matrix(4/3, 4/3)).save(qa/f"page_{index+1:02d}.png")
    result = {"status": "rendered_for_visual_review", "pdf": target.relative_to(root).as_posix(),
              "pages": count, "slides": bounds, "pptx": "not_generated_runtime_unavailable"}
    (root/"outputs/logs/slides_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": result["pdf"], "pages": count, "overflow": 0}))


if __name__ == "__main__":
    main()
