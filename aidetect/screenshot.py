"""Screenshot each tab of a heatmap page (dev helper for checking the design). python screenshot.py page.html out_prefix
Needs playwright (a throwaway uv venv, see README)."""
import sys
from playwright.sync_api import sync_playwright

page_path, prefix = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 860, "height": 900}, device_scale_factor=2)
    pg.goto(f"file://{page_path}")
    n = pg.locator(".hmx").first.locator(".tabs label").count()
    for i in range(n):
        pg.locator(".hmx").first.locator(".tabs label").nth(i).click()
        pg.locator(".hmx").first.screenshot(path=f"{prefix}_{i}.png")
    b.close()
print("ok", n)
