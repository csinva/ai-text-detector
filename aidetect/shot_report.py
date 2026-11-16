"""Screenshot the first mixed-document card of the report on its Feature importance tab. python shot_report.py report.html out.png"""
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=1.5)
    pg.goto(f"file://{sys.argv[1]}")
    card = pg.locator(".hmx").first
    card.locator(".tabs label").nth(1).click()
    card.screenshot(path=sys.argv[2]); b.close()
