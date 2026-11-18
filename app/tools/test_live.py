#!/usr/bin/env python3
"""Test the deployed page in headless Chromium: first visit (service-worker isolation, download with progress), analysis against the reference outputs, cached second visit.
  python tools/test_live.py [https://csinva.io/ai-text-detector-app/]"""
import json, os, sys, time
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://csinva.io/ai-text-detector-app/"
G = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")))["model"]; fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info, flush=True); (None if ok else fails.append(name))


with sync_playwright() as p:
    b = p.chromium.launch(); ctx = b.new_context(); page = ctx.new_page(); logs = []; reqs = []
    page.on("console", lambda m: logs.append(m.type + ": " + m.text[:160])); page.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)[:200])); page.on("request", lambda r: reqs.append(r.url))
    t0 = time.time(); page.goto(URL); seen = []
    while time.time() - t0 < 900:
        try: st = page.evaluate("[document.getElementById('loader').classList.contains('done'), document.getElementById('loader-title').textContent, document.getElementById('loader-detail').textContent, document.querySelector('#loader .bar').getAttribute('aria-valuenow')]")
        except Exception: time.sleep(0.5); continue    # the service worker reloads the page once
        seen.append(st)
        if st[0] or "could not" in st[1]: break
        time.sleep(1)
    check("first visit: model loads", seen[-1][0], f"{time.time() - t0:.0f}s, last: {seen[-1][1:]}; console errors: {[l for l in logs if 'rror' in l][:3]}")
    vals = sorted({int(s[3]) for s in seen}); check("progress bar advanced", len(vals) > 5 and max(vals) >= 90, f"values {vals[:14]}")
    check("downloaded the 7 shards", sum("detector.onnx.part" in u for u in reqs) >= 7, str(sum("detector.onnx.part" in u for u in reqs)))
    print("  isolated:", page.evaluate("window.crossOriginIsolated"), "|", page.evaluate("document.getElementById('runinfo').textContent"))
    for k in ("human", "ai", "mixed"):
        g = G[k]; page.fill("#input", g["text"]); page.evaluate("window.__lastResult = null"); t1 = time.time(); page.click("#analyze"); page.wait_for_function("window.__lastResult", timeout=300000); r = page.evaluate("window.__lastResult")
        mae = sum(abs(a[2] - c[2]) for a, c in zip(r["words"], g["words"])) / len(g["words"]) if len(r["words"]) == len(g["words"]) else 9
        check(f"sample '{k}' matches reference", abs(r["docLogit"] - g["doc_logit"]) < 0.4 and mae < 0.04, f"doc {r['docProb']:.3f} vs {g['doc_prob']:.3f}, word MAE {mae:.4f}, {time.time() - t1:.1f}s")
    page.screenshot(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshot_live.png"), full_page=True); page.close()
    page = ctx.new_page(); reqs2 = []; page.on("request", lambda r: reqs2.append(r.url)); t0 = time.time(); page.goto(URL)
    for _ in range(600):
        try:
            if page.evaluate("document.getElementById('loader').classList.contains('done')"): break
        except Exception: pass
        time.sleep(0.5)
    title = page.evaluate("document.getElementById('loader-title').textContent")
    check("second visit: loaded from the cache, no shard download", sum("detector.onnx.part" in u for u in reqs2) == 0 and "cache" in title, f"{time.time() - t0:.1f}s; {title}")
    b.close()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED"); sys.exit(1 if fails else 0)
