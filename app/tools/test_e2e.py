#!/usr/bin/env python3
"""End-to-end test of the app in headless Chromium: tokenizer parity with `tokenizers`, first load (download progress bar), analysis against the python int8 ONNX outputs,
second visit (served from Cache Storage, no shard requests), and the one-thread fallback without cross-origin isolation.
  PLAYWRIGHT_BROWSERS_PATH=... python app/tools/test_e2e.py [--no-isolation]"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__)); G = json.load(open(os.path.join(HERE, "golden.json"))); PORT = 8799
ISO = "--no-isolation" not in sys.argv
srv = subprocess.Popen([sys.executable, os.path.join(HERE, "serve.py"), str(PORT)] + ([] if ISO else ["--no-isolation"])); time.sleep(1.5)
URL = f"http://127.0.0.1:{PORT}/index.html"; fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info); (None if ok else fails.append(name))


try:
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--enable-features=SharedArrayBuffer"]); ctx = b.new_context()
        # ---- 1. tokenizer parity
        page = ctx.new_page(); page.goto(f"http://127.0.0.1:{PORT}/tools/blank.html")
        res = page.evaluate(r"""async (cases) => {
            const { BPETokenizer } = await import('/js/tokenizer.js'); const tj = await (await fetch('/model/tokenizer.json')).json(); const tk = new BPETokenizer(tj); const bad = [];
            for (const c of cases) { const r = tk.encode(c.text); const plain = c.text === c.text.normalize('NFC') && !/[\u{10000}-\u{10FFFF}]/u.test(c.text);  // offsets: UTF-16 units of the NFC text (tokenizers: code points of the original text)
            const ok = JSON.stringify(r.ids) === JSON.stringify(c.ids) && (!plain || JSON.stringify(r.offsets) === JSON.stringify(c.offsets)) && r.offsets.every(o => o[0] <= o[1] && o[1] <= r.text.length); if (!ok) bad.push({text: c.text.slice(0, 40), n: r.ids.length, m: c.ids.length}); }
            return bad; }""", G["tokenizer"])
        check("tokenizer ids and offsets match the tokenizers library", not res, f"{len(G['tokenizer'])} cases, mismatches: {res}")
        page.close()
        # ---- 2. first load with progress
        page = ctx.new_page(); reqs = []; page.on("request", lambda r: reqs.append(r.url)); logs = []; page.on("console", lambda m: logs.append(m.text)); page.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)))
        page.goto(URL); seen = []; t0 = time.time()
        while time.time() - t0 < 600 and not page.evaluate("document.getElementById('loader').classList.contains('done') || document.getElementById('loader-title').textContent.includes('could not')"):
            seen.append(page.evaluate("[document.getElementById('loader-title').textContent, document.getElementById('loader-detail').textContent, document.querySelector('#loader .bar').getAttribute('aria-valuenow')]")); time.sleep(0.25)
        load_s = time.time() - t0
        check("model loads", page.evaluate("document.getElementById('loader').classList.contains('done')"), f"{load_s:.1f}s; last status: {seen[-1] if seen else None}; console: {[l for l in logs if 'rror' in l][:3]}")
        vals = [int(s[2]) for s in seen]; check("progress bar advances during the first download", len(set(vals)) > 3 and max(vals) >= 90, f"values seen: {sorted(set(vals))[:12]}...")
        check("first load downloaded the shards", sum("detector.onnx.part" in u for u in reqs) == 7, str(sum("detector.onnx.part" in u for u in reqs)))
        title = page.evaluate("document.getElementById('loader-title').textContent"); check("first-load message says downloaded", "downloaded" in title, title)
        info = page.evaluate("document.getElementById('runinfo').textContent"); print("  runinfo:", info)
        # ---- 3. analyze the samples and compare to python
        for k, g in G["model"].items():
            page.fill("#input", g["text"]); page.evaluate("window.__lastResult = null"); t1 = time.time(); page.click("#analyze")
            try: page.wait_for_function("window.__lastResult", timeout=120000)
            except Exception: print("  TIMEOUT; msg:", page.evaluate("document.getElementById('msg').textContent"), "console:", logs[-8:]); raise
            r = page.evaluate("window.__lastResult"); dt = time.time() - t1
            words = r["words"]; gw = g["words"]; same = len(words) == len(gw) and all(abs(a[0] - b_[0]) + abs(a[1] - b_[1]) == 0 for a, b_ in zip(words, gw))
            mae = sum(abs(a[2] - b_[2]) for a, b_ in zip(words, gw)) / max(len(gw), 1) if same else 9
            check(f"sample '{k}': words, doc score and word probabilities match python", same and abs(r["docProb"] - g["doc_prob"]) < 0.02 and mae < 0.02, f"doc {r['docProb']:.3f} vs {g['doc_prob']:.3f}, word MAE {mae:.4f}, {dt:.1f}s ({r['ms'] / 1000:.1f}s in worker)")
        check("heatmap rendered", page.evaluate("document.querySelectorAll('#result .hmx .w').length") > 50)
        page.screenshot(path=os.path.join(HERE, "screenshot_mixed.png"), full_page=True)
        # long text -> several windows
        long_text = " ".join([G["model"]["human"]["text"], G["model"]["ai"]["text"]] * 6); page.fill("#input", long_text); page.evaluate("window.__lastResult = null"); page.click("#analyze")
        page.wait_for_function("window.__lastResult", timeout=600000); r = page.evaluate("window.__lastResult")
        check("long text uses several windows", r["windows"] >= 2, f"{r['tokens']} tokens, {r['windows']} windows, {r['ms'] / 1000:.1f}s")
        page.close()
        # ---- 4. second visit: from Cache Storage
        page = ctx.new_page(); reqs2 = []; page.on("request", lambda r: reqs2.append(r.url)); t0 = time.time(); page.goto(URL)
        page.wait_for_function("document.getElementById('loader').classList.contains('done')", timeout=300000); t2 = time.time() - t0
        title = page.evaluate("document.getElementById('loader-title').textContent")
        check("second visit loads from the cache without downloading shards", sum("detector.onnx.part" in u for u in reqs2) == 0 and "cache" in title, f"{t2:.1f}s; shard requests: {sum('detector.onnx.part' in u for u in reqs2)}; {title}")
        page.fill("#input", G["model"]["mixed"]["text"]); page.click("#analyze"); page.wait_for_function("window.__lastResult", timeout=300000)
        check("analysis works after a cached load", abs(page.evaluate("window.__lastResult.docProb") - G["model"]["mixed"]["doc_prob"]) < 0.02)
        page.close(); b.close()
finally:
    srv.terminate()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED"); sys.exit(1 if fails else 0)
