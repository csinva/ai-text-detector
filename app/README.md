# AI text detector web app

A static page that runs the detector in the browser. Type or paste text and the text lights up as it is read: the heat map is drawn on the words you type, and it updates when you pause, finish a few words or paste. Words waiting for the model shimmer, and a light sweeps across them as they are scored. The text never leaves the device.

**Live:** https://csinva.io/ai-text-detector/ (https://csinva.github.io/ai-text-detector/ redirects there). This folder is the whole site; `.github/workflows/pages.yml` publishes it (without `tools/`) to GitHub Pages.

- **Runtime.** [ONNX Runtime Web](https://onnxruntime.ai/) (WebAssembly, SIMD, threads). The task asked for WebLLM, but WebLLM only runs chat LLMs compiled to the MLC format (token generation, causal attention); this model is a bidirectional classifier with a token head and a document head, so it is exported to ONNX instead. The user experience is the same: a progress bar on the first visit, then the model is reused from the browser.
- **Model.** The round-4 model (`runs/lic5_nano_r15`), exported by `tools/export_onnx.py`: `model/detector.onnx.part000..006` (7 shards of at most 48 MB, GitHub Pages rejects files over 100 MB; 347 MB in total) plus `model/tokenizer.json` (4.8 MB) and `model/manifest.json` (sizes, sha256, version). int8 weights; matches the fp32 torch model (document logit correlation 0.998, word probability correlation 0.996, AUROC within 0.7 points).
- **First visit.** `js/worker.js` downloads the shards with a progress bar, checks the sha256, stores them in Cache Storage and starts the model. **Later visits** read the shards from Cache Storage (no network) and show "loaded from your browser cache". A new `manifest.json` version replaces the old shards. The page has a link to remove the cached model.
- **Live reading.** `js/app.js` re-reads the text 0.65 s after you pause, after at least 3 finished words if a second has passed since the last read, and at once on paste. The worker keeps only the newest request, reuses unchanged 1,024-token windows and streams long texts in window by window.
- **Threads.** WASM threads need cross-origin isolation, which GitHub Pages cannot set with headers; `coi-serviceworker.js` (MIT) provides it (the first visit reloads once). Without it the model runs on one thread, about 3x slower.
- **Memory.** About 1 GB of RAM while the model loads, then about 0.7 GB. Use a desktop browser.
- **Browsers.** Tested in Chromium (the engine of Chrome, Edge, Brave and Opera) with and without cross-origin isolation. Firefox 114+ and Safari 16.4+ provide the features used but could not be tested on this server.

## Layout

```
app/
  index.html, style.css
  js/app.js        live editor (heat layer behind the textarea), triggers, score card, loading bar
  js/worker.js     download + cache + onnxruntime-web session, analysis (newest text wins, window reuse, partial results)
  js/tokenizer.js  byte-level BPE with character offsets (same ids as the tokenizers library)
  js/detector.js   windows of 1,024 tokens, overlap averaging, token -> word aggregation (ported from aidetect/infer.py)
  ort/             onnxruntime-web 1.22.0 (ort.wasm.min.mjs, ort-wasm-simd-threaded.{mjs,wasm}), MIT
  model/           shards, tokenizer.json, manifest.json, config.json
  coi-serviceworker.js, LICENSE, NOTICE
  tools/           export_onnx.py, shard.py, verify_onnx.py, make_golden.py, serve.py, test_e2e.py, test_live.py (not deployed)
```

## Rebuild the model

```bash
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
M=$V45_DATA_ROOT/data/aidetect/runs/lic5_nano_r15/model.pt
python app/tools/export_onnx.py --model $M --out app/model --tmp $V45_DATA_ROOT/data/aidetect/app_tmp   # ONNX + int8 + tokenizer
python app/tools/verify_onnx.py --model $M                          # compare with the fp32 torch model
python app/tools/shard.py                                           # 48 MB shards + manifest.json
python app/tools/make_golden.py                                     # golden outputs for the browser test
```

`model/detector.onnx` (the unsplit file) is git-ignored; the shards are committed because Pages serves them.

## Test

```bash
python app/tools/serve.py 8765            # http://127.0.0.1:8765/ (adds the COOP/COEP headers)
PLAYWRIGHT_BROWSERS_PATH=... python app/tools/test_e2e.py     # headless Chromium: tokenizer parity, first load, analysis vs python, live typing, paste, cached second visit
python app/tools/test_live.py [url]       # the deployed page
```

## Speed

Headless Chromium on a 176-core server (8 WASM threads): about 0.5 s for a 130-token text, 4.8 s for 1,566 tokens (two windows). With one thread: 1.7 s and 27 s. Loading the cached model takes about 2 to 5 s.

## Limits

Texts over 30,000 characters are cut. The tokenizer does not treat special-token strings such as `<|endoftext|>` specially. The model's known weak spots (informal human text, paraphrasing tools) are in `reports/01_runs.html`; the score is not evidence about any person.
