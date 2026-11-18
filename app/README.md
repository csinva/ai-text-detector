# AI text detector web app

**Deployed from a separate public repository, https://github.com/csinva/ai-text-detector-app (live: https://csinva.io/ai-text-detector-app/).** That repository holds the shipped model shards and contains no training code or internal paths; this folder is the source. Model shards are not tracked here (they are git-ignored; rebuild them with the commands below, or copy them from the public repository).

A static page that runs the detector in the browser. Paste text, get a document score and a word-level heatmap. The text never leaves the device.

- **Runtime.** [ONNX Runtime Web](https://onnxruntime.ai/) (WebAssembly, SIMD, threads). The task asked for WebLLM, but WebLLM only runs chat LLMs compiled to the MLC format (token generation, causal attention); this model is a bidirectional classifier with a token head and a document head, so it is exported to ONNX instead. The user experience is the same: a progress bar on the first visit, then the model is reused from the browser.
- **Model.** `model/detector.onnx.part000..006` (7 shards of at most 48 MB, GitHub Pages rejects files over 100 MB; 347 MB in total) plus `model/tokenizer.json` (4.8 MB) and `model/manifest.json` (sizes, sha256, version). int8 weights; matches the fp32 torch model (document logit correlation 0.999, word probability correlation 0.999, AUROC within 0.3 points).
- **First visit.** `js/worker.js` downloads the shards with a progress bar, checks the sha256, stores them in Cache Storage and starts the model. **Later visits** read the shards from Cache Storage (no network) and show "loaded from your browser cache". A new `manifest.json` version replaces the old shards. The page has a link to remove the cached model.
- **Threads.** WASM threads need cross-origin isolation, which GitHub Pages cannot set with headers; `coi-serviceworker.js` (MIT) provides it (the first visit reloads once). Without it the model runs on one thread, which is slower.
- **Memory.** About 1 GB of RAM while the model loads (shards, the joined model and the runtime copy), then about 0.7 GB. Use a desktop browser; Chrome, Edge, Firefox and Safari 16.4+ work.

## Layout

```
app/
  index.html, style.css
  js/app.js        UI, loading bar, heatmap card
  js/worker.js     download + cache + onnxruntime-web session, analysis
  js/tokenizer.js  byte-level BPE with character offsets (same ids and offsets as the tokenizers library)
  js/detector.js   windows of 1,024 tokens, overlap averaging, token -> word aggregation (ported from aidetect/infer.py)
  ort/             onnxruntime-web 1.22.0 (ort.wasm.min.mjs, ort-wasm-simd-threaded.{mjs,wasm}), MIT
  model/           shards, tokenizer.json, manifest.json, config.json
  coi-serviceworker.js
  tools/           export_onnx.py, shard.py, verify_onnx.py, make_golden.py, serve.py, test_e2e.py (not deployed)
```

## Rebuild the model

```bash
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
M=$V45_DATA_ROOT/data/aidetect/runs/lic3_nano_r10/model.pt        # or runs/lic5_nano_r15/model.pt
python app/tools/export_onnx.py --model $M --out app/model --tmp $V45_DATA_ROOT/data/aidetect/app_tmp   # ONNX + int8 + tokenizer
python app/tools/verify_onnx.py --model $M                          # compare with the fp32 torch model
python app/tools/shard.py                                           # 48 MB shards + manifest.json
python app/tools/make_golden.py                                     # golden outputs for the browser test
```

`model/detector.onnx` (the unsplit file) is git-ignored; only the shards are committed.

## Test locally

```bash
python app/tools/serve.py 8765            # http://127.0.0.1:8765/ (adds the COOP/COEP headers)
PLAYWRIGHT_BROWSERS_PATH=... python app/tools/test_e2e.py     # headless Chromium: tokenizer parity, first load, analysis vs python, cached second visit
```

## Deploy

Copy `app/` (without `tools/`, with the shards in `model/`) into the public repository and push to `main`; its workflow publishes the site. Browsers: tested in Chromium (Chrome, Edge, Brave engine) only; Firefox and Safari are expected to work but could not be tested on this server.

## Speed

Headless Chromium on a 176-core server (8 WASM threads): about 0.5 s for a 130-token text, 4.8 s for 1,566 tokens (two windows). With one thread (no cross-origin isolation): 1.7 s and 27 s. Loading the cached model takes about 2 to 5 s.

## Limits

Texts over 30,000 characters are cut. The tokenizer does not treat special-token strings such as `<|endoftext|>` specially. The model's known weak spots (informal human text, paraphrasing tools) are in `reports/01_runs.html`; the score is not evidence about any person.
