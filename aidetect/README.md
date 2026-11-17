# AI-generated text detector from voyage-4-nano

voyage-4-nano (Qwen3 layers, 12 layers, hidden 1,024) fine-tuned with a document head and a per-token head: P(AI-generated) for a text and a probability for every word (heatmap). Full write-up: `reports/00_ai_text_detector.html`.

## Use

```bash
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
M=$V45_DATA_ROOT/data/aidetect/runs/lic3_nano_r10/model.pt          # fp32 weights, 1.4 GB (the frozen embedding table is most of it)
~/voyage-4.5/.venv-aid/bin/python aidetect/predict.py --model $M --file doc.txt --html heat.html --json out.json            # CPU, fp32
~/voyage-4.5/.venv-aid/bin/python aidetect/predict.py --model $M --file doc.txt --quant 2 --html heat.html                  # CPU, int8 (about 2x faster, same scores)
```

```python
import sys; sys.path.insert(0, "aidetect")
from infer import Predictor
p = Predictor(M, device="cpu", quant=2)       # device="cuda" for GPU; quant: 0 fp32, 2 int8 (skips the MLP down projection and heads)
r = p.predict([text])[0]
r["doc_prob"]                                  # P(AI-generated) for the whole text
r["words"]                                     # [(start_char, end_char, P(AI)), ...] for each whitespace-delimited word
```

Texts longer than 1,024 tokens are scored in 1,024-token windows. Calibration differs between domains (see the report, Sec. 4); pick the threshold on human text from your domain rather than relying on 0.5.

## Licensing

Only commercially usable data and weights are used (audit table: `reports/00_ai_text_detector.html` Table 1; machine-readable: `licensed/license_audit.json` under the data root, written by `aidetect/12_licensed_data.py` and `aidetect/14_audit_extra.py`). Rule: keep text whose rights trace to an explicit commercial-use licence (public domain, CC0, CC BY, CC BY-SA, MIT, Apache-2.0, Open Parliament Licence, per-row licences); drop non-commercial, research-only, undeclared and scraped third-party text even if an aggregate card says MIT or Apache; drop machine text from generators with non-commercial weights. Kept: Common Pile subsets (Stack Exchange, arXiv abstracts, PubMed Central CC BY, Gutenberg, pre-1929 books, UK Hansard, Foodista; 2017 or earlier), RAID books / wiki / abstracts without MPT-30B-Chat text, MAGE SQuAD domain without OPT / LLaMA-1 / GLM-130B text, Heterogeneous AI Spans without JMLR, TextSight 2026, WasItAIGenerated rows licensed CC BY 4.0 or MIT, and AI text generated here with Qwen3, Mistral-7B-Instruct-v0.3, Phi-4-mini, OLMo-2 and SmolLM3 (Apache-2.0 / MIT). Dropped: M4GT / COLING-2025, SemEval-2024-8, HC3, RoFT, the Reddit / news / review / poem / recipe parts of RAID and MAGE, and more (see the table). Backbones: voyage-4-nano and Qwen3 (Apache-2.0). CC BY-SA text carries attribution and share-alike duties. This is an engineering audit, not legal advice.

## Files (run in this order)

| file | purpose |
|---|---|
| `00_download.py`, `12_licensed_data.py` | download the licensable sources, filter them into `$V45_DATA_ROOT/data/aidetect/licensed/` and write the licence audit |
| `13_*`, `14_human_corpus.py`, `14_generate.py`, `15_generate2.py`, `15_human_extra.py`, `gen.slurm`, `gen2.slurm`, `14_audit_extra.py` | search for openly licensed text, sample human passages from Common Pile, generate AI text with Apache/MIT models (vLLM), extend the audit |
| `01_prep.py [--full]` | training mixtures and eval sets from the licensed tree (`--full`: also train on RAID's held-out generators / domain / attacks) |
| `02_tokenize.py [--full]` | token ids, per-token soft labels |
| `03_train.py` | trainer: packed varlen FlashAttention-3, bf16 autocast, torch.compile, MFU logging, checkpoint/resume |
| `04_eval.py` | document (AUROC, TPR at 1% / 5% FPR) and word-level (AUROC, F1, IoU, boundary MAE) evaluation |
| `05_baselines.py` | open-source detectors on the same sets |
| `06_cpu_bench.py` | CPU latency |
| `09_report.py` | builds the report |
| `model.py`, `infer.py`, `predict.py` | model, windowed inference with word aggregation, explanations (integrated gradients, attention rollout), CLI |
| `heatmap.py` | HTML renderer (score meter, document map, shaded words) |
| `*.slurm` | one-GPU jobs (`train_eval.slurm`, `run.slurm`), CPU jobs (`cpu.slurm`, `cpu_eval.slurm`) |

Environment: `uv venv .venv-aid` with torch 2.8.0, transformers 4.51.3, flash-attn 2.8.3, tokenizers, pandas, pyarrow, scikit-learn. FlashAttention-3 (`flash_attn_interface`, `flash_attn_3`) is copied from the bxjg venv.
