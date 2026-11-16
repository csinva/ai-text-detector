# AI-generated text detector from voyage-4-nano

voyage-4-nano (Qwen3 layers, 12 layers, hidden 1,024) fine-tuned with a document head and a per-token head: P(AI-generated) for a text and a probability for every word (heatmap). Full write-up: `reports/04_ai_text_detector.html`.

## Use

```bash
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
M=$V45_DATA_ROOT/data/aidetect/runs/final_v2/model.pt          # fp32 weights, 1.4 GB (the frozen embedding table is most of it)
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

## Files (run in this order)

| file | purpose |
|---|---|
| `00_download.py` | RAID, MAGE, RoFT, COLING-2025 (M4GT), SemEval-2024-8, HC3 into `$V45_DATA_ROOT/data/aidetect/raw` |
| `01_prep.py [--full] [--bal]` | training mixtures and eval sets (`--full`: also train on RAID's held-out generators/domains/attacks; `--bal`: balance human/AI inside each source domain) |
| `02_tokenize.py [--full] [--bal]` | token ids, per-token soft labels |
| `03_train.py` | trainer: packed varlen FlashAttention-3, bf16 autocast, torch.compile, MFU logging, checkpoint/resume |
| `04_eval.py` | document (AUROC, TPR at 1% / 5% FPR) and word-level (AUROC, F1, IoU, boundary MAE) evaluation |
| `05_baselines.py` | open-source detectors on the same sets |
| `06_cpu_bench.py` | CPU latency |
| `09_report.py` | builds the report |
| `model.py`, `infer.py`, `predict.py` | model, windowed inference with word aggregation, CLI |
| `*.slurm` | one-GPU jobs (`train_eval.slurm`, `run.slurm`), CPU jobs (`cpu.slurm`, `cpu_eval.slurm`) |

Environment: `uv venv .venv-aid` with torch 2.8.0, transformers 4.51.3, flash-attn 2.8.3, tokenizers, pandas, pyarrow, scikit-learn. FlashAttention-3 (`flash_attn_interface`, `flash_attn_3`) is copied from the bxjg venv.
