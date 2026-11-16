#!/usr/bin/env python3
"""Build reports/00_ai_text_detector.html from the run outputs under $V45_DATA_ROOT/data/aidetect/runs.

  python 09_report.py [--final final] [--ood final_ood]   (CPU only; renders the heatmap figures with the final model on CPU)
"""
import argparse, html, json, os, subprocess, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect")
R = f"{D}/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "reports", "00_ai_text_detector.html")

ap = argparse.ArgumentParser()
ap.add_argument("--final", default="final_v2"); ap.add_argument("--ood", default="final_ood_v2"); ap.add_argument("--v1", default="final")
a = ap.parse_args()
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
fin, ood, v1 = J(f"{R}/{a.final}/eval.json"), J(f"{R}/{a.ood}/eval.json"), J(f"{R}/{a.v1}/eval.json")
base = J(f"{R}/baselines.json")
cpu = J(f"{R}/{a.final}/cpu_bench.json")
q8 = {**J(f"{R}/{a.final}/eval_int8_cpu_words.json"), **J(f"{R}/{a.final}/eval_int8_cpu.json")}  # two CPU jobs: document sets (300 docs each) and long word-level sets (150 docs)
prep = J(f"{D}/prep_full/prep_stats.json")

SET_LABEL = {"mage_test": "MAGE test", "mage_ood_gpt": "MAGE OOD (GPT)", "mage_ood_gpt_para": "MAGE OOD (GPT, paraphrased)", "raid_id": "RAID, seen generators/domains/attacks",
             "raid_unseen_model": "RAID, unseen generators", "raid_unseen_domain": "RAID, unseen domains", "raid_unseen_attack": "RAID, unseen attacks",
             "raid_unseen_all": "RAID, all three unseen", "raid_attacks": "RAID, 12 attacks (balanced)", "semeval_a_test": "SemEval-2024 task 8A test (M4)", "hc3": "HC3 (ChatGPT answers)",
             "textsight_2026": "TextSight 2026 (Claude Opus 5 etc.)", "spans_test": "Heterogeneous AI Spans test (doc level)"}
DOC = list(SET_LABEL)
BASES = {"desklib": "desklib v1.01", "fakespot": "Fakespot RoBERTa-base", "hello": "Hello-SimpleAI RoBERTa", "openai-large": "OpenAI RoBERTa-large"}


def d(v, ref, digits=1):
    """value with the difference to the reference in parentheses (blue positive, red negative)."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return "n/a"
    s = f"{v:.{digits}f}"
    if ref is None or (isinstance(ref, float) and np.isnan(ref)): return s
    x = v - ref
    if abs(x) < 0.5 * 10 ** -digits: return f'{s} <span class=dd>({x:+.{digits}f})</span>'
    return f'{s} <span class="dd {"pos" if x > 0 else "neg"}">({x:+.{digits}f})</span>'


def get(res, s, m, kind="doc"):
    try: return res[s][kind][m]
    except Exception: return float("nan")


def tab_doc(metric, title, tid):
    cols = [("ours", fin, None)] + [(BASES[k], {s: v for s, v in base.get(k, {}).items()}, k) for k in BASES if k in base]
    h = "<tr><th>Evaluation set</th><th>n (AI)</th><th>This model</th>" + "".join(f"<th>{c[0]}</th>" for c in cols[1:]) + "</tr>"
    rows = []
    for s in DOC:
        ours = get(fin, s, metric)
        n = fin.get(s, {}).get("doc", {})
        r = f"<tr><td>{SET_LABEL[s]}</td><td>{n.get('n', '')} ({n.get('n_ai', '')})</td><td><b>{ours:.1f}</b></td>"
        for name, res, k in cols[1:]:
            r += f"<td>{d(get(res, s, metric), ours)}</td>"
        rows.append(r + "</tr>")
    mean = lambda res: float(np.nanmean([get(res, s, metric) for s in DOC]))
    r = f"<tr><td><b>Mean over sets</b></td><td></td><td><b>{mean(fin):.1f}</b></td>" + "".join(f"<td>{d(mean(c[1]), mean(fin))}</td>" for c in cols[1:]) + "</tr>"
    return f'<div class=wide><table id={tid}><caption style="text-align:left"><b>{title}</b></caption>{h}{"".join(rows)}{r}</table></div>'


def tab_ood():
    sets = ["raid_id", "raid_unseen_model", "raid_unseen_domain", "raid_unseen_attack", "raid_unseen_all", "raid_attacks"]
    h = "<tr><th>RAID evaluation subset</th><th colspan=2>AUROC</th><th colspan=2>TPR at 1% FPR</th></tr><tr><th></th><th>held-out training</th><th>full training</th><th>held-out training</th><th>full training</th></tr>"
    rows = ""
    for s in sets:
        o1, f1, o2, f2 = get(ood, s, "auroc"), get(fin, s, "auroc"), get(ood, s, "tpr1"), get(fin, s, "tpr1")
        rows += f"<tr><td>{SET_LABEL[s]}</td><td>{o1:.1f}</td><td>{d(f1, o1)}</td><td>{o2:.1f}</td><td>{d(f2, o2)}</td></tr>"
    return f'<div class=wide><table id=tab-7><caption style="text-align:left"><b>Table 7.</b> Held-out RAID subsets: the model trained without the three held-out generators, two domains and three attacks, and the model trained on them (diff to the held-out model). Same documents, 8% of RAID source documents are never trained on in either.</caption>{h}{rows}</table></div>'


def tab_v1():
    h = "<tr><th rowspan=2>Evaluation set</th><th colspan=2>AUROC</th><th colspan=2>False positive rate at P(AI) = 0.5 (%)</th><th colspan=2>True positive rate at 0.5 (%)</th></tr><tr><th>v1</th><th>v2</th><th>v1</th><th>v2</th><th>v1</th><th>v2</th></tr>"
    rows = ""
    for s_ in DOC:
        if s_ == "spans_test": continue
        rows += f"<tr><td>{SET_LABEL[s_]}</td><td>{get(v1, s_, 'auroc'):.1f}</td><td>{d(get(fin, s_, 'auroc'), get(v1, s_, 'auroc'))}</td><td>{get(v1, s_, 'fpr'):.1f}</td><td>{d(get(fin, s_, 'fpr'), get(v1, s_, 'fpr'))}</td><td>{get(v1, s_, 'recall'):.1f}</td><td>{d(get(fin, s_, 'recall'), get(v1, s_, 'recall'))}</td></tr>"
    return f'<div class=wide><table id=tab-8><caption style="text-align:left"><b>Table 8.</b> Effect of balancing human and AI texts inside each source domain (v2) against balancing only overall (v1). Same recipe and tokens otherwise. Difference to v1 in parentheses; for false positive rate lower is better.</caption>{h}{rows}</table></div>'


def word_rows(res):
    out = []
    for s, lab in [("spans_test", "Heterogeneous AI Spans test (Claude, mixed spans)"), ("raid_splice", "RAID splices (held-out documents; unseen generators only for the held-out model)"), ("semeval_c_test", "SemEval-2024 8C test (human then AI)"),
                   ("semeval_c_dev", "SemEval-2024 8C dev"), ("roft", "RoFT (human then GPT-2/3 family)")]:
        w = res.get(s, {}).get("word", {})
        out.append((s, lab, w))
    return out


def tab_word():
    h = "<tr><th>Set</th><th>Model</th><th>Word AUROC</th><th>Word F1</th><th>Doc IoU</th><th>Boundary MAE (words)</th><th>Median AE</th><th>Within 5 words (%)</th></tr>"
    rows = ""
    for (s, lab, w), (_, _, wo) in zip(word_rows(fin), word_rows(ood)):
        bnd = s in ("semeval_c_test", "semeval_c_dev", "roft")
        for name, ww, ref in [("full", w, None), ("held-out", wo, w)]:
            if not ww: continue
            cell = lambda m, dg=1: (f"{ww[m]:.{dg}f}" if ref is None else d(ww[m], ref.get(m), dg)) if m in ww else "n/a"
            rows += f"<tr><td>{lab if name == 'full' else ''}</td><td>{name}</td><td>{cell('word_auroc')}</td><td>{cell('word_f1')}</td><td>{cell('mean_doc_iou')}</td>"
            rows += (f"<td>{cell('boundary_mae_words')}</td><td>{cell('boundary_median_ae')}</td><td>{cell('boundary_within5')}</td></tr>" if bnd else "<td>n/a</td><td>n/a</td><td>n/a</td></tr>")
    return f'<div class=wide><table id=tab-9><caption style="text-align:left"><b>Table 9.</b> Word-level results (word = whitespace-delimited string; truth = more than half of its characters inside an AI span). "full" is the final model, "held-out" the RAID-held-out model with its diff to full in parentheses.</caption>{h}{rows}</table></div>'


def tab_cpu():
    if not cpu: return "<p>(CPU benchmark not run)</p>"
    df = pd.DataFrame(cpu)
    h = "<tr><th>Precision</th><th>Threads</th>" + "".join(f"<th>{n} tokens</th>" for n in (128, 512, 1024)) + "</tr>"
    rows = ""
    for (v, th), g in df.groupby(["variant", "threads"], sort=False):
        rows += f"<tr><td>{v}</td><td>{th}</td>" + "".join(f"<td>{g[g.tokens == n].sec_per_doc.iloc[0]:.2f} s ({g[g.tokens == n].tok_per_s.iloc[0]:.0f} tok/s)</td>" for n in (128, 512, 1024)) + "</tr>"
    return f'<div class=wide><table id=tab-10><caption style="text-align:left"><b>Table 10.</b> CPU latency per document with word-level output (Intel Xeon Platinum 8468V on a shared Slurm node, so timings vary by about 30% between runs; PyTorch SDPA, one document at a time). int8 = dynamic quantization of all linear layers except the MLP down projections and the two heads; quantizing those as well lowers accuracy (below).</caption>{h}{rows}</table></div>'


def heat(text, words, truth=None, maxc=1400):
    out, pos = [], 0
    tm = np.zeros(len(text) + 1, dtype=np.int8)
    for s, e in (truth or []): tm[s:e] = 1
    for s, e, p in words:
        out.append(html.escape(text[pos:s]))
        u = "border-bottom:2px solid #1b5fb0;" if truth and tm[s:e].mean() > 0.5 else ""
        out.append(f'<span style="background:rgba(214,60,40,{p:.2f});{u}" title="P(AI)={p:.2f}">{html.escape(text[s:e])}</span>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def figs(model_path):
    from infer import Predictor
    P = Predictor(model_path, device="cpu", quant=False)
    ev = lambda n: pd.read_parquet(f"{D}/prep/eval/{n}.parquet")
    sp = ev("spans_test"); ts = ev("textsight_2026")
    items = []
    rs = np.random.RandomState(4)
    fr = [sum(max(0, min(e, 1500) - s_) for s_, e in sps) / 1500 for sps in sp.spans]
    mixed = sp[(sp.label == 1) & (np.array(fr) > 0.25) & (np.array(fr) < 0.75)].sample(2, random_state=3)
    for i, r in enumerate(mixed.itertuples()):
        t = r.text[:1500]
        sps = [(s, min(e, 1500)) for s, e in r.spans if s < 1500]
        items.append((f"Mixed document from the Heterogeneous AI Spans test set (AI spans written by {r.model})", t, sps))
    ai = ts[(ts.label == 1)].sample(2, random_state=1)
    for r in ai.itertuples():
        items.append((f"TextSight 2026 AI passage ({r.model}), fully AI-written", r.text, [(0, len(r.text))]))
    hu = ts[(ts.label == 0)].sample(1, random_state=2)
    for r in hu.itertuples():
        items.append((f"TextSight 2026 human passage ({r.grp if r.grp != 'textsight_2026' else r.model})", r.text, []))
    from heatmap import render_card
    preds = P.predict([t for _, t, _ in items])
    html_ = ""
    for k, ((title, t, sps), pr) in enumerate(zip(items, preds)):
        html_ += render_card(t, pr, P.explain(t, steps=64), title=title, truth=sps, uid=f"fig1_{k}")
    return html_


def mfu_stats(run):
    L = [json.loads(l) for l in open(f"{R}/{run}/train_log.jsonl")]
    L = [x for x in L if x["step"] > 10 and "mfu" in x]
    return dict(steps=len({x["step"] for x in L}) + 10, mfu=np.mean([x["mfu"] for x in L]) * 100, tok_s=np.mean([x["tok_s"] for x in L]), tokens=L[-1]["tokens"], last=L[-1])


CPU_ONLY = {2582812, 2582813, 2582931, 2582932}


def sacct_hours(jobids):
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,JobName,Elapsed,AllocTRES", "-j", ",".join(map(str, jobids))], capture_output=True, text=True).stdout
    rows = []
    for l in out.strip().splitlines():
        j, n, el, tres = l.split("|")
        h, m, s = (int(x) for x in el.replace("-", ":").split(":")[-3:]); dd = int(el.split("-")[0]) if "-" in el else 0
        sec = dd * 86400 + h * 3600 + m * 60 + s
        g = 0 if int(j.split("_")[0]) in CPU_ONLY else 1  # sacct does not record gres here: every job but the CPU benchmark used exactly one GPU
        rows.append((j, n, sec / 3600, g, sec / 3600 * g / 8))
    return rows




def corpus_table():
    srcs = json.load(open(f"{D}/tok_full_bal/srcs.json"))
    off = np.load(f"{D}/tok_full_bal/train_off.npy"); src = np.load(f"{D}/tok_full_bal/train_src.npy"); lab = np.load(f"{D}/tok_full_bal/train_lab.npy", mmap_mode="r")
    lens = np.diff(off)
    desc = {"raid": ("RAID train (11 generators, 8 domains, 12 attacks), balanced human / AI", "https://huggingface.co/datasets/liamdugan/raid"),
            "raid_splice": ("sentence-level splices of one RAID human text with an AI text of the same prompt (1 to 8 runs)", "https://huggingface.co/datasets/liamdugan/raid"),
            "mage": ("MAGE train, balanced (27 generators, 10 domains)", "https://huggingface.co/datasets/yaful/MAGE"),
            "m4gt": ("M4GT-Bench English as packaged in COLING-2025 MGT, balanced", "https://huggingface.co/datasets/Jinyan1/COLING_2025_MGT_en"),
            "spans": ("Heterogeneous AI Spans train + validation (Oct 2026): human-origin text with Claude-written spans", "https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans"),
            "semeval_c": ("SemEval-2024 task 8 subtask C train (human prefix, machine suffix)", "https://huggingface.co/datasets/d0rj/SemEval2024-task8")}
    h = "<tr><th>Source</th><th>Documents</th><th>Tokens (M)</th><th>AI share of tokens (%)</th><th>Description</th></tr>"
    rows, tot_d, tot_t = "", 0, 0
    for i, sname in enumerate(srcs):
        m = src == i
        toks = int(lens[m].sum())
        idx = np.nonzero(m)[0]
        sel = np.random.RandomState(0).choice(idx, min(3000, len(idx)), replace=False)
        w = sum(float(lab[off[j]:off[j + 1]].astype(np.float32).sum()) / 255 for j in sel) / sum(int(lens[j]) for j in sel)
        rows += f"<tr><td>{sname}</td><td>{int(m.sum()):,}</td><td>{toks / 1e6:.1f}</td><td>{100 * w:.0f}</td><td><a href='{desc[sname][1]}'>{desc[sname][0]}</a></td></tr>"
        tot_d += int(m.sum()); tot_t += toks
    rows += f"<tr><td><b>Total</b></td><td><b>{tot_d:,}</b></td><td><b>{tot_t / 1e6:.0f}</b></td><td></td><td>spans documents are sampled 4x as often (fewest documents, newest generators)</td></tr>"
    return f'<div class=wide><table id=tab-1><caption style="text-align:left"><b>Table 1.</b> Training mixture (final model). Tokens are the first 3,072 per document; training crops windows of up to 1,024.</caption>{h}{rows}</table></div>'


EVAL_ROWS = [
    ("MAGE test, OOD GPT, OOD GPT paraphrased", "https://huggingface.co/datasets/yaful/MAGE", "MAGE test split (12,000 sampled) and its two out-of-distribution sets (newer GPT, and the same texts paraphrased)"),
    ("RAID held-out groups", "https://huggingface.co/datasets/liamdugan/raid", "8% of RAID source documents with all their generations, never trained on; subsets with generators gpt4, mistral-chat, cohere, domains recipes and reviews, attacks paraphrase, homoglyph and article deletion unseen (in the held-out model)"),
    ("SemEval-2024 task 8A test", "https://huggingface.co/datasets/d0rj/SemEval2024-task8", "M4 monolingual test (12,000 sampled); the training mix contains M4GT-Bench, so documents are unseen but the sources are not"),
    ("HC3", "https://huggingface.co/datasets/Hello-SimpleAI/HC3", "human vs ChatGPT answers, never in training (8,000 sampled, balanced)"),
    ("TextSight 2026", "https://huggingface.co/datasets/textsightai/ai-text-detection-benchmark-2026", "Sept 2026: 644 human passages (public domain literature, pre-2022 Wikipedia, non-native-English PMC papers) and 56 passages from five current models"),
    ("Heterogeneous AI Spans test", "https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans", "Oct 2026: 594 mixed documents and 594 human controls; exact character spans of Claude-written text; word-level and document-level"),
    ("RAID splices", "https://huggingface.co/datasets/liamdugan/raid", "2,903 splices of held-out human texts with held-out generators (word level)"),
    ("SemEval-2024 task 8C dev and test", "https://huggingface.co/datasets/d0rj/SemEval2024-task8", "505 and 11,123 documents, human prefix then machine suffix; boundary word index (word level)"),
    ("RoFT", "https://huggingface.co/datasets/liamdugan/roft", "4,000 human-prefix / machine-continuation texts from GPT-2, GPT-3 and CTRL families (word level)"),
]


def eval_table():
    h = "<tr><th>Set</th><th>Level</th><th>Source and notes</th></tr>"
    rows = "".join(f"<tr><td><a href='{u}'>{n}</a></td><td>{'word' if any(k in n for k in ('splices', '8C', 'RoFT')) else 'document' + (' + word' if 'Spans' in n else '')}</td><td>{d_}</td></tr>" for n, u, d_ in EVAL_ROWS)
    return f'<div class=wide><table id=tab-2><caption style="text-align:left"><b>Table 2.</b> Held-out evaluation sets. Training texts are removed if their normalized first 400 characters match any evaluation text.</caption>{h}{rows}</table></div>'


MFU_ROWS = [("flash-attn 2, 64k-token micro-batches", 37.5), ("FlashAttention-3, 32k-token micro-batches", 40.6), ("FlashAttention-3, 64k-token micro-batches (used)", 40.2),
            ("FlashAttention-3, 128k-token micro-batches", 40.6), ("FlashAttention-3, 64k, torch.compile max-autotune", 40.9)]


def mfu_table(ms):
    h = "<tr><th>Configuration (fwd + bwd of one micro-batch, bf16)</th><th>MFU (%)</th></tr>"
    ref = MFU_ROWS[0][1]
    rows = "".join(f"<tr><td>{n}</td><td>{d(v, ref if i else None)}</td></tr>" for i, (n, v) in enumerate(MFU_ROWS))
    rows += f"<tr><td><b>Final run, whole optimizer step (data loading, clipping, AdamW included), mean over {ms['steps']:,} steps</b></td><td><b>{d(ms['mfu'], ref)}</b></td></tr>"
    return f'<div class=wide><table id=tab-3><caption style="text-align:left"><b>Table 3.</b> Model FLOPs utilization on one H100 (989 TFLOP/s dense bf16). FLOPs per token = 6 x 188.7M matmul parameters + 12 x layers x 2,048 x (sum of squared sequence lengths / tokens) for bidirectional attention; the frozen 155M embedding table and the loss are not counted. Difference to the first row in parentheses.</caption>{h}{rows}</table></div>'


ABL = [("abl_base", "lr 3e-5 (reference)"), ("abl_causal", "lr 3e-5, causal attention"), ("abl_emb", "lr 3e-5, embeddings trained"), ("abl_lr1e4", "lr 1e-4"), ("abl_lr2e4", "lr 2e-4"), ("abl_lr4e4", "lr 4e-4")]


def abl_table():
    ev = {k: J(f"{R}/{k}/eval.json") for k, _ in ABL}
    m = lambda r: float(np.nanmean([get(r, s, "auroc") for s in DOC]))
    b = ev["abl_base"]
    ref = [m(b), get(b, "raid_unseen_domain", "auroc"), get(b, "spans_test", "word_auroc", "word"), get(b, "spans_test", "mean_doc_iou", "word"), get(b, "semeval_c_test", "boundary_mae_words", "word")]
    h = "<tr><th>Run (80M tokens)</th><th>Mean document AUROC, 13 sets</th><th>RAID unseen domains AUROC</th><th>Spans word AUROC</th><th>Spans doc IoU</th><th>SemEval 8C boundary MAE (words, lower is better)</th></tr>"
    rows = ""
    for k, lab in ABL:
        r = ev[k]
        vals = [m(r), get(r, "raid_unseen_domain", "auroc"), get(r, "spans_test", "word_auroc", "word"), get(r, "spans_test", "mean_doc_iou", "word"), get(r, "semeval_c_test", "boundary_mae_words", "word")]
        rows += f"<tr><td>{lab}</td>" + "".join(f"<td>{d(v, rf if k != 'abl_base' else None)}</td>" for v, rf in zip(vals, ref)) + "</tr>"
    return f'<div class=wide><table id=tab-4><caption style="text-align:left"><b>Table 4.</b> Ablations, 80M training tokens each on the held-out mixture, evaluated on at most 3,000 documents per set. Difference to the reference in parentheses.</caption>{h}{rows}</table></div>'


def int8_table():
    if not q8: return "<p>(int8 evaluation not run)</p>"
    h = "<tr><th>Set (300 documents sampled)</th><th>bf16 GPU AUROC</th><th>int8 CPU AUROC</th></tr>"
    rows = ""
    for s in DOC:
        if s in q8 and "doc" in q8[s]:
            rows += f"<tr><td>{SET_LABEL[s]}</td><td>{get(fin, s, 'auroc'):.1f}</td><td>{d(get(q8, s, 'auroc'), get(fin, s, 'auroc'))}</td></tr>"
    keep = [s for s in DOC if s in q8 and "doc" in q8[s]]
    g, c = np.nanmean([get(fin, s, "auroc") for s in keep]), np.nanmean([get(q8, s, "auroc") for s in keep])
    rows += f"<tr><td><b>Mean</b></td><td><b>{g:.1f}</b></td><td><b>{d(c, g)}</b></td></tr>"
    w = ", ".join(f"{s} word AUROC {get(q8, s, 'word_auroc', 'word'):.1f} ({get(q8, s, 'word_auroc', 'word') - get(fin, s, 'word_auroc', 'word'):+.1f})" for s in ("spans_test", "semeval_c_test") if s in q8)
    return f'<div class=wide><table id=tab-11><caption style="text-align:left"><b>Table 11.</b> int8 dynamic quantization (all linear layers except the MLP down projections and the two heads), evaluated on CPU, against the bf16 GPU model (difference in parentheses; the GPU column uses all documents, the CPU column a sample of 300 documents per set (150 for the word-level sets), so differences include sampling noise). Word level: {w}.</caption>{h}{rows}</table></div>'


CSS = """<style>.pos{color:var(--accent)} .neg{color:var(--bad)} .wide{overflow-x:auto} .dd{font-size:.72rem;color:var(--text-2);font-weight:400} .dd.pos{color:var(--accent)} .dd.neg{color:var(--bad)}
.hm{margin:1rem 0;border:1px solid var(--border);border-radius:8px;padding:.6rem .8rem} .hmt{font-size:.85rem;color:var(--text-2);margin-bottom:.4rem} .hmb{font-family:Georgia,serif;font-size:.92rem;line-height:1.75;white-space:pre-wrap}
table{border-collapse:collapse;font-size:.85rem;margin:.6rem 0} th,td{border:1px solid var(--border);padding:.25rem .5rem;text-align:left;vertical-align:top} caption{margin-bottom:.3rem;font-size:.85rem;color:var(--text-2)}
.legend span{display:inline-block;padding:0 .5rem;margin-right:.2rem;font-size:.8rem}</style>"""
import heatmap
CSS = CSS.replace("</style>", heatmap.CSS + "</style>")


def main():
    from time import strftime
    ms = mfu_stats(a.final)
    mean_auc = lambda res: float(np.nanmean([get(res, s, "auroc") for s in DOC]))
    bmean = {k: float(np.nanmean([get(base[k], s, "auroc") for s in DOC])) for k in base}
    best_b = max(bmean, key=bmean.get) if bmean else None
    cpu8 = [x for x in cpu if x["variant"] == "int8" and x["threads"] == 8 and x["tokens"] == 512] if cpu else []
    cpu1 = [x for x in cpu if x["variant"] == "int8" and x["threads"] == 1 and x["tokens"] == 512] if cpu else []
    nh_final = sum(r[4] for r in sacct_hours(JOB_FINAL))
    summ = (f"I fine-tuned voyage-4-nano (12 layers, 189M non-embedding parameters) into an AI-text detector with a document head and a per-token head that gives a word-level heatmap, "
            f"on one H100 in {nh_final:.2f} H100 node hours ({nh_final * 8:.1f} GPU hours, including evaluation) at {ms['mfu']:.1f}% MFU ({ms['tok_s'] / 1e3:.0f}k tokens/s). "
            f"On 13 held-out document sets the mean AUROC is {mean_auc(fin):.1f}" + (f" against {bmean[best_b]:.1f} for the best of the open detectors tried ({BASES[best_b]})" if best_b else "") +
            f"; on the 2026 Heterogeneous AI Spans test set it reaches word AUROC {get(fin, 'spans_test', 'word_auroc', 'word'):.2f} and document IoU {get(fin, 'spans_test', 'mean_doc_iou', 'word'):.1f}, "
            f"and on SemEval-2024 8C the boundary error is {get(fin, 'semeval_c_test', 'boundary_mae_words', 'word'):.1f} words. "
            + (f"A 512-token document takes {cpu8[0]['sec_per_doc']:.2f} s on 8 CPU cores and {cpu1[0]['sec_per_doc']:.2f} s on one core (int8). " if cpu8 and cpu1 else "")
            + f"Main weakness: at a fixed 0.5 threshold human text from some domains is over-flagged (false positive rate {get(fin, 'hc3', 'fpr'):.0f}% on HC3 answers, {get(fin, 'semeval_a_test', 'fpr'):.0f}% on SemEval 8A student essays, "
              f"{get(fin, 'mage_test', 'fpr'):.0f}% on MAGE test) although the ranking stays high, so the threshold should be set on human text from the target domain.")
    parts = []
    A = parts.append
    A(f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>AI text detector</title><link rel=stylesheet href='theme.css'>{CSS}</head><body>")
    A("<nav class=topnav><a href='index.html'>&larr; All reports</a></nav>")
    A("<h1>00. AI-generated text detector with word-level heatmaps, from voyage-4-nano</h1>")
    A(f"<p class=box><b>Summary.</b> {summ} Details: data in Sec. <a href='#sec-2'>2</a>, training and MFU in Sec. <a href='#sec-3'>3</a>, document results in Sec. <a href='#sec-4'>4</a>, "
      f"word-level results and heatmaps in Sec. <a href='#sec-5'>5</a>, CPU speed in Sec. <a href='#sec-6'>6</a>, caveats in Sec. <a href='#sec-7'>7</a>. Generated {strftime('%Y-%m-%d %H:%M')} UTC.</p>")
    A("<h2 id=sec-1>1. Design</h2><ul>"
      "<li><b>Model.</b> The shipped voyage-4-nano weights (<code>/data/new/prod_models/voyage-4-nano</code>, Qwen3 layers, hidden size 1,024, 16 query and 8 key-value heads, 12 layers). The 2,048-dimension embedding projection is dropped. A linear token head maps every final hidden state to a logit for &ldquo;this token is AI-written&rdquo;; a linear document head acts on the mean of the hidden states. Attention is bidirectional, so a word is judged with the text before and after it (Table <a href='#tab-4'>4</a> shows causal attention is worse for heatmaps).</li>"
      "<li><b>Labels.</b> Every training document has AI character spans: none for human text, the whole text for AI text, and exact spans for mixed documents. A token's target is the fraction of its characters inside AI spans. The document target is the mean token target of the cropped window, so for a mixed text the document score estimates the share of AI text (see Figure <a href='#fig-1'>1</a>). Loss = BCE on tokens + BCE on the document logit.</li>"
      "<li><b>Word heatmap.</b> A word's probability is the mean probability of the tokens ending inside it. The document score is the length-weighted mean of the document logits of 1,024-token windows (the last window is anchored to the end).</li>"
      "<li><b>Splices.</b> Open data with word-level truth is small. RAID contains a human text and several AI texts for the same prompt, so alternating runs of sentences from the two gives topic-matched mixed documents with exact labels (144,880 in training). The Heterogeneous AI Spans data (Oct 2026) adds real model-written spans in human-origin text.</li></ul>")
    A("<h2 id=sec-2>2. Data</h2>")
    A("<p>All data is open. Datasets published in the last two months were used where they exist: Heterogeneous AI Spans (2026-10-05) for word-level training and testing, and the TextSight 2026 control set (2026-09-30), which contains output of current frontier models and is hard for older detectors.</p>")
    A(corpus_table()); A(eval_table())
    A("<p>Held-out design. 8% of RAID source documents (all their generations and attacks) are excluded from both models. The <i>held-out</i> model additionally never sees three generators (gpt4, mistral-chat, cohere), two domains (recipes, reviews) and three attacks (paraphrase, homoglyph, article deletion), which gives the out-of-distribution RAID subsets; the <i>final</i> model trains on those too. Everything else in Table <a href='#tab-2'>2</a> is held out for both.</p>")
    A("<h2 id=sec-3>3. Training and MFU</h2>")
    A(f"<p>One H100, bf16 autocast over fp32 master weights, fused AdamW (betas 0.9 / 0.98, weight decay 0.01, head lr 1e-3), cosine schedule with 3% warmup, lr 1.5e-4, {ms['tokens'] / 1e9:.2f}B tokens (3,814 optimizer steps of 262k tokens; the run was preempted once and resumed from its step-3250 checkpoint with optimizer state), windows of up to 1,024 tokens with 25% of windows cut to a random length between 16 and 1,024. The embedding table is frozen (training it did not help, Table <a href='#tab-4'>4</a>).</p>")
    A("<p>Documents are packed back to back into fixed 65,536-token micro-batches and attended with FlashAttention-3 varlen (no padding tokens, static shapes so the non-attention parts of each layer compile with <code>torch.compile</code>, fused QKV and gate-up projections). The PyTorch profiler splits kernel time (approximately, since it also counts launch wrappers) into GEMMs 36%, FlashAttention 12%, fused elementwise kernels (norms, RoPE, SwiGLU, residuals) 22% and other 30%. Micro-batch size and autotuning change MFU by under 1 point, and FlashAttention-3 over flash-attn 2 gains 2.7 points.</p>")
    A(mfu_table(ms)); A(abl_table())
    A("<h2 id=sec-4>4. Document-level results</h2>")
    A(f"<p>AUROC and true positive rate at 1% false positive rate are threshold free. The 1% threshold is set on the human texts of each set, so it shows what a per-set threshold would give; with the fixed 0.5 threshold the false positive rate of the final model on human text is {get(fin, 'raid_id', 'fpr'):.1f}% (RAID seen), {get(fin, 'mage_test', 'fpr'):.1f}% (MAGE), {get(fin, 'hc3', 'fpr'):.1f}% (HC3), {get(fin, 'semeval_a_test', 'fpr'):.1f}% (SemEval 8A) and {get(fin, 'textsight_2026', 'fpr'):.1f}% (TextSight 2026). Baselines are ranked by their own AI probability and truncate at 512 tokens. Their training data is not fully documented: desklib scores 100.0 and 99.8 on SemEval 8A and HC3, which is consistent with, but does not show, training on those sources. Differences to this model in parentheses.</p>")
    A(tab_doc("auroc", "Table 5. Document AUROC (0-100).", "tab-5")); A(tab_doc("tpr1", "Table 6. True positive rate (%) at 1% false positive rate.", "tab-6"))
    A("<p>Held-out generators, domains and attacks:</p>"); A(tab_ood())
    A("<p>Source-domain balance matters for calibration. In the first version (v1) human and AI texts were balanced only over the whole M4GT and MAGE samples, so a domain with few human texts (OUTFOX student essays: 2,106 human against 37,986 AI essays) taught the model that essay style means AI. v2 balances inside every domain:</p>"); A(tab_v1())
    A("<h2 id=sec-5>5. Word-level results</h2>"); A(tab_word())
    A("<p>Boundary error uses the maximum likelihood split of the word probabilities into human then AI words, the setting of SemEval-2024 8C. RoFT texts are short (about 10 sentences) and human annotators score poorly on them, so low numbers there are expected.</p>")
    A("""<h3 id=sec-5-1>5.1 How the heatmaps are produced</h3><p>Each card in Figure <a href='#fig-1'>1</a> has three views of the same text. Words are whitespace-delimited strings; a word's value is taken over the tokens that end inside it.</p><ul>
<li><b>AI probability</b> is the supervised token head: a linear layer on every final hidden state, trained to output the share of the token's characters that lie in AI-written spans. It localizes AI-written text; it is an output of the classifier, not an explanation of the document score. The word value is the mean over its tokens.</li>
<li><b>Feature importance</b> is integrated gradients of the document logit with respect to the input embeddings: 64 midpoint steps (32 by default in predict.py) from a baseline text to the real text, gradient times embedding difference, summed over embedding dimensions and over the tokens of a word. Rose words raise the AI score, blue words lower it. The baseline is the end-of-text token at every position (a zero baseline converged poorly, completeness gaps of several logits, because RMSNorm amplifies near-zero inputs). That baseline reads as AI (+6.7 logits for a short test text), so values are relative to it and most words of human-looking text are blue. The attributions sum to the logit difference between the text and the baseline; the residual ("completeness gap") is printed under each view and was at most 0.07 logits in the examples.</li>
<li><b>Attention</b> is attention rollout: per layer, the head-averaged attention matrix is mixed with the identity (residual connection) and the 12 layers are multiplied; the value is the column mean times the number of tokens, so 1.0 means uniform. It shows where the representation draws its information from, not whether a word pushes toward AI or human, and it is not a causal explanation.</li></ul>
<p>The document logit is a linear function of the mean of the final hidden states, so for a mixed text it estimates the AI share and integrated gradients explain that estimate. Explanations cost one forward and backward pass per step (seconds on one CPU for a few hundred tokens), use the fp32 model, and cover the first 1,024 tokens. Only the probability view has ground truth (dotted underline); the importance and attention views are checked by completeness only.</p>""")
    A("<h3 id=fig-1>Figure 1. Word heatmaps (click the tabs)</h3>")
    A(figs(f"{R}/{a.final}/model.pt"))
    A("<h2 id=sec-6>6. CPU inference</h2>"); A(tab_cpu()); A(int8_table())
    A("<p>Quantizing the MLP down projections as well (all linear layers) was no faster in a direct test (0.37 against 0.39 s per document, 8 threads, 300 RAID documents; <code>aidetect/diag_quant.py</code>) and moved the document logit by +3.8 on average (AUROC 99.98 to 99.25, false positive rate at 0.5 from 1.3% to 45%), probably because of outlier activations at the down projection input. Keeping those layers and the heads in fp32 leaves the logit within 0.03 on average (correlation 0.999).</p>")
    A("<h2 id=sec-7>7. Caveats</h2><ul>"
      "<li>Human text in the Heterogeneous AI Spans data comes from public-domain books, old JMLR papers, Wikitext and Fed documents, so the model could partly learn modern prose against old prose there. The matched human controls (the same human text with and without AI spans) limit this but do not remove it. Word-level scores on that set are the easiest in this report.</li>"
      "<li>TextSight 2026 has 56 AI passages, so its TPR values move in steps of 1.8 points. It uses genre-matched short passages and is the closest check here to current frontier output.</li>"
      "<li>Calibration drifts across domains: a threshold fixed at 0.5 gives false positive rates that differ by source (Sec. <a href='#sec-4'>4</a>). Choose the threshold on human text from the target domain.</li>"
      "<li>The MAGE and SemEval sets share sources with training sets of the same benchmark (new documents, same distribution). RAID unseen generators, domains and attacks, HC3, TextSight and Heterogeneous AI Spans are the shifted checks.</li>"
      "<li>Baselines are single checkpoints run as packaged, with 512-token truncation; they were not tuned on these sets.</li></ul>")
    rows = sacct_hours(COST_JOBS)
    crow = "".join(f"<tr><td>{n}</td><td>{j}</td><td>{h_:.2f}</td><td>{g}</td><td>{nh:.3f}</td></tr>" for j, n, h_, g, nh in rows)
    tot = sum(r[4] for r in rows)
    try:
        docs_min = ms["tok_s"] * 60 / (np.load(f"{D}/tok_full_bal/train_off.npy")[-1] / len(np.load(f"{D}/tok_full_bal/train_src.npy")))
    except Exception:
        docs_min = None
    A(f"<details><summary>Computational cost</summary><p>Everything ran on one H100 (80GB HBM3) at a time through Slurm partition P4. Training throughput {ms['tok_s']:,.0f} tokens/s per H100 ({ms['mfu']:.1f}% MFU)"
      + (f", about {docs_min:,.0f} documents/minute at the mean tokenized document length" if docs_min else "") + f". Evaluating all sets takes about {fin.get('_total_seconds', 0) / 60:.0f} minutes on one H100.</p>"
      f"<table id=tab-12><caption style='text-align:left'><b>Table 12.</b> H100 node hours (elapsed hours x GPUs / 8) per job; total {tot:.2f}. The CPU job used no GPU.</caption><tr><th>Job</th><th>Slurm id</th><th>Elapsed (h)</th><th>GPUs</th><th>H100 node hours</th></tr>{crow}</table></details>")
    A(REPRO)
    open(OUT, "w").write("\n".join(parts))
    print("wrote", os.path.abspath(OUT), "chars", sum(map(len, parts)))


JOB_FINAL = [2582843, 2582925]  # preempted at step 3404, resumed from the step-3250 checkpoint
COST_JOBS = [2582501, 2582509, 2582504, 2582510, 2582515, 2582568, 2582595, 2582610, 2582746, 2582842, 2582843, 2582925, 2582844, 2582926, 2582857, 2582931]
REPRO = """<details><summary>Reproducibility</summary>
<p>Code is in <code>~/voyage-4.5/aidetect/</code> (branch worktree-ai-detector); data root <code>/data18/voyage_4_5_experiments/data/aidetect</code>; python env <code>~/voyage-4.5/.venv-aid</code> (uv: torch 2.8.0, transformers 4.51.3, flash-attn 2.8.3, flash_attn_3 copied from the bxjg venv, tokenizers, pandas, scikit-learn).</p>
<pre>
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
python aidetect/00_download.py                    # RAID, MAGE, RoFT, COLING-2025 en, SemEval-2024-8, HC3 (spans and textsight via snapshot_download, see Sec. 2)
python aidetect/01_prep.py; python aidetect/01_prep.py --full      # eval sets (identical), training mixtures (held-out / full)
python aidetect/02_tokenize.py; python aidetect/02_tokenize.py --full
sbatch --export=ALL --job-name=04-main aidetect/main.slurm         # trains final (tok_full) then final_ood (tok), 1B tokens each, then evaluates both
python aidetect/04_eval.py --model $V45_DATA_ROOT/data/aidetect/runs/final/model.pt --out .../final/eval.json
python aidetect/05_baselines.py --out .../baselines.json
MODEL=.../final/model.pt OUT=.../final/cpu_bench.json sbatch aidetect/cpu.slurm; MODEL=... OUT=.../final/eval_int8_cpu.json sbatch aidetect/cpu_eval.slurm
python aidetect/09_report.py                      # this page
python aidetect/predict.py --model .../final/model.pt --file doc.txt --html heat.html   # use the detector
</pre>
<p>Prompt for a coding agent: "In ~/voyage-4.5/aidetect build an AI-text detector from /data/new/prod_models/voyage-4-nano on one H100: Qwen3 backbone with bidirectional attention, a token head and a document head, trained on RAID, MAGE, M4GT (COLING-2025), open-text-detector/heterogeneous-ai-spans, SemEval-2024 8C train and RAID sentence splices with span labels (decontaminate against all eval texts), packed varlen FlashAttention-3 training with bf16 autocast, torch.compile, frozen embeddings, lr 1.5e-4, 1B tokens. Evaluate on MAGE test and OOD, RAID held-out groups, SemEval 8A, HC3, TextSight 2026, Heterogeneous AI Spans test (document AUROC, TPR at 1% FPR; word AUROC, IoU) and SemEval 8C / RoFT (boundary MAE), compare with desklib/ai-text-detector-v1.01 and three RoBERTa detectors, benchmark CPU latency (fp32, bf16, int8) and report MFU. Regenerate reports/00_ai_text_detector.html with aidetect/09_report.py."</p></details></body></html>"""

if __name__ == "__main__":
    main()
