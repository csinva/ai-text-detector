#!/usr/bin/env python3
"""Build reports/00_ai_text_detector.html from the run outputs under $V45_DATA_ROOT/data/aidetect/{runs,licensed}.

  python 09_report.py        (CPU only; renders the heatmap figures with the reference model on CPU)
"""
import html, importlib.util, json, os, subprocess, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
D = os.path.join(os.environ["V45_DATA_ROOT"], "data/aidetect"); R = f"{D}/runs"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "reports", "00_ai_text_detector.html")
spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "04_eval.py")); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}

REF, OOD, V1 = "lic2_nano", "lic2_nano_ood", "lic_nano"
MODELS = [("lic2_nano", "voyage-4-nano, 12 layers (reference)", 12), ("lic2_qwen_emb12", "Qwen3-Embedding-0.6B, first 12 layers", 12),
          ("lic2_qwen_base12", "Qwen3-0.6B-Base, first 12 layers", 12)]
V1_MODELS = [("lic_nano", "voyage-4-nano, 12 layers"), ("lic_qwen_emb12", "Qwen3-Embedding-0.6B, 12 layers"), ("lic_qwen_base12", "Qwen3-0.6B-Base, 12 layers"), ("lic_qwen_emb28", "Qwen3-Embedding-0.6B, 28 layers")]
res = {k: J(f"{R}/{k}/eval.json") for k, _, _ in MODELS}; res[OOD] = J(f"{R}/{OOD}/eval.json")
v1res = {k: {**J(f"{R}/{k}/eval.json"), **J(f"{R}/{k}/eval_gen.json")} for k, _ in V1_MODELS}
fin, ood = res[REF], res[OOD]
base = J(f"{R}/baselines_lic2.json"); cpu = J(f"{R}/{REF}/cpu_bench.json")
q8 = {**J(f"{R}/{REF}/eval_int8_cpu_words.json"), **J(f"{R}/{REF}/eval_int8_cpu.json")}
audit = J(f"{D}/licensed/license_audit.json")

SET_LABEL = {"mage_squad_test": "MAGE test, SQuAD domain", "raid_id": "RAID, seen generators/attacks", "raid_unseen_model": "RAID, unseen generators", "raid_unseen_domain": "RAID, unseen domain (abstracts)",
             "raid_unseen_attack": "RAID, unseen attacks", "raid_unseen_all": "RAID, generator + domain + attack unseen", "raid_attacks": "RAID, 12 attacks (balanced)",
             "textsight_2026": "TextSight 2026 (Claude Opus 5 etc.)", "wasit_2026": "WasItAIGenerated 2026-09 (humanizers, 2026 models)", "spans_test": "Heterogeneous AI Spans test (doc level)",
             "gen_id": "Generated text, held-out passages (6 generators)", "gen_unseen_gen": "Generated text, unseen generators (OLMo-2, Phi-4-mini)", "gen_unseen_domain": "Generated text, unseen registers (Hansard, Foodista)", "gen_mixed": "Mixed human + AI continuation (doc level)"}
DOC = list(SET_LABEL)
BASES = {"desklib": "desklib v1.01", "fakespot": "Fakespot RoBERTa-base", "openai-large": "OpenAI RoBERTa-large"}


def d(v, ref, digits=1):
    """value with the difference to the reference in parentheses (blue positive, red negative)."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return "n/a"
    s = f"{v:.{digits}f}"
    if ref is None or (isinstance(ref, float) and np.isnan(ref)): return s
    x = v - ref
    if abs(x) < 0.5 * 10 ** -digits: return f'{s} <span class=dd>({x:+.{digits}f})</span>'
    return f'{s} <span class="dd {"pos" if x > 0 else "neg"}">({x:+.{digits}f})</span>'


def get(r, s, m, kind="doc"):
    try: return r[s][kind][m]
    except Exception: return float("nan")


def mean_auc(r, sets=DOC):
    return float(np.nanmean([get(r, s, "auroc") for s in sets]))


def tab_models():
    cols = [(k, lab) for k, lab, _ in MODELS]
    h = "<tr><th>Evaluation set (AUROC)</th>" + "".join(f"<th>{lab}</th>" for _, lab in cols) + "</tr>"
    rows = ""
    for s in DOC:
        ref = get(fin, s, "auroc")
        rows += f"<tr><td>{SET_LABEL[s]}</td><td><b>{ref:.1f}</b></td>" + "".join(f"<td>{d(get(res[k], s, 'auroc'), ref)}</td>" for k, _ in cols[1:]) + "</tr>"
    rows += f"<tr><td><b>Mean over sets</b></td><td><b>{mean_auc(fin):.1f}</b></td>" + "".join(f"<td>{d(mean_auc(res[k]), mean_auc(fin))}</td>" for k, _ in cols[1:]) + "</tr>"
    for lab, s, m, kind in [("Spans test, word AUROC", "spans_test", "word_auroc", "word"), ("Spans test, document IoU", "spans_test", "mean_doc_iou", "word"), ("RAID splices, word AUROC", "raid_splice", "word_auroc", "word")]:
        ref = get(fin, s, m, kind)
        rows += f"<tr><td>{lab}</td><td><b>{ref:.1f}</b></td>" + "".join(f"<td>{d(get(res[k], s, m, kind), ref)}</td>" for k, _ in cols[1:]) + "</tr>"
    return f'<div class=wide><table id=tab-5><caption style="text-align:left"><b>Table 5.</b> Same recipe, same data (v2) and same 600M training tokens, three starting points. Difference to voyage-4-nano in parentheses.</caption>{h}{rows}</table></div>'


def tab_v1v2():
    h = "<tr><th>Evaluation set</th><th colspan=2>AUROC</th><th colspan=2>TPR at 1% FPR</th></tr><tr><th></th><th>v1 data</th><th>v2 data</th><th>v1 data</th><th>v2 data</th></tr>"
    rows = ""
    for s_ in DOC:
        a_, b_, c_, d_ = get(v1res["lic_nano"], s_, "auroc"), get(fin, s_, "auroc"), get(v1res["lic_nano"], s_, "tpr1"), get(fin, s_, "tpr1")
        rows += f"<tr><td>{SET_LABEL[s_]}</td><td>{'n/a' if np.isnan(a_) else format(a_, '.1f')}</td><td>{d(b_, a_)}</td><td>{'n/a' if np.isnan(c_) else format(c_, '.1f')}</td><td>{d(d_, c_)}</td></tr>"
    return f'<div class=wide><table id=tab-6b><caption style="text-align:left"><b>Table 6.</b> Effect of the larger licensed data on voyage-4-nano. v1 data = RAID (books, wiki, abstracts), MAGE-SQuAD and the Spans data, 110M tokens; v2 data adds 62k Common Pile human passages and their AI counterparts written by six Apache/MIT models. Same recipe, 400M tokens for v1 and 600M for v2. Difference to v1 in parentheses; n/a = not evaluated for v1.</caption>{h}{rows}</table></div>'


def tab_doc(metric, title, tid):
    cols = [(BASES[k], base.get(k, {})) for k in BASES if k in base]
    h = "<tr><th>Evaluation set</th><th>n (AI)</th><th>voyage-4-nano detector</th>" + "".join(f"<th>{c[0]}</th>" for c in cols) + "</tr>"
    rows = ""
    for s in DOC:
        ours = get(fin, s, metric); n = fin.get(s, {}).get("doc", {})
        rows += f"<tr><td>{SET_LABEL[s]}</td><td>{n.get('n', '')} ({n.get('n_ai', '')})</td><td><b>{ours:.1f}</b></td>" + "".join(f"<td>{d(get(r_, s, metric), ours)}</td>" for _, r_ in cols) + "</tr>"
    mean = lambda r_: float(np.nanmean([get(r_, s, metric) for s in DOC]))
    rows += f"<tr><td><b>Mean over sets</b></td><td></td><td><b>{mean(fin):.1f}</b></td>" + "".join(f"<td>{d(mean(r_), mean(fin))}</td>" for _, r_ in cols) + "</tr>"
    return f'<div class=wide><table id={tid}><caption style="text-align:left"><b>{title}</b></caption>{h}{rows}</table></div>'


def wasit_groups():
    """AUROC of human against each generator group on WasItAIGenerated, ours and the open detectors, and the benchmark's own ZeroGPT / tropa3 columns."""
    w = pd.read_parquet(f"{D}/prep_lic2/eval/wasit_2026.parquet")
    y = w.label.values
    cols = {"voyage-4-nano detector": fin.get("wasit_2026", {}).get("by_model", {})}
    for k, lab in BASES.items(): cols[lab] = base.get(k, {}).get("wasit_2026", {}).get("by_model", {})
    for lab, c in [("ZeroGPT (vendor column)", "zerogpt"), ("tropa3 (WasItAIGenerated's own detector)", "tropa3")]:
        p = pd.to_numeric(w[c], errors="coerce").values; ok = ~np.isnan(p); sub = {}
        for g in sorted(w.model.unique()):
            if g == "human": continue
            m = ok & ((w.model.values == g) | (y == 0))
            if (w.model.values[m] == g).sum() > 4: sub[g] = dict(auroc=100 * ev.roc_auc_score(y[m], p[m]))
        cols[lab] = sub
    groups = [g for g in sorted(w.model.unique()) if g != "human"]
    h = "<tr><th>AI text group (against all human texts)</th><th>n</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    rows = ""
    for g in groups:
        ours = cols["voyage-4-nano detector"].get(g, {}).get("auroc", float("nan"))
        rows += f"<tr><td>{g}</td><td>{int((w.model == g).sum())}</td><td><b>{ours:.1f}</b></td>" + "".join(f"<td>{d(c.get(g, {}).get('auroc', float('nan')), ours)}</td>" for n, c in list(cols.items())[1:]) + "</tr>"
    return f'<div class=wide><table id=tab-7><caption style="text-align:left"><b>Table 7.</b> WasItAIGenerated 2026-09 (English rows with CC BY 4.0 or MIT text, {int((y == 0).sum())} human texts): AUROC of each AI group against the human texts. Groups starting with humanized/ are AI text rewritten by a commercial humanizer; ai_paraphrase is AI text paraphrased by another model. Difference to the voyage-4-nano detector in parentheses.</caption>{h}{rows}</table></div>'


def tab_ood():
    sets = ["raid_id", "raid_unseen_model", "raid_unseen_domain", "raid_unseen_attack", "raid_unseen_all", "raid_attacks"]
    h = "<tr><th>RAID evaluation subset</th><th colspan=2>AUROC</th><th colspan=2>TPR at 1% FPR</th></tr><tr><th></th><th>held-out training</th><th>full training</th><th>held-out training</th><th>full training</th></tr>"
    rows = ""
    for s in sets:
        o1, f1, o2, f2 = get(ood, s, "auroc"), get(fin, s, "auroc"), get(ood, s, "tpr1"), get(fin, s, "tpr1")
        rows += f"<tr><td>{SET_LABEL[s]}</td><td>{o1:.1f}</td><td>{d(f1, o1)}</td><td>{o2:.1f}</td><td>{d(f2, o2)}</td></tr>"
    return f'<div class=wide><table id=tab-9><caption style="text-align:left"><b>Table 9.</b> Held-out RAID subsets: the model trained without three generators, one domain and three attacks, and the model trained on them (diff to the held-out model). Same documents; 8% of RAID source documents are never trained on in either.</caption>{h}{rows}</table></div>'


def tab_word():
    h = "<tr><th>Set</th><th>Model</th><th>Word AUROC</th><th>Word F1</th><th>Doc IoU</th></tr>"
    rows = ""
    for s, lab in [("spans_test", "Heterogeneous AI Spans test (Claude spans)"), ("raid_splice", "RAID splices (held-out documents)"), ("gen_mixed", "Mixed human prefix + AI continuation (6 generators)")]:
        w, wo = fin.get(s, {}).get("word", {}), ood.get(s, {}).get("word", {})
        for name, ww, ref in [("full", w, None), ("held-out", wo, w)]:
            if not ww: continue
            cell = lambda m: (f"{ww[m]:.1f}" if ref is None else d(ww[m], ref.get(m))) if m in ww else "n/a"
            rows += f"<tr><td>{lab if name == 'full' else ''}</td><td>{name}</td><td>{cell('word_auroc')}</td><td>{cell('word_f1')}</td><td>{cell('mean_doc_iou')}</td></tr>"
    return f'<div class=wide><table id=tab-10><caption style="text-align:left"><b>Table 10.</b> Word-level results (word = whitespace-delimited string; truth = more than half of its characters inside an AI span). "full" is the reference model, "held-out" the RAID-held-out model with its diff to full in parentheses. SemEval-8C and RoFT boundary tests were removed with their unlicensed data.</caption>{h}{rows}</table></div>'


def tab_cpu():
    if not cpu: return "<p>(CPU benchmark not run)</p>"
    df = pd.DataFrame(cpu)
    h = "<tr><th>Precision</th><th>Threads</th>" + "".join(f"<th>{n} tokens</th>" for n in (128, 512, 1024)) + "</tr>"
    rows = ""
    for (v, th), g in df.groupby(["variant", "threads"], sort=False):
        rows += f"<tr><td>{v}</td><td>{th}</td>" + "".join(f"<td>{g[g.tokens == n].sec_per_doc.iloc[0]:.2f} s ({g[g.tokens == n].tok_per_s.iloc[0]:.0f} tok/s)</td>" for n in (128, 512, 1024)) + "</tr>"
    return f'<div class=wide><table id=tab-11><caption style="text-align:left"><b>Table 11.</b> CPU latency per document with word-level output (Intel Xeon Platinum 8468V on a shared Slurm node, so timings vary by about 30% between runs; PyTorch SDPA, one document at a time). int8 = dynamic quantization of all linear layers except the MLP down projections and the two heads.</caption>{h}{rows}</table></div>'


def int8_table():
    if not q8: return "<p>(int8 evaluation not run)</p>"
    keep = [s for s in DOC if s in q8 and "doc" in q8[s]]
    h = "<tr><th>Set (sampled documents)</th><th>bf16 GPU AUROC</th><th>int8 CPU AUROC</th></tr>"
    rows = "".join(f"<tr><td>{SET_LABEL[s]}</td><td>{get(fin, s, 'auroc'):.1f}</td><td>{d(get(q8, s, 'auroc'), get(fin, s, 'auroc'))}</td></tr>" for s in keep)
    g, c = np.nanmean([get(fin, s, "auroc") for s in keep]), np.nanmean([get(q8, s, "auroc") for s in keep])
    rows += f"<tr><td><b>Mean</b></td><td><b>{g:.1f}</b></td><td><b>{d(c, g)}</b></td></tr>"
    w = ", ".join(f"{s} word AUROC {get(q8, s, 'word_auroc', 'word'):.1f} ({get(q8, s, 'word_auroc', 'word') - get(fin, s, 'word_auroc', 'word'):+.1f})" for s in ("spans_test", "raid_splice") if s in q8)
    return f'<div class=wide><table id=tab-12><caption style="text-align:left"><b>Table 12.</b> int8 CPU model against the bf16 GPU model (difference in parentheses; the GPU column uses all documents, the CPU column a sample, so differences include sampling noise). Word level: {w}.</caption>{h}{rows}</table></div>'


def figs(model_path):
    from infer import Predictor
    from heatmap import render_card
    P = Predictor(model_path, device="cpu", quant=0)
    ev_ = lambda n: pd.read_parquet(f"{D}/prep_lic2/eval/{n}.parquet")
    sp, rs, ts = ev_("spans_test"), ev_("raid_splice"), ev_("textsight_2026")
    CUT = 1400
    clip = lambda spans: [(a, min(b, CUT)) for a, b in spans if a < CUT]
    share = lambda spans: sum(b - a for a, b in clip(spans)) / CUT
    items = []
    sp = sp[sp.label == 1].assign(fr=[share(x) for x in sp[sp.label == 1].spans])
    sp = sp[(sp.fr > 0.2) & (sp.fr < 0.8) & (sp.text.str.len() > CUT)]
    for k, (m, g) in enumerate(sp.groupby("model")):
        for r in g.sample(min(2 if k == 0 else 1, len(g)), random_state=3).itertuples():
            items.append((f"Mixed: human-origin text with AI spans by {r.model}, {100 * r.fr:.0f}% AI", r.text[:CUT], clip(r.spans)))
    items = items[:4]
    rs = rs.assign(fr=[share(x) for x in rs.spans]); rs = rs[(rs.fr > 0.25) & (rs.fr < 0.75) & (rs.text.str.len() > CUT)]
    for dom, g in rs.groupby("domain"):
        r = g.sample(1, random_state=5).iloc[0]
        items.append((f"Mixed: sentence splice of human and {r.model} text, {r.domain}, {100 * r.fr:.0f}% AI", r.text[:CUT], clip(r.spans)))
    gm = ev_("gen_mixed"); gm = gm[gm.label == 1].assign(fr=[share(x) for x in gm[gm.label == 1].spans]); gm = gm[(gm.fr > 0.25) & (gm.fr < 0.75) & (gm.text.str.len() > 900)]
    for dom in ("stackexchange", "pubmed"):
        g_ = gm[gm.domain == dom]
        if len(g_):
            r = g_.sample(1, random_state=2).iloc[0]; items.append((f"Mixed: human {r.domain} passage continued by {r.model}, {100 * r.fr:.0f}% AI", r.text[:CUT], clip(r.spans)))
    ai = ts[ts.label == 1].sample(1, random_state=1).iloc[0]; items.append((f"Pure AI: TextSight 2026 passage by {ai.model}", ai.text, [(0, len(ai.text))]))
    hu = ts[ts.label == 0].sample(1, random_state=2).iloc[0]; items.append((f"Pure human: TextSight 2026 passage ({hu.model})", hu.text, []))
    preds = P.predict([t for _, t, _ in items])
    return "".join(render_card(t, pr, title=title, truth=sps) for (title, t, sps), pr in zip(items, preds))


def run_stats(run):
    p = f"{R}/{run}/train_log.jsonl"
    L = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    L = list({x["step"]: x for x in L if x.get("step", 0) > 10 and "mfu" in x}.values())  # a preempted run repeats the steps after its last checkpoint: keep the last record of each step
    if not L: return dict(steps=0, mfu=float("nan"), tok_s=float("nan"), tokens=0, secs=0)
    return dict(steps=len({x["step"] for x in L}) + 10, mfu=np.mean([x["mfu"] for x in L]) * 100, tok_s=np.mean([x["tok_s"] for x in L]), tokens=L[-1]["tokens"], secs=sum(x["step_s"] for x in L))


def sacct_hours(jobids):
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,JobName,Elapsed", "-j", ",".join(map(str, jobids))], capture_output=True, text=True).stdout
    rows = []
    for l in out.strip().splitlines():
        j, n, el = l.split("|")
        dd = int(el.split("-")[0]) if "-" in el else 0
        h, m, s = (int(x) for x in el.split("-")[-1].split(":"))
        sec = dd * 86400 + h * 3600 + m * 60 + s
        g = 0 if int(j.split("_")[0]) in CPU_ONLY else 1  # sacct does not record gres here: every job but the CPU jobs used exactly one GPU
        rows.append((j, n, sec / 3600, g, sec / 3600 * g / 8))
    return rows


CPU_ONLY = set()
LICENSED_JOBS = {}  # filled from jobs.json written by the launcher scripts, see main()


def audit_table():
    keep = [a for a in audit if a["decision"].startswith("keep")]; drop = [a for a in audit if a["decision"] == "drop"]
    row = lambda a: (f"<tr><td>{('<a href=' + repr(a['url']) + '>' + html.escape(a['name']) + '</a>') if a['url'] else html.escape(a['name'])}</td><td>{html.escape(a['declared'])}</td>"
                     f"<td>{a['decision']}</td><td>{html.escape(a['why'])}</td><td>{'' if a['kept'] is None else format(a['kept'], ',')}</td></tr>")
    h = "<tr><th>Source</th><th>Declared licence</th><th>Decision</th><th>Basis</th><th>Rows kept</th></tr>"
    return f'<div class=wide><table id=tab-1><caption style="text-align:left"><b>Table 1.</b> Licence audit of every dataset and model used or inspected. Kept first, then dropped.</caption>{h}{"".join(map(row, keep))}{"".join(map(row, drop))}</table></div>'


def corpus_table():
    srcs = json.load(open(f"{D}/tok_lic2_full/srcs.json"))
    off = np.load(f"{D}/tok_lic2_full/train_off.npy"); src = np.load(f"{D}/tok_lic2_full/train_src.npy"); lab = np.load(f"{D}/tok_lic2_full/train_lab.npy", mmap_mode="r")
    lens = np.diff(off)
    desc = {"raid": "RAID train, domains books (CMU book summaries, CC BY-SA), wiki (Wikipedia, CC BY-SA) and abstracts (arXiv, CC0), 10 of 11 generators, 12 attacks, balanced per domain",
            "raid_splice": "sentence-level splices of one RAID human text with an AI text of the same prompt (1 to 8 runs)", "mage_squad": "MAGE train and validation, SQuAD domain (CC BY-SA), permissive generators only, balanced",
            "spans": "Heterogeneous AI Spans train and validation without JMLR: public-domain and CC BY-SA human text with Claude and GPT spans"}
    for r_ in ("stackexchange", "arxiv_abstract", "pubmed", "gutenberg", "pre1929_books", "uk_hansard", "foodista"):
        desc["gen_" + r_] = f"Common Pile {r_} passages (human, 2017 or earlier, per-document open licence) and AI texts written from them by six Apache/MIT models (continuation, rewrite, write, casual), balanced"
    h = "<tr><th>Source</th><th>Documents</th><th>Tokens (M)</th><th>AI share of tokens (%)</th><th>Description</th></tr>"
    rows, tot_d, tot_t = "", 0, 0
    for i, sname in enumerate(srcs):
        m = src == i; toks = int(lens[m].sum()); idx = np.nonzero(m)[0]
        sel = np.random.RandomState(0).choice(idx, min(3000, len(idx)), replace=False)
        w = sum(float(lab[off[j]:off[j + 1]].astype(np.float32).sum()) / 255 for j in sel) / sum(int(lens[j]) for j in sel)
        rows += f"<tr><td>{sname}</td><td>{int(m.sum()):,}</td><td>{toks / 1e6:.1f}</td><td>{100 * w:.0f}</td><td>{desc.get(sname, '')}</td></tr>"
        tot_d += int(m.sum()); tot_t += toks
    rows += f"<tr><td><b>Total</b></td><td><b>{tot_d:,}</b></td><td><b>{tot_t / 1e6:.0f}</b></td><td></td><td>spans documents are sampled 4x as often (fewest documents, newest generators); 400M training tokens = {400e6 / tot_t:.1f} epochs</td></tr>"
    return f'<div class=wide><table id=tab-2><caption style="text-align:left"><b>Table 2.</b> Training mixture of the reference model (licensed data only). Tokens are the first 3,072 per document; training crops windows of up to 1,024.</caption>{h}{rows}</table></div>'


EVAL_ROWS = [("MAGE test, SQuAD domain", "https://huggingface.co/datasets/yaful/MAGE", "the MAGE test split restricted to the SQuAD domain and permissive generators"),
             ("RAID held-out groups", "https://huggingface.co/datasets/liamdugan/raid", "8% of RAID source documents with all their generations, never trained on; subsets with generators gpt4, mistral-chat, cohere, domain abstracts and attacks paraphrase, homoglyph, article deletion unseen (in the held-out model)"),
             ("TextSight 2026", "https://huggingface.co/datasets/textsightai/ai-text-detection-benchmark-2026", "Sept 2026: 644 human passages (public domain literature, pre-2022 Wikipedia, non-native-English PMC papers) and 56 passages from five current models"),
             ("WasItAIGenerated 2026-09", "https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09", "Sept 2026: English rows whose text is CC BY 4.0 or MIT: human texts, AI text from 2026 models (GPT-6, Claude Opus 5.5, Gemini 3.8 and others), paraphrased AI text and AI text rewritten by six commercial humanizers"),
             ("Heterogeneous AI Spans test", "https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans", "Oct 2026: mixed documents and human controls without JMLR text; exact character spans of Claude-written text; word and document level"),
             ("RAID splices", "https://huggingface.co/datasets/liamdugan/raid", "sentence splices of held-out human texts with held-out generators (word level)"),
             ("Generated text (gen_id, gen_unseen_gen, gen_unseen_domain, gen_mixed)", "https://huggingface.co/common-pile", "8% of the Common Pile passages (by hash) and their AI counterparts, never trained on: seen generators and registers, OLMo-2 and Phi-4-mini texts that the held-out model never saw, Hansard and Foodista registers that it never saw, and human-prefix + AI-continuation texts with exact spans (word level)")]


def eval_table():
    h = "<tr><th>Set</th><th>Level</th><th>Source and notes</th></tr>"
    rows = "".join(f"<tr><td><a href='{u}'>{n}</a></td><td>{'word' if ('splices' in n or 'gen_mixed' in n) else 'document' + (' + word' if 'Spans' in n else '')}</td><td>{d_}</td></tr>" for n, u, d_ in EVAL_ROWS)
    return f'<div class=wide><table id=tab-3><caption style="text-align:left"><b>Table 3.</b> Held-out evaluation sets (licensed data only). Training texts are removed if their normalized first 400 characters match any evaluation text.</caption>{h}{rows}</table></div>'


MFU_ROWS = [("flash-attn 2, 64k-token micro-batches", 37.5), ("FlashAttention-3, 32k-token micro-batches", 40.6), ("FlashAttention-3, 64k-token micro-batches (used)", 40.2),
            ("FlashAttention-3, 128k-token micro-batches", 40.6), ("FlashAttention-3, 64k, torch.compile max-autotune", 40.9)]


def mfu_table():
    h = "<tr><th>Configuration</th><th>MFU (%)</th></tr>"
    ref = MFU_ROWS[0][1]
    rows = "".join(f"<tr><td>{n}</td><td>{d(v, ref if i else None)}</td></tr>" for i, (n, v) in enumerate(MFU_ROWS))
    for k, lab, _ in MODELS:
        s = run_stats(k)
        rows += f"<tr><td><b>Training run: {lab}</b> (whole optimizer step, mean over {s['steps']:,} steps; {s['tok_s'] / 1e3:.0f}k tokens/s)</td><td><b>{d(s['mfu'], ref)}</b></td></tr>"
    return f'<div class=wide><table id=tab-4><caption style="text-align:left"><b>Table 4.</b> Model FLOPs utilization on one H100 (989 TFLOP/s dense bf16). First five rows: forward and backward of one micro-batch measured on nano with batches of an earlier data mix (same sequence-length statistics), before the licence audit. FLOPs per token = 6 x matmul parameters + 12 x layers x 2,048 x (sum of squared sequence lengths / tokens); the frozen embedding table and the loss are not counted. Difference to the first row in parentheses.</caption>{h}{rows}</table></div>'


CSS = """<style>.pos{color:var(--accent)} .neg{color:var(--bad)} .wide{overflow-x:auto} .dd{font-size:.72rem;color:var(--text-2);font-weight:400} .dd.pos{color:var(--accent)} .dd.neg{color:var(--bad)}
table{border-collapse:collapse;font-size:.85rem;margin:.6rem 0} th,td{border:1px solid var(--border);padding:.25rem .5rem;text-align:left;vertical-align:top} caption{margin-bottom:.3rem;font-size:.85rem;color:var(--text-2)}</style>"""
import heatmap
CSS = CSS.replace("</style>", heatmap.CSS + "</style>")


def main():
    from time import strftime
    jobs = J(f"{R}/jobs.json")
    global CPU_ONLY
    CPU_ONLY = set(jobs.get("cpu_only", []))
    ms = {k: run_stats(k) for k, _, _ in MODELS}
    best_b = max(base, key=lambda k: mean_auc({s: v for s, v in base[k].items()}), default=None) if base else None
    bm = {k: mean_auc(v) for k, v in base.items()}
    cpu8 = [x for x in cpu if x["variant"] == "int8" and x["threads"] == 8 and x["tokens"] == 512] if cpu else []
    cpu1 = [x for x in cpu if x["variant"] == "int8" and x["threads"] == 1 and x["tokens"] == 512] if cpu else []
    nh = sum(r[4] for r in sacct_hours(jobs.get("ref_train", [])))
    _off = np.load(f"{D}/tok_lic2_full/train_off.npy"); ndocs, ntok = len(_off) - 1, int(_off[-1])
    n_drop = sum(a["decision"] == "drop" for a in audit); n_keep = sum(a["decision"].startswith("keep") for a in audit)
    v1 = v1res["lic_nano"]
    top2 = [k for k in ("desklib", "fakespot") if k in base]
    cmp_ = lambda st: ", ".join(f"{BASES[k]} {get(base[k], st, 'auroc'):.1f}" for k in top2)
    summ = (f"Every dataset and model was audited for commercial use ({n_keep} kept, {n_drop} dropped, <a href='#tab-1'>Table 1</a>). Data without an explicit commercial-use licence, including most standard detection benchmarks (M4GT, SemEval-2024-8, HC3, RoFT, "
            f"and the Reddit, news, review, poem and recipe domains of MAGE and RAID), was removed, and everything was retrained on what remains. The licensed human text is 62k passages from Common Pile (Stack Exchange, arXiv abstracts, PubMed Central, Gutenberg and pre-1929 books, UK Hansard, Foodista; "
            f"dated 2017 or earlier where a date exists) plus RAID and MAGE-SQuAD text; AI text comes from RAID and from six Apache-2.0 or MIT models that I ran on those passages (continuations with exact spans, rewrites, new texts, casual texts): {ndocs:,} training documents, {ntok / 1e6:.0f}M tokens. "
            f"voyage-4-nano with a document head and a per-token head (word-level heatmap) reaches mean AUROC {mean_auc(fin):.1f} on {len(DOC)} held-out sets" +
            (f" against {max(bm.values()):.1f} for the best open detector run for comparison ({BASES[max(bm, key=bm.get)]})" if bm else "") +
            f". On the hard 2026 sets: WasItAIGenerated (humanizers, 2026 models, {int(fin['wasit_2026']['doc']['n'] - fin['wasit_2026']['doc']['n_ai'])} licensed human texts) AUROC {get(fin, 'wasit_2026', 'auroc'):.1f} (licensed v1 data only: {get(v1, 'wasit_2026', 'auroc'):.1f}; {cmp_('wasit_2026')}), "
            f"TextSight 2026 {get(fin, 'textsight_2026', 'auroc'):.1f} (v1 {get(v1, 'textsight_2026', 'auroc'):.1f}; {cmp_('textsight_2026')}). "
            f"Generated text from generators and registers the held-out model never saw: AUROC {get(ood, 'gen_unseen_gen', 'auroc'):.1f} and {get(ood, 'gen_unseen_domain', 'auroc'):.1f}. Word level on Claude-written spans: AUROC {get(fin, 'spans_test', 'word_auroc', 'word'):.2f}, document IoU {get(fin, 'spans_test', 'mean_doc_iou', 'word'):.1f}. "
            f"Other backbones with the same recipe and data: Qwen3-Embedding-0.6B (12 layers) mean AUROC {mean_auc(res['lic2_qwen_emb12']):.1f}, Qwen3-0.6B-Base (12 layers) {mean_auc(res['lic2_qwen_base12']):.1f}, against {mean_auc(fin):.1f} for voyage-4-nano (<a href='#tab-5'>Table 5</a>); on the smaller v1 data a 28-layer Qwen3-Embedding gave no gain. "
            f"Training takes {ms[REF]['secs'] / 3600:.2f} H100 hours at {ms[REF]['mfu']:.1f}% MFU. " +
            (f"A 512-token document takes {cpu8[0]['sec_per_doc']:.2f} s on 8 CPU cores and {cpu1[0]['sec_per_doc']:.2f} s on one core (int8). " if cpu8 and cpu1 else "") +
            f"Main limitation: AI text rewritten by commercial humanizers is still detected poorly (<a href='#tab-7'>Table 7</a>), and human text from informal registers beyond Stack Exchange is absent from the licensed data.")
    parts = []; A = parts.append
    A(f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>AI text detector</title><link rel=stylesheet href='theme.css'>{CSS}</head><body>")
    A("<nav class=topnav><a href='index.html'>&larr; All reports</a></nav>")
    A("<h1>00. AI-generated text detector with word-level heatmaps, from voyage-4-nano (licensed data only)</h1>")
    A(f"<p class=box><b>Summary.</b> {summ} Generated {strftime('%Y-%m-%d %H:%M')} UTC.</p>")
    A("<h2 id=sec-1>1. Design</h2><ul>"
      "<li><b>Model.</b> The shipped voyage-4-nano weights (Apache-2.0; Qwen3 layers, hidden size 1,024, 16 query and 8 key-value heads, 12 layers). The 2,048-dimension embedding projection is dropped. A linear token head maps every final hidden state to a logit for &ldquo;this token is AI-written&rdquo;; a linear document head acts on the mean of the hidden states. Attention is bidirectional, so a word is judged with the text before and after it.</li>"
      "<li><b>Labels.</b> Every training document has AI character spans: none for human text, the whole text for AI text, and exact spans for mixed documents. A token's target is the fraction of its characters inside AI spans. The document target is the mean token target of the cropped window, so for a mixed text the document score estimates the share of AI text. Loss = BCE on tokens + BCE on the document logit.</li>"
      "<li><b>Word heatmap.</b> A word's probability is the mean probability of the tokens ending inside it. The document score is the length-weighted mean of the document logits of 1,024-token windows (the last window is anchored to the end).</li>"
      "<li><b>Splices.</b> RAID contains a human text and several AI texts for the same prompt, so alternating runs of sentences from the two gives topic-matched mixed documents with exact word labels (96,276 in training). The Heterogeneous AI Spans data adds real model-written spans in human-origin text.</li></ul>")
    A("<h2 id=sec-2>2. Data and licences</h2>")
    A("<p><b>Rule.</b> Text is kept only if its rights trace to an explicit licence that allows commercial use (public domain, CC0, CC BY, CC BY-SA, MIT, Apache-2.0, Open Parliament Licence) or to a per-row licence of that kind. It is dropped when the licence is non-commercial, research-only, &ldquo;other&rdquo; or undeclared, and when it is scraped third-party text (news, reviews, forum posts, poems, recipes), even if the aggregate dataset card says MIT or Apache, because a card cannot grant rights to text its author does not hold. Machine-written text is dropped when the generator's weights are released under non-commercial or research-only terms (LLaMA-1, OPT, GLM-130B, MPT-30B-Chat). CC BY-SA text carries attribution and share-alike duties. OpenAI and Llama 2 terms forbid using their outputs to train competing generative models, not classifiers. This is an engineering audit from published terms, not legal advice.</p>")
    A(audit_table())
    A("<p>Datasets published in the last two months and used: Heterogeneous AI Spans (2026-10-05), TextSight 2026 (2026-09-30) and WasItAIGenerated 2026-09 (2026-09-25, per-row licences). Other recent sets were inspected and dropped for licence reasons (Table <a href='#tab-1'>1</a>).</p>")
    A(corpus_table()); A(eval_table())
    A("<p>Held-out design. 8% of RAID source documents (all their generations and attacks) are excluded from every model. The <i>held-out</i> model additionally never sees three generators (gpt4, mistral-chat, cohere), the abstracts domain and three attacks (paraphrase, homoglyph, article deletion); the reference model trains on those too. Everything else in <a href='#tab-3'>Table 3</a> is held out for all models.</p>")
    A("<h2 id=sec-3>3. Training and MFU</h2>")
    A(f"<p>One H100, bf16 autocast over fp32 master weights, fused AdamW (betas 0.9 / 0.98, weight decay 0.01, head lr 1e-3), cosine schedule with 3% warmup, lr 1.5e-4, 600M tokens (about 3.7 epochs of the 161M-token mixture; 262k tokens per optimizer step; the held-out model 450M), windows of up to 1,024 tokens with 25% of windows cut to a random length between 16 and 1,024, frozen embedding table. Hyperparameters (lr, bidirectional attention, frozen embeddings) were chosen in short runs on an earlier data mix that included non-commercial sources; those runs and their models were discarded and are not reported, and the final models were retrained from the original weights.</p>")
    A("<p>Documents are packed back to back into fixed 65,536-token micro-batches and attended with FlashAttention-3 varlen (no padding, static shapes so the non-attention parts of each layer compile with <code>torch.compile</code>, fused QKV and gate-up projections). The PyTorch profiler splits kernel time (approximately) into GEMMs 36%, FlashAttention 12%, fused elementwise kernels 22% and other 30%; micro-batch size and autotuning change MFU by under 1 point and FlashAttention-3 over flash-attn 2 gains 2.7 points.</p>")
    A(mfu_table())
    A("<h3 id=sec-3-1>3.1 Backbone comparison</h3><p>Qwen3-Embedding-0.6B and Qwen3-0.6B-Base have the layer shape of voyage-4-nano (hidden 1,024, 16 and 8 heads, MLP 3,072) but 28 layers. Taking the first 12 layers matches nano's compute (189M matmul parameters); the 28-layer run uses all 440M. All start from their published weights with the same bidirectional-attention fine-tuning, so the causal-trained Qwen3-Embedding must adapt to bidirectional attention.</p>")
    A(tab_models()); A(tab_v1v2())
    A("<h2 id=sec-4>4. Document-level results</h2>")
    A(f"<p>AUROC and true positive rate at 1% false positive rate are threshold free (the 1% threshold is set on the human texts of each set). At the fixed 0.5 threshold the reference model's false positive rate on human text is {get(fin, 'mage_squad_test', 'fpr'):.1f}% (MAGE-SQuAD), {get(fin, 'raid_id', 'fpr'):.1f}% (RAID), {get(fin, 'textsight_2026', 'fpr'):.1f}% (TextSight 2026) and {get(fin, 'wasit_2026', 'fpr'):.1f}% (WasItAIGenerated). The comparison detectors are black boxes with training data that is not fully documented: they may have seen some of these sources. Differences to the voyage-4-nano detector in parentheses.</p>")
    A(tab_doc("auroc", "Table 6. Document AUROC (0-100).", "tab-6")); A(wasit_groups()); A(tab_doc("tpr1", "Table 8. True positive rate (%) at 1% false positive rate.", "tab-8"))
    A("<p>Held-out generators, domain and attacks:</p>"); A(tab_ood())
    A("<h2 id=sec-5>5. Word-level results</h2>"); A(tab_word())
    A("<h3 id=fig-1>Figure 1. Word heatmaps on mixed and pure texts</h3><p>Each word is shaded by the mean P(AI) of its tokens (the token head). Dotted underline marks words inside a true AI span. The first four cards are real Claude-written spans inside human-origin text (Heterogeneous AI Spans test set), the next three are RAID splices that alternate sentences of a human text and an AI text written for the same prompt (held-out source documents), then two human Stack Exchange and PubMed passages continued by the open generators (gen_mixed), then one fully AI and one fully human passage from TextSight 2026. For a mixed text the document score estimates the share of AI text. The cards are fixed random draws, not selected for success: document scores track the true AI share on most cards and overshoot on some (a Cohere books splice with 72% AI scores 98%, a PubMed continuation with 44% AI scores 66%).</p>")
    A(figs(f"{R}/{REF}/model.pt"))
    A("<h2 id=sec-6>6. CPU inference</h2>"); A(tab_cpu()); A(int8_table())
    A("<p>Quantizing the MLP down projections as well (all linear layers) moved the document logit by +3.8 on average and lowered AUROC in a direct test on the earlier model (AUROC 99.98 to 99.25, false positive rate at 0.5 from 1.3% to 45%), probably because of outlier activations at the down projection input; keeping those layers and the heads in fp32 leaves the logit within 0.03 on average (correlation 0.999) (<code>aidetect/diag_quant.py</code>).</p>")
    A("<h2 id=sec-7>7. Caveats</h2><ul>"
      "<li>Licence screening narrows the human text to books, Wikipedia, arXiv abstracts, SQuAD passages, public-domain fiction, Federal Reserve and parliamentary text. Informal text (forums, reviews, student essays, social posts) is absent from training. WasItAIGenerated has human texts of that kind (essays, learner writing, speeches, documentation) and is the main out-of-domain check; look at its false positive rate before trusting a 0.5 threshold.</li>"
      "<li>Human text in the Heterogeneous AI Spans data is old and formal, so the model could partly learn modern prose against old prose there. The matched human controls limit this but do not remove it; word-level scores on that set are the easiest in this report.</li>"
      "<li>TextSight 2026 has 56 AI passages, so its TPR values move in steps of 1.8 points.</li>"
      "<li>MAGE-SQuAD, RAID and the Spans data are in-distribution sources for training (held-out documents); RAID unseen generators, domain and attacks, TextSight 2026 and WasItAIGenerated are the shifted checks. Several widely used benchmarks (SemEval-2024 task 8, HC3, RoFT, MAGE out-of-distribution sets) are no longer evaluated because their text is not commercially licensed.</li>"
      "<li>The comparison detectors are single checkpoints run as packaged, truncated at 512 tokens, and were not tuned on these sets. The licence audit covers data and model weights, not the legal status of model outputs beyond the terms quoted in Sec. <a href='#sec-2'>2</a>; have counsel review it before commercial release.</li></ul>")
    rows = sacct_hours(jobs.get("all", []))
    crow = "".join(f"<tr><td>{n}</td><td>{j}</td><td>{h_:.2f}</td><td>{g}</td><td>{nh_:.3f}</td></tr>" for j, n, h_, g, nh_ in rows)
    tot = sum(r[4] for r in rows)
    mrows = "".join(f"<tr><td>{lab}</td><td>{ms[k]['tokens'] / 1e6:.0f}</td><td>{ms[k]['secs'] / 3600:.2f}</td><td>{ms[k]['tok_s'] / 1e3:.0f}k</td><td>{ms[k]['mfu']:.1f}</td></tr>" for k, lab, _ in MODELS)
    A(f"<details><summary>Computational cost</summary><p>Everything ran on one H100 (80GB HBM3) at a time through Slurm partition P4 (jobs were preempted and resumed from checkpoints when needed). Evaluating all sets takes about {fin.get('_total_seconds', 0) / 60:.0f} minutes on one H100 for the reference model.</p>"
      f"<table><caption style='text-align:left'><b>Training throughput</b> (logged optimizer steps only)</caption><tr><th>Run</th><th>Tokens (M)</th><th>H100 hours</th><th>Tokens/s</th><th>MFU (%)</th></tr>{mrows}</table>"
      f"<table id=tab-13><caption style='text-align:left'><b>Table 13.</b> H100 node hours (elapsed hours x GPUs / 8) per Slurm job, including preempted and failed attempts (the 28-layer run ran out of memory once, two jobs were preempted and resumed); total {tot:.2f}. CPU jobs used no GPU.</caption><tr><th>Job</th><th>Slurm id</th><th>Elapsed (h)</th><th>GPUs</th><th>H100 node hours</th></tr>{crow}</table></details>")
    A(REPRO)
    open(OUT, "w").write("\n".join(parts))
    print("wrote", os.path.abspath(OUT), "chars", sum(map(len, parts)))


REPRO = """<details><summary>Reproducibility</summary>
<p>Code is in this repository (<code>aidetect/</code>); data root <code>/data18/voyage_4_5_experiments/data/aidetect</code>; python env <code>~/voyage-4.5/.venv-aid</code> (uv: torch 2.8.0, transformers 4.51.3, flash-attn 2.8.3, flash_attn_3 copied from the bxjg venv, tokenizers, pandas, scikit-learn).</p>
<pre>
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
python aidetect/00_download.py                    # licensed sources only: RAID (domains filtered later), MAGE, Heterogeneous AI Spans, TextSight, WasItAIGenerated
python aidetect/12_licensed_data.py               # licence filter -> data/aidetect/licensed/ and license_audit.json
python aidetect/01_prep.py; python aidetect/01_prep.py --full       # eval sets (identical), training mixtures (held-out / full)
python aidetect/02_tokenize.py; python aidetect/02_tokenize.py --full
NAME=lic_nano EXTRA="--tok tok_lic_full --lr 1.5e-4" TOKENS=400e6 sbatch --export=ALL --job-name=00-lic-nano aidetect/train_eval.slurm
NAME=lic_qwen_emb12 EXTRA="--tok tok_lic_full --lr 1.5e-4 --n_layers 12 --backbone .../models/Qwen3-Embedding-0.6B" EVALX="--n_layers 12" TOKENS=400e6 sbatch aidetect/train_eval.slurm   # likewise Qwen3-0.6B-Base, 28 layers
NAME=lic_nano_ood EXTRA="--tok tok_lic --lr 1.5e-4" TOKENS=300e6 sbatch aidetect/train_eval.slurm
CMD="aidetect/05_baselines.py --models desklib,fakespot,openai-large --out .../runs/baselines_lic.json" sbatch aidetect/run.slurm
MODEL=.../lic_nano/model.pt OUT=.../lic_nano/cpu_bench.json sbatch aidetect/cpu.slurm; MODEL=... OUT=.../lic_nano/eval_int8_cpu.json sbatch aidetect/cpu_eval.slurm
python aidetect/09_report.py                      # this page
python aidetect/predict.py --model .../lic_nano/model.pt --file doc.txt --html heat.html   # use the detector
</pre>
<p>Prompt for a coding agent: "In aidetect/ build an AI-text detector from voyage-4-nano on one H100 using only commercially licensed data: audit every dataset and model first (public domain, CC0, CC BY, CC BY-SA, MIT, Apache-2.0 only; drop non-commercial, research-only, undeclared and scraped third-party text, and generators with non-commercial weights), write the audit to license_audit.json, and keep only compliant rows (RAID books, wiki and abstracts without mpt-chat, MAGE SQuAD without OPT, LLaMA-1 and GLM-130B texts, Heterogeneous AI Spans without JMLR, TextSight 2026, WasItAIGenerated rows licensed CC BY 4.0 or MIT). Train a Qwen3-layer backbone with a token head and a document head (bidirectional attention, packed FlashAttention-3 training, bf16, torch.compile, lr 1.5e-4, 600M tokens, RAID sentence splices with span labels, plus 62k licensed Common Pile passages from 2017 or earlier and AI texts written from them by Apache-2.0 or MIT models with vLLM) from voyage-4-nano, Qwen3-Embedding-0.6B and Qwen3-0.6B-Base (12 layers each), evaluate document AUROC and TPR at 1% FPR and word-level AUROC and IoU, compare with desklib/ai-text-detector-v1.01, fakespot-ai/roberta-base-ai-text-detection-v1 and openai-community/roberta-large-openai-detector, benchmark CPU latency (fp32, bf16, int8) and report MFU. Regenerate reports/00_ai_text_detector.html with aidetect/09_report.py."</p></details></body></html>"""

if __name__ == "__main__":
    main()
