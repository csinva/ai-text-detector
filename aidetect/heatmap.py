"""HTML rendering of the detector's word-level views.

Views (tabs, CSS only):
  AI probability      the supervised token head, averaged over each word: where is the text AI-written?
  Feature importance  integrated gradients of the document logit: which words pushed the classifier toward AI (rose) or human (blue)?
  Attention           attention rollout: which words the classifier's representation draws on most.

render_card(...) returns a self-contained fragment (styles scoped to .hmx); render_page(...) wraps it in a page.
"""
from __future__ import annotations
import html
import numpy as np

CSS = """
.hmx{--ink:#1c1917;--muted:#78716c;--line:#e7e5e4;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);background:#fff;
  border:1px solid var(--line);border-radius:18px;box-shadow:0 1px 2px rgba(28,25,23,.04),0 12px 32px rgba(28,25,23,.06);padding:1.3rem 1.5rem 1.1rem;margin:1.4rem 0;max-width:46rem}
.hmx *{box-sizing:border-box}
.hmx .top{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap}
.hmx .ttl{font-size:.74rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600;max-width:30rem;line-height:1.4}
.hmx .score{font-size:2.3rem;font-weight:700;letter-spacing:-.03em;line-height:1}
.hmx .score small{font-size:.8rem;font-weight:500;color:var(--muted);letter-spacing:0;margin-left:.35rem}
.hmx .meter{position:relative;height:8px;border-radius:99px;margin:.8rem 0 .25rem;background:linear-gradient(90deg,#34d399 0%,#fde68a 50%,#fb7185 100%)}
.hmx .meter i{position:absolute;top:-4px;width:16px;height:16px;margin-left:-8px;border-radius:50%;background:#fff;border:3px solid var(--ink);box-shadow:0 1px 4px rgba(0,0,0,.25)}
.hmx .ends{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted)}
.hmx .map{display:flex;gap:2px;height:14px;margin:1rem 0 .2rem}
.hmx .map b{flex:1;border-radius:3px;background:#f5f5f4}
.hmx .maplab{font-size:.68rem;color:var(--muted);margin-bottom:.9rem}
.hmx input{display:none}
.hmx .tabs{display:inline-flex;background:#f5f5f4;border-radius:99px;padding:3px;margin-bottom:.9rem}
.hmx .tabs label{font-size:.78rem;font-weight:600;color:var(--muted);padding:.32rem .85rem;border-radius:99px;cursor:pointer;transition:all .15s}
.hmx .tabs label:hover{color:var(--ink)}
.hmx .view{display:none}
.hmx .txt{font-family:Charter,"Iowan Old Style","Palatino Linotype",Georgia,serif;font-size:1.02rem;line-height:2.05;white-space:pre-wrap;color:#292524}
.hmx .w{background:var(--c,transparent);border-radius:.38em;padding:.12em .08em;margin:0 -.08em;-webkit-box-decoration-break:clone;box-decoration-break:clone}
.hmx .w.t{text-decoration:underline dotted 2px #1c1917;text-underline-offset:.28em}
.hmx .leg{display:flex;align-items:center;gap:.6rem;margin-top:1rem;font-size:.7rem;color:var(--muted)}
.hmx .leg span.bar{flex:0 0 9rem;height:8px;border-radius:99px}
.hmx .note{font-size:.72rem;color:var(--muted);margin-top:.5rem;line-height:1.45}
"""

TAB_CSS = """
.hmx input.r{r}:checked ~ .views .v{r}{display:block}
.hmx input.r{r}:checked ~ .tabs label.l{r}{background:#fff;color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.12)}
"""
CSS += "".join(TAB_CSS.replace("{r}", str(i)) for i in (1, 2, 3))

# colour stops: (value, (r, g, b, a))
PROB = [(0.0, (251, 191, 36, 0.0)), (0.35, (251, 191, 36, 0.22)), (0.65, (251, 113, 133, 0.5)), (1.0, (225, 29, 72, 0.82))]
BAR_PROB = "linear-gradient(90deg,#fff 0%,rgba(251,191,36,.35) 35%,rgba(251,113,133,.6) 65%,rgba(225,29,72,.85) 100%)"
BAR_DIV = "linear-gradient(90deg,rgba(37,99,235,.8),rgba(147,197,253,.35) 35%,#fff 50%,rgba(253,164,175,.45) 65%,rgba(225,29,72,.85))"
BAR_ATT = "linear-gradient(90deg,#fff,rgba(167,139,250,.5) 50%,rgba(109,40,217,.88))"


def lerp(stops, v):
    v = float(np.clip(v, 0, 1))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return tuple(ca[i] + t * (cb[i] - ca[i]) for i in range(4))
    return stops[-1][1]


def rgba(c):
    return f"rgba({c[0]:.0f},{c[1]:.0f},{c[2]:.0f},{c[3]:.2f})"


def c_prob(p, **_):
    return rgba(lerp(PROB, p))


def make_div(vals):
    s = max(np.percentile(np.abs(vals), 97), 1e-6)

    def f(v):
        t = float(np.clip(v / s, -1, 1)); a = 0.85 * abs(t) ** 0.8
        return rgba((225, 29, 72, a)) if t > 0 else rgba((37, 99, 235, a))
    return f


def make_seq(vals, rgb=(109, 40, 217)):
    s = max(np.percentile(vals, 98), 1e-6)
    return lambda v: rgba((*rgb, 0.85 * float(np.clip(v / s, 0, 1)) ** 0.8))


def words_html(text, words, color, truth=None, tip=lambda v: f"{v:.2f}"):
    """words: [(start, end, value)]. A word is drawn together with the plain spaces after it so neighbouring highlights form a ribbon."""
    tm = np.zeros(len(text) + 1, dtype=np.int8)
    for s, e in (truth or []):
        tm[s:e] = 1
    out, pos = [], 0
    for i, (s, e, v) in enumerate(words):
        gap = text[pos:s]
        if out and gap and gap.strip() == "" and "\n" not in gap:  # plain spaces: belong to the previous word's chip
            out[-1] = out[-1][:-7] + html.escape(gap) + "</span>"
        else:
            out.append(html.escape(gap))
        cls = "w t" if truth and tm[s:e].mean() > 0.5 else "w"
        out.append(f'<span class="{cls}" style="--c:{color(v)}" title="{tip(v)}">{html.escape(text[s:e])}</span>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def minimap(words, n=64):
    if not words:
        return ""
    v = np.array([w[2] for w in words]); k = min(n, len(v))
    segs = [c.mean() for c in np.array_split(v, k)]
    return '<div class="map">' + "".join(f'<b style="background:{c_prob(x) if x > 0.04 else "#f5f5f4"}"></b>' for x in segs) + "</div>"


def render_card(text, pred, expl=None, title="", truth=None, uid="a"):
    """pred: Predictor.predict(...)[0] (needs 'words'); expl: Predictor.explain(...) or None; truth: [(start, end)] AI spans or None."""
    p = pred["doc_prob"]; words = pred["words"]
    mean_w = float(np.mean([w[2] for w in words])) if words else 0.0
    views = []
    views.append(("AI probability", words_html(text, words, c_prob, truth, lambda v: f"P(AI) = {100 * v:.0f}%"), BAR_PROB, "0%", "100%",
                  "Supervised per-token head, averaged over each word: where in the text the detector believes AI wrote."))
    if expl:
        iv = np.array([w[2] for w in expl["ig"]])
        views.append(("Feature importance", words_html(text, expl["ig"], make_div(iv), truth, lambda v: f"IG = {v:+.3f} logits"), BAR_DIV, "toward human", "toward AI",
                      f"Integrated gradients of the document logit ({expl['n_tokens']} tokens, baseline = end-of-text token, completeness gap {expl['ig_gap']:+.2f}). "
                      "Rose words raise the AI score, blue words lower it."))
        av = np.array([w[2] for w in expl["attn"]])
        views.append(("Attention", words_html(text, expl["attn"], make_seq(av), truth, lambda v: f"{v:.2f}x uniform attention"), BAR_ATT, "less", "more",
                      "Attention rollout over the 12 layers (head-averaged, residual connections included): how strongly each word feeds the final representation, 1.0 = uniform."))
    radios = "".join(f'<input type="radio" class="r r{i + 1}" name="{uid}" id="{uid}{i}"{" checked" if i == 0 else ""}>' for i in range(len(views)))
    tabs = '<div class="tabs">' + "".join(f'<label class="l{i + 1}" for="{uid}{i}">{html.escape(v[0])}</label>' for i, v in enumerate(views)) + "</div>"
    body = "".join(
        f'<div class="view v{i + 1}"><div class="txt">{h}</div><div class="leg"><span>{lo}</span><span class="bar" style="background:{bar}"></span><span>{hi}</span>'
        f'{"<span>&nbsp;&middot; dotted underline = true AI text</span>" if truth else ""}</div><div class="note">{html.escape(note)}</div></div>'
        for i, (_, h, bar, lo, hi, note) in enumerate(views))
    return (f'<div class="hmx"><div class="top"><div class="ttl">{html.escape(title)}</div>'
            f'<div class="score">{100 * p:.0f}%<small>AI-generated</small></div></div>'
            f'<div class="meter"><i style="left:{100 * p:.1f}%"></i></div><div class="ends"><span>human</span><span>AI</span></div>'
            f'{minimap(words)}<div class="maplab">Document map: {pred["n_tokens"]} tokens, mean word probability {100 * mean_w:.0f}%</div>'
            f'{radios}{tabs}<div class="views">{body}</div></div>')


def render_page(text, pred, expl=None, title="AI-generated text detector"):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>body{{background:#faf9f7;margin:0;padding:2.5rem 1rem}}{CSS}</style></head><body>'
            f'{render_card(text, pred, expl, title)}</body></html>')
