"""HTML rendering of the detector's word-level AI probability.

render_card(...) returns a self-contained fragment (styles scoped to .hmx): score meter, document map and the text with every word
shaded by P(AI). render_page(...) wraps it in a page.
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
.hmx .maplab{font-size:.68rem;color:var(--muted);margin-bottom:1rem}
.hmx .txt{font-family:Charter,"Iowan Old Style","Palatino Linotype",Georgia,serif;font-size:1.02rem;line-height:2.05;white-space:pre-wrap;color:#292524}
.hmx .w{background:var(--c,transparent);border-radius:.38em;padding:.12em .08em;margin:0 -.08em;-webkit-box-decoration-break:clone;box-decoration-break:clone}
.hmx .w.t{text-decoration:underline dotted 2px #1c1917;text-underline-offset:.28em}
.hmx .leg{display:flex;align-items:center;gap:.6rem;margin-top:1rem;font-size:.7rem;color:var(--muted)}
.hmx .leg span.bar{flex:0 0 9rem;height:8px;border-radius:99px;background:linear-gradient(90deg,#fff 0%,rgba(251,191,36,.35) 35%,rgba(251,113,133,.6) 65%,rgba(225,29,72,.85) 100%);border:1px solid #f0efed}
"""

PROB = [(0.0, (251, 191, 36, 0.0)), (0.35, (251, 191, 36, 0.22)), (0.65, (251, 113, 133, 0.5)), (1.0, (225, 29, 72, 0.82))]


def c_prob(p):
    p = float(np.clip(p, 0, 1))
    for (a, ca), (b, cb) in zip(PROB[:-1], PROB[1:]):
        if p <= b:
            t = (p - a) / (b - a)
            c = tuple(ca[i] + t * (cb[i] - ca[i]) for i in range(4))
            return f"rgba({c[0]:.0f},{c[1]:.0f},{c[2]:.0f},{c[3]:.2f})"
    return "rgba(225,29,72,.82)"


def words_html(text, words, truth=None):
    """words: [(start, end, P(AI))]. A word is drawn together with the plain spaces after it so neighbouring highlights form a ribbon."""
    tm = np.zeros(len(text) + 1, dtype=np.int8)
    for s, e in (truth or []):
        tm[s:e] = 1
    out, pos = [], 0
    for s, e, v in words:
        gap = text[pos:s]
        if out and gap and gap.strip() == "" and "\n" not in gap:  # plain spaces belong to the previous word's chip
            out[-1] = out[-1][:-7] + html.escape(gap) + "</span>"
        else:
            out.append(html.escape(gap))
        cls = "w t" if truth and tm[s:e].mean() > 0.5 else "w"
        out.append(f'<span class="{cls}" style="--c:{c_prob(v)}" title="P(AI) = {100 * v:.0f}%">{html.escape(text[s:e])}</span>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def minimap(words, n=64):
    if not words:
        return ""
    v = np.array([w[2] for w in words]); k = min(n, len(v))
    return '<div class="map">' + "".join(f'<b style="background:{c_prob(x) if x > 0.04 else "#f5f5f4"}"></b>' for x in (c.mean() for c in np.array_split(v, k))) + "</div>"


def render_card(text, pred, title="", truth=None):
    """pred: Predictor.predict(...)[0] (needs 'words'); truth: [(start, end)] AI character spans or None (dotted underline)."""
    p = pred["doc_prob"]; words = pred["words"]
    mean_w = float(np.mean([w[2] for w in words])) if words else 0.0
    leg = '<div class="leg"><span>P(AI) 0%</span><span class="bar"></span><span>100%</span>' + ("<span>&nbsp;&middot; dotted underline = true AI text</span>" if truth else "") + "</div>"
    return (f'<div class="hmx"><div class="top"><div class="ttl">{html.escape(title)}</div><div class="score">{100 * p:.0f}%<small>AI-generated</small></div></div>'
            f'<div class="meter"><i style="left:{100 * p:.1f}%"></i></div><div class="ends"><span>human</span><span>AI</span></div>'
            f'{minimap(words)}<div class="maplab">Document map: {pred["n_tokens"]} tokens, mean word probability {100 * mean_w:.0f}%</div>'
            f'<div class="txt">{words_html(text, words, truth)}</div>{leg}</div>')


def render_page(text, pred, title="AI-generated text detector", truth=None):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>body{{background:#faf9f7;margin:0;padding:2.5rem 1rem}}{CSS}</style></head><body>'
            f'{render_card(text, pred, title, truth)}</body></html>')
