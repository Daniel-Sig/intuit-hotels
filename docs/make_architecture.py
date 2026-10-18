"""Generate docs/architecture.svg (hand-laid-out, on a grid): python docs/make_architecture.py docs/architecture.svg
The PDF and PNG are that SVG rendered with headless Chrome (--print-to-pdf / --screenshot at 2x, cropped to 3000x1500)."""
import sys
from xml.sax.saxutils import escape

INK = "#1b2330"
MUTED = "#5a6677"
RED = "#a63a3a"
AMBER = "#a8650a"
KINDS = {  # fill, stroke
    "code": ("#e6eef9", "#2f5d9e"),
    "llm": ("#fbeed6", AMBER),
    "rank": ("#dcf1ee", "#12776e"),
    "store": ("#e9edf3", "#44505f"),
    "stop": ("#f7e3e3", RED),
    "plain": ("#ffffff", "#8a94a3"),
}
BADGE = {"llm": ("LLM", AMBER, "#fbeed6"), "code": ("code", "#2f5d9e", "#e6eef9"), "rank": ("ranker", "#12776e", "#dcf1ee")}

out = []


def add(s):
    out.append(s)


def text(x, y, s, size=12, weight="normal", fill=INK, anchor="start", halo=False):
    attrs = (f'x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
             f'font-family="Segoe UI, Arial, sans-serif"')
    if halo:
        attrs += ' stroke="#ffffff" stroke-width="4" paint-order="stroke" stroke-linejoin="round"'
    add(f"<text {attrs}>{escape(s)}</text>")


def box(x, y, w, h, kind, title, lines=(), badges=(), rx=8):
    fill, stroke = KINDS[kind]
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    text(x + 12, y + 23, title, 13.5, "bold")
    for i, line in enumerate(lines):
        text(x + 12, y + 44 + 16 * i, line, 12)
    cx = x + w - 8
    for b in reversed(badges):
        label, color, bg = BADGE[b]
        bw_ = 8 + 6.4 * len(label)
        cx -= bw_
        add(f'<rect x="{cx:.1f}" y="{y + 8}" width="{bw_:.1f}" height="16" rx="3" fill="{bg}" stroke="{color}" stroke-width="1"/>')
        text(cx + bw_ / 2, y + 20, label, 10.5, "bold", color, "middle")
        cx -= 4


def arrow(points, color=INK, dashed=False, both=False, width=1.8):
    d = "M" + " L".join(f"{px} {py}" for px, py in points)
    marker = {INK: "ink", AMBER: "amber", RED: "red", "#44505f": "grey"}[color]
    extra = ' stroke-dasharray="6 4"' if dashed else ""
    start = f' marker-start="url(#a-{marker})"' if both else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra} marker-end="url(#a-{marker})"{start}/>')


def panel(x, y, w, h, label, sub=""):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#f6f7f9" stroke="#c9d0da" stroke-width="1.2"/>')
    text(x + 18, y + 26, label, 13, "bold", MUTED)
    if sub:
        text(x + 18 + 9.4 * len(label) + 12, y + 26, sub, 12, "normal", MUTED)


W, H = 1500, 750
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
    f'aria-label="Architecture of the review assistant: an offline step builds the search indexes and statistics; each question is '
    f'planned, searched, counted, gated, answered and checked, and every model output passes a code check before it reaches the user.">')
add('<defs>')
for name, color in (("ink", INK), ("amber", AMBER), ("red", RED), ("grey", "#44505f")):
    add(f'<marker id="a-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{color}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# title + legend
text(40, 30, "Review Intelligence Assistant: architecture", 18, "bold")
lx = 820
for kind, label in (("code", "plain code"), ("llm", "LLM"), ("rank", "local ranker"), ("store", "store")):
    fill, stroke = KINDS[kind]
    add(f'<rect x="{lx}" y="17" width="16" height="16" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    text(lx + 22, 30, label, 12)
    lx += 22 + 8 * len(label) + 22
add(f'<path d="M{lx} 25 L{lx + 28} 25" stroke="{AMBER}" stroke-width="1.8" stroke-dasharray="6 4"/>')
text(lx + 34, 30, "LLM call", 12)
lx += 34 + 8 * 8 + 18
add(f'<path d="M{lx} 25 L{lx + 28} 25" stroke="{RED}" stroke-width="1.8"/>')
text(lx + 34, 30, "decline", 12)

bw = 150
cols = [55 + 205 * i for i in range(7)]  # seven boxes, 55px gaps
ctr = [c + bw / 2 for c in cols]


def link(i, j, y, label):
    arrow([(cols[i] + bw, y), (cols[j], y)])
    text((cols[i] + bw + cols[j]) / 2, y - 8, label, 11, fill=MUTED, anchor="middle", halo=True)


# ---------------- OFFLINE ----------------
panel(20, 45, 1460, 150, "OFFLINE", "rebuilt daily or weekly; no LLM")
oy, oh = 85, 88
box(cols[0], oy, bw, oh, "plain", "Kaggle CSV", ["515K hotel reviews"])
box(cols[1], oy, bw, oh, "code", "Prepare", ["clean, split by date,", "chunk by comment"])
box(cols[2], oy, bw, oh, "store", "12 BM25 indexes", ["one per city and", "sentiment"])
box(cols[3], oy, bw, oh, "store", "Train parquet", ["360K reviews, for", "statistics"])
arrow([(cols[0] + bw, oy + 44), (cols[1], oy + 44)])
arrow([(cols[1] + bw, oy + 44), (cols[2], oy + 44)])
arrow([(ctr[1], oy), (ctr[1], 72), (ctr[3], 72), (ctr[3], oy)])
text((ctr[1] + ctr[3]) / 2, 66, "train reviews", 11, fill=MUTED, halo=True, anchor="middle")
text(930, 120, "Each run writes a new versioned copy; it must pass the golden", 12, fill=MUTED)
text(930, 140, "set, then replicas switch to it one at a time (rollback = old copy).", 12, fill=MUTED)

# ---------------- ONLINE ----------------
panel(20, 215, 1460, 375, "ONLINE", "POST /ask: stateless, scale out with replicas")
r1, bh = 300, 95
mid = r1 + bh // 2
box(cols[0], r1, bw, bh, "code", "FastAPI", ["POST /ask; checks", "the input (422)"])
box(cols[1], r1, bw, bh, "llm", "1 · Plan", ["LLM: city, topic,", "scope. Code: exact", "values + operation"], ["llm", "code"])
box(cols[2], r1, bw, bh, "rank", "2 · Search", ["BM25 in matching", "indexes, reranked", "by cross-encoder"], ["rank"])
box(cols[3], r1, bw, bh, "code", "3 · Statistics", ["pandas over all", "matching reviews:", "number or breakdown"], ["code"])
box(cols[4], r1, bw, bh, "code", "4 · Gate", ["too few or poor", "matches: decline", "(no LLM call)"], ["code"])
box(cols[5], r1, bw, bh, "llm", "5 · Answer", ["LLM writes the answer", "code: IDs, numbers", "LLM: claim check"], ["llm"])
box(cols[6], r1, bw, bh, "plain", "Response", ["answer, citations,", "evidence, stats,", "decline reason"])
link(0, 1, mid, "valid")
link(1, 2, mid, "filters")
link(2, 3, mid, "reviews")
link(3, 4, mid, "stats")
link(4, 5, mid, "evidence")
link(5, 6, mid, "checked")

# stores feed the online path
arrow([(ctr[2], oy + oh), (ctr[2], r1)], "#44505f", dashed=True)
text(ctr[2] + 10, 232, "index", 11, fill=MUTED, halo=True)
arrow([(ctr[3], oy + oh), (ctr[3], r1)], "#44505f", dashed=True)
text(ctr[3] + 10, 232, "all rows", 11, fill=MUTED, halo=True)

# LLM provider
py, ph = 455, 78
px, pw = cols[1], cols[5] + bw - cols[1]
box(px, py, pw, ph, "llm", "LLM provider (OpenAI-compatible API): one model, three calls per question",
    ["One model and one setting for all calls (gpt-4o-mini, Gemini or a local model). Each call is separate and sees only its own prompt.",
     "Timeout and retries on every call. If the model is down, the service still returns evidence and statistics."])
add(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="8" fill="none" stroke="{AMBER}" stroke-width="1.6" stroke-dasharray="6 4"/>')
arrow([(ctr[1], r1 + bh), (ctr[1], py)], AMBER, dashed=True, both=True)
text(ctr[1] + 10, 431, "call 1: plan (JSON)", 11, fill=AMBER, halo=True)
arrow([(ctr[5] - 35, r1 + bh), (ctr[5] - 35, py)], AMBER, dashed=True, both=True)
text(ctr[5] - 45, 431, "call 2: answer", 11, fill=AMBER, anchor="end", halo=True)
arrow([(ctr[5] + 35, r1 + bh), (ctr[5] + 35, py)], AMBER, dashed=True, both=True)
text(ctr[5] + 45, 431, "call 3: claim check", 11, fill=AMBER, halo=True)

# declines
for i, label in ((1, "unknown city or off-topic"), (4, "too few or irrelevant matches")):
    add(f'<rect x="{ctr[i] - 100}" y="250" width="200" height="28" rx="15" fill="{KINDS["stop"][0]}" stroke="{RED}" stroke-width="1.6"/>')
    text(ctr[i], 269, label, 12, "bold", RED, "middle")
    arrow([(ctr[i], r1), (ctr[i], 278)], RED)
text(cols[0], 553, "Declines return a reason and message and never call the LLM for an answer. Too few matches still returns the few found.", 12, fill=RED)
text(cols[0], 573, "Relevance is judged right after the rerank. If an answer check fails, the evidence is still returned with answer: null.", 12, fill=RED)

# ---------------- OPERATE ----------------
panel(20, 610, 1460, 125, "OPERATE", "observability and evaluation")
arrow([(750, 590), (750, 610)], "#44505f")
text(762, 605, "every stage emits timings, counters, decline reasons", 11, fill=MUTED, halo=True)
ys, hs = 650, 64
for x, w, title, line in ((55, 325, "Logs", "one JSON line per request, no question text"),
                          (410, 325, "Metrics at /metrics", "latency per stage, declines, LLM failures"),
                          (765, 325, "Alerts", "errors, latency, failed checks, spend"),
                          (1120, 325, "Evaluation", "40-question golden set + LLM judge")):
    box(x, ys, w, hs, "plain", title, [line])

add("</svg>")
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(out))
print("wrote", sys.argv[1])
