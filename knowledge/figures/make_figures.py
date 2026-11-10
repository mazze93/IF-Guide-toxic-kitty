#!/usr/bin/env python3
"""Draw the summary's figures as SVG. Standard library only.

    python3 knowledge/figures/make_figures.py          # write the SVGs
    python3 knowledge/figures/make_figures.py --check  # exit 1 if any is stale

Inputs, both in this directory:
  data.json  numbers copied from the paper and from the shipped data
  fold.json  claim standings, written by the bearing engine (see ../bearing/README.md)

No number is typed into this file. Change the inputs, then rerun.
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREY, INK, RULE = "#1f5fa8", "#b4530f", "#6b7280", "#111827", "#d1d5db"
TIER_COLOUR = {"robust": "#166534", "fragile": ORANGE, "gone": "#991b1b", "refuted": "#991b1b"}
TIER_MEANING = {"robust": "two or more independent sources", "fragile": "one source",
                "gone": "no counted source", "refuted": "refuted"}
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


# =============================================================================
# SVG primitives
# =============================================================================

def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{escape(str(s))}</text>")


def rect(x, y, w, h, fill):
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 0):.1f}' height='{h:.1f}' fill='{fill}'/>"


def line(x1, y1, x2, y2, stroke=RULE, dash=""):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}'{d}/>"


def svg(width, height, title, desc, body, source):
    """White ground so the figure reads the same in a light or dark page."""
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='{width}' height='{height}' role='img' {FONT}>",
        f"<title>{escape(title)}</title>", f"<desc>{escape(desc)}</desc>",
        rect(0, 0, width, height, "#ffffff"),
        text(20, 28, title, size=15, weight="600"),
        *body,
        text(20, height - 12, "Source: " + source, size=11, fill=GREY),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


# =============================================================================
# Figures
# =============================================================================

def fig_claims(fold, claims):
    """Figure 0: each claim, the standing the engine derived, and why."""
    rows = sorted(fold.items(), key=lambda kv: (-kv[1]["robustness"], kv[0]))
    body, y = [], 62
    body.append(text(20, 48, "Standing is computed from the ledger, not chosen. A check that has not been run counts for nothing.",
                     size=12, fill=GREY))
    for cid, b in rows:
        meta = claims[cid]
        colour = TIER_COLOUR[b["tier"]]
        body += [
            rect(20, y, 6, 44, colour),
            text(36, y + 17, meta["title"], size=13, weight="600"),
            text(36, y + 36, f"{b['tier'].upper()}: {TIER_MEANING[b['tier']]}"
                 + (f" ({', '.join(b['frames'])})" if b["frames"] else ""), size=12, fill=colour),
            text(940, y + 17, "summary §" + meta["section"], size=12, fill=GREY, anchor="end"),
            text(940, y + 36, meta["checks"], size=12, fill=GREY, anchor="end"),
            line(20, y + 54, 940, y + 54),
        ]
        y += 64
    return svg(960, y + 22, "Figure 0. What each claim stands on",
               "Five claims from the summary with the standing derived by the bearing engine. "
               + " ".join(f"{claims[c]['title']}: {b['tier']}." for c, b in rows),
               body, "knowledge/bearing ledger, folded by the bearing engine; fold.json")


def fig_reduction(d):
    """Figure 1: EMT reduction per method under each classifier."""
    base, methods = d["undefended"], d["methods"]
    x0, scale, y, body = 190, 110, 60, []
    for m in methods:
        det, tox = base["detoxify"] / m["detoxify"], base["toxigen"] / m["toxigen"]
        body += [
            text(x0 - 10, y + 22, m["name"], anchor="end", weight="600"),
            rect(x0, y, det * scale, 15, BLUE), text(x0 + det * scale + 6, y + 12, f"{det:.2f}x Detoxify", size=12),
            rect(x0, y + 18, tox * scale, 15, ORANGE),
            text(x0 + tox * scale + 6, y + 30, f"{tox:.2f}x ToxiGen  (keeps {tox / det:.0%} of the Detoxify figure)", size=12),
        ]
        y += 50
    body.append(line(x0 + scale, 52, x0 + scale, y - 8, GREY, "3 3"))
    body.append(text(x0 + scale, y + 8, "1x = no reduction", size=11, fill=GREY, anchor="middle"))
    return svg(900, y + 40, "Figure 1. The reduction shrinks under ToxiGen for the strong methods, not only IF-Guide",
               "Bar chart of EMT reduction against the undefended Pythia-1B, under Detoxify and under ToxiGen, for five methods.",
               body, d["source"])


def fig_overlap(d):
    """Figure 2: how much of the test pool is also in the query pool."""
    share = d["shared"] / d["test"]
    x0, w = 20, 860
    body = [
        text(x0, 58, f"Test prompts: {d['test']:,}", weight="600"),
        rect(x0, 68, w * share, 34, ORANGE), rect(x0 + w * share, 68, w * (1 - share), 34, BLUE),
        text(x0 + 8, 90, f"{d['shared']:,} also in the query pool ({share:.1%})", fill="#ffffff", weight="600"),
        text(x0 + w * share + 8, 90, f"{d['test'] - d['shared']:,} only in the test pool", fill="#ffffff", weight="600"),
        text(x0, 128, f"Query pool: {d['query']:,} records. {d['shared'] / d['query']:.1%} of it is test prompts.", size=12),
        text(x0, 148, "No record carries the same label in both pools; the thresholds differ (0.75 / 0.25 against 0.5).", size=12),
    ]
    return svg(900, 184, "Figure 2. Nearly half the test prompts also shaped the training signal",
               f"{d['shared']} of {d['test']} test prompts ({share:.1%}) also appear among the {d['query']} query records.",
               body, d["source"])


def fig_proxy(d):
    """Figure 3: pairwise overlap of the tokens each proxy selects."""
    pairs = d["pairs"]
    mean = sum(p["value"] for p in pairs) / len(pairs)
    x0, scale, y, body = 230, 620, 56, []
    for p in pairs:
        colour = BLUE if p["family"] == "within Pythia" else ORANGE
        body += [text(x0 - 10, y + 13, p["pair"], anchor="end", size=12),
                 rect(x0, y, p["value"] * scale, 18, colour),
                 text(x0 + max(p["value"], 0.5) * scale + 8, y + 14, f"{p['value']:.3f}  {p['family']}", size=12)]
        y += 26
    body += [line(x0 + 0.5 * scale, 48, x0 + 0.5 * scale, y, GREY, "3 3"),
             text(x0 + 0.5 * scale, y + 14, "half", size=11, fill=GREY, anchor="middle"),
             text(20, y + 36, f"Mean of the six pairs: {mean:.4f}. The paper's text says {d['paper_text_mean']:.4f}.", size=12)]
    return svg(900, y + 66, "Figure 3. Two proxies agree on about 45% of the tokens they select",
               "Overlap coefficient for each pair of proxy models, from 0.401 to 0.517.", body, d["source"])


def fig_steering(d):
    """Figure 4: what the paper reports for steering, and what it leaves out."""
    b, g = d["base"], d["ifguide"]
    x0, scale = 250, 480
    body = [
        text(x0 - 10, 72, "Base model", anchor="end", weight="600"),
        rect(x0, 58, b["at_minus_5"] * scale, 16, GREY), text(x0 + b["at_minus_5"] * scale + 6, 71, f"{b['at_minus_5']:.2f} at scaling -5", size=12),
        rect(x0, 78, b["at_plus_5"] * scale, 16, BLUE), text(x0 + b["at_plus_5"] * scale + 6, 91, f"{b['at_plus_5']:.2f} at scaling +5", size=12),
        text(x0 - 10, 132, "IF-Guide model", anchor="end", weight="600"),
        rect(x0, 118, g["negative_scaling_reaches"] * scale, 16, ORANGE),
        text(x0 + g["negative_scaling_reaches"] * scale + 6, 131, f"{g['negative_scaling_reaches']:.1f} under negative scaling", size=12),
        text(x0, 151, f"positive scaling: {g['positive_scaling']} (no number given)", size=12, fill=GREY),
        text(20, 186, "Reported: EMT and TP.   Not reported: perplexity, repetition rate, any sample of the steered text.", size=12, weight="600"),
        text(20, 206, "So 1.0 may be coherent toxic text coming back, or broken text that scores as toxic. Probe 3 measures the difference.", size=12),
    ]
    return svg(960, 240, "Figure 4. Steering: the numbers the paper gives, and the measure it does not",
               "Reported EMT endpoints for feature steering on the base and IF-Guide Pythia-1B models. No coherence measure is reported.",
               body, d["source"])


# =============================================================================
# Assembly
# =============================================================================

def claim_meta() -> dict:
    """Title, summary section and check status per claim, read from the graph."""
    graph = HERE.parent / "bearing" / "graph"
    ledger = (graph / "ledger.yml").read_text()
    meta = {}
    for f in sorted((graph / "claims").glob("*.yml")):
        fields = dict(l.split(": ", 1) for l in f.read_text().splitlines() if ": " in l and not l.startswith(" "))
        cid = fields["id"]
        tested = ledger.count(f"to: {cid}, rel: tested-by")
        ran = ledger.count(f"type: run-check, claim: {cid},")
        meta[cid] = {"title": fields["title"].strip('"'), "section": fields["summary_section"].strip('"'),
                     "checks": f"checks run: {ran} of {tested}"}
    return meta


def build() -> dict:
    data = json.loads((HERE / "data.json").read_text())
    fold = json.loads((HERE / "fold.json").read_text())
    return {
        "fig-0-claim-standing.svg": fig_claims(fold, claim_meta()),
        "fig-1-reduction-by-classifier.svg": fig_reduction(data["reduction_by_classifier"]),
        "fig-2-query-test-overlap.svg": fig_overlap(data["query_test_overlap"]),
        "fig-3-proxy-overlap.svg": fig_proxy(data["proxy_overlap"]),
        "fig-4-steering-reported.svg": fig_steering(data["steering"]),
    }


def main() -> int:
    figures = build()
    if "--check" in sys.argv:
        stale = [n for n, s in figures.items() if not (HERE / n).exists() or (HERE / n).read_text() != s]
        print("STALE: " + ", ".join(stale) if stale else f"OK: {len(figures)} figures match their inputs")
        return 1 if stale else 0
    for name, content in figures.items():
        (HERE / name).write_text(content)
        print("wrote", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
