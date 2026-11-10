#!/usr/bin/env python3
"""Probe 1: is IF-Guide's measured reduction tied to Detoxify?

Re-scores two saved outputs files (base model and IF-Guide model, written
by `src/run_toxicity_eval.py --save_outputs` or `generate_outputs.py`) with
several classifiers and applies the rule in `lib/verdicts.py`.

Only classifiers that measure what Detoxify measures, trained on unrelated
data, can decide the verdict. See `lib/scorers.py` for each one's role and
knowledge/summary_ifguide.md section 7 for why.

    python3 01_classifier_triangulation.py \\
        --base-outputs results/base.outputs.json \\
        --ifguide-outputs results/ifguide.outputs.json

    python3 01_classifier_triangulation.py --self-test        # verdict logic only

Status: the scoring path runs and has been exercised on two public models
as a pipeline check. It has not been run on an IF-Guide model, because none
is public. Two classifiers (OpenAI moderation, Perspective) are declared
but not implemented, so with the classifiers that do run here the verdict
can only be INCONCLUSIVE: one deciding classifier is not enough.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lib import verdicts
from lib.metrics import emt_tp, reduction
from lib.outputs_io import completions_per_prompt, load_outputs, same_prompts, stored_detoxify_per_prompt
from lib.scorers import CLASSIFIERS, load_hf_scorer
from lib.selftest import run_scenarios

LOCAL_DEFAULT = "olid-offensive,toxigen,dynabench-hate"
DECIDING = [name for name, c in CLASSIFIERS.items() if c.role == "deciding"]

# Known-answer texts: a classifier that scores the neutral line above the hostile
# one is reading the wrong label column, and its numbers must not be used.
NEUTRAL = "The library opens at nine on weekdays."
HOSTILE = "You are a stupid, worthless idiot and I hate people like you."


# =============================================================================
# Scoring
# =============================================================================

def known_answer_check(name: str, score) -> None:
    neutral, hostile = score([NEUTRAL, HOSTILE])
    if hostile <= neutral:
        raise RuntimeError(
            f"{name}: scored the hostile line {hostile:.3f} and the neutral line {neutral:.3f}. "
            "Refusing to use it; check its positive label."
        )


def score_per_prompt(score, completions: list) -> list:
    flat = [text for prompt in completions for text in prompt]
    flat_scores = score(flat)
    out, cursor = [], 0
    for prompt in completions:
        out.append(flat_scores[cursor:cursor + len(prompt)])
        cursor += len(prompt)
    return out


def measure(names: list, base: list, ifguide: list) -> tuple:
    """Return (rows, not_run). One row per classifier that produced numbers."""
    rows, not_run = [], []
    anchor_base, _ = emt_tp(stored_detoxify_per_prompt(base))
    anchor_ifg, _ = emt_tp(stored_detoxify_per_prompt(ifguide))
    rows.append(row(CLASSIFIERS["detoxify"], anchor_base, anchor_ifg))
    for name in names:
        classifier = CLASSIFIERS[name]
        if classifier.hf_model is None:
            not_run.append({"classifier": name, "role": classifier.role, "needs": classifier.needs})
            continue
        score = load_hf_scorer(classifier)
        known_answer_check(name, score)
        base_emt, _ = emt_tp(score_per_prompt(score, completions_per_prompt(base)))
        ifg_emt, _ = emt_tp(score_per_prompt(score, completions_per_prompt(ifguide)))
        rows.append(row(classifier, base_emt, ifg_emt))
    return rows, not_run


def row(classifier, base_emt: float, ifg_emt: float) -> dict:
    return {
        "classifier": classifier.name, "role": classifier.role,
        "measures": classifier.measures, "lineage": classifier.lineage,
        "base_emt": round(base_emt, 4), "ifguide_emt": round(ifg_emt, 4),
        "reduction": round(reduction(base_emt, ifg_emt), 3),
    }


def judge(rows: list) -> verdicts.Verdict:
    return verdicts.classifier_triangulation(
        {r["classifier"]: r["reduction"] for r in rows},
        {r["classifier"]: r["base_emt"] for r in rows},
        DECIDING,
    )


# =============================================================================
# Self-test scenarios (synthetic)
# =============================================================================

def scenario(anchor: float, deciding: dict):
    base = {"detoxify": 0.585, **{name: 0.5 for name in deciding}}
    return lambda: verdicts.classifier_triangulation({"detoxify": anchor, **deciding}, base, DECIDING)


SCENARIOS = {
    "confirmed": (scenario(5.0, {"olid-offensive": 2.0, "openai-moderation": 2.4}), verdicts.CONFIRMED),
    "falsified": (scenario(5.0, {"olid-offensive": 4.7, "openai-moderation": 5.3}), verdicts.FALSIFIED),
    "inconclusive": (scenario(5.0, {"olid-offensive": 3.5, "openai-moderation": 3.6}), verdicts.INCONCLUSIVE),
    "no-anchor-effect": (scenario(1.0, {"olid-offensive": 1.0, "openai-moderation": 1.0}), verdicts.INCONCLUSIVE),
    "one-deciding-only": (scenario(5.0, {"olid-offensive": 2.0}), verdicts.INCONCLUSIVE),
}


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-outputs", type=Path)
    p.add_argument("--ifguide-outputs", type=Path)
    p.add_argument("--classifiers", default=LOCAL_DEFAULT,
                   help=f"comma-separated, from: {', '.join(n for n in CLASSIFIERS if n != 'detoxify')}")
    p.add_argument("--label", default="", help="free text recorded in the result, e.g. what the two models are")
    p.add_argument("--output", type=Path, default=Path("results/01_classifier_triangulation.json"))
    p.add_argument("--self-test", nargs="?", const="all", choices=["all", *SCENARIOS],
                   help="run the verdict rule on synthetic scenarios and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_scenarios("probe 1", SCENARIOS, args.self_test)
    if not (args.base_outputs and args.ifguide_outputs):
        print("Give --base-outputs and --ifguide-outputs, or --self-test.", file=sys.stderr)
        return 2
    names = [n.strip() for n in args.classifiers.split(",") if n.strip()]
    unknown = [n for n in names if n not in CLASSIFIERS or n == "detoxify"]
    if unknown:
        print(f"Unknown classifier(s): {unknown}", file=sys.stderr)
        return 2

    base, ifguide = load_outputs(args.base_outputs), load_outputs(args.ifguide_outputs)
    if not same_prompts(base, ifguide):
        print("The two outputs files do not cover the same prompts in the same order.", file=sys.stderr)
        return 2

    rows, not_run = measure(names, base, ifguide)
    outcome = judge(rows)
    result = {
        "probe": "classifier_triangulation", "label": args.label,
        "base_outputs": str(args.base_outputs), "ifguide_outputs": str(args.ifguide_outputs),
        "prompts": len(base), "completions_per_prompt": len(base[0]["outputs"]),
        "rows": rows, "not_run": not_run,
        "verdict": outcome.verdict, "reason": outcome.reason, "details": outcome.details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))

    print(f"{'classifier':<18}{'role':<10}{'base EMT':>9}{'IF-G EMT':>10}{'reduction':>11}")
    for r in rows:
        print(f"{r['classifier']:<18}{r['role']:<10}{r['base_emt']:>9.3f}{r['ifguide_emt']:>10.3f}{r['reduction']:>10.2f}x")
    for item in not_run:
        print(f"{item['classifier']:<18}{item['role']:<10}not run: {item['needs']}")
    print(f"\n{outcome.verdict}: {outcome.reason}\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
