#!/usr/bin/env python3
"""Probe 3: does negative steering bring back coherent toxic text?

This script holds the verdict rule and nothing else. It does not steer a
model: computing the direction, hooking activations and measuring
perplexity are not implemented here. Run the sweep with your own tooling,
record each cell, and pass the file in:

    python3 03_activation_reactivation.py --measurements my_sweep.json
    python3 03_activation_reactivation.py --self-test

The measurements file:

    {"emt_base": 0.585, "emt_ifguide": 0.118,    unsteered, same prompts and classifier
     "points": [{"hook": "layer.final.last-token", "direction": "ifguide-self",
                 "scaling": -5.0, "emt": 0.52,
                 "perplexity_ratio": 1.3,          steered text under the unsteered model,
                                                   over the unsteered baseline
                 "repetition_rate": 0.06}, ...]}   lib.metrics.repetition_rate

Why coherence is recorded: the paper's section 4.6 reports EMT = 1.0 under
negative steering and nothing about the text. Broken, repetitive output
also scores as toxic. A point counts only when its text is still coherent.

The rule is in `lib/verdicts.py` (`steering_reactivation`) and in
knowledge/summary_ifguide.md section 7. The two coherence thresholds are
untested choices; check them on the base model before trusting a verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lib import verdicts
from lib.selftest import run_scenarios
from lib.verdicts import REFERENCE_HOOK, SELF_DIRECTION, SteerPoint

BASE, IFG = 0.585, 0.118  # the paper's Pythia-1B EMT values, used only to build scenarios
HOOKS = (REFERENCE_HOOK, "layer.mid.last-token")


def sweep(recovery_at_minus_five: float, coherent: bool = True, hooks=HOOKS, direction=SELF_DIRECTION):
    ppl, rep = (1.2, 0.05) if coherent else (6.0, 0.7)
    points = []
    for hook in hooks:
        for scaling, share in ((0.0, 0.0), (-2.5, 0.5), (-5.0, 1.0)):
            emt = IFG + share * recovery_at_minus_five * (BASE - IFG)
            points.append(SteerPoint(hook, direction, scaling, emt, ppl, rep))
    return points


def judge(points):
    return verdicts.steering_reactivation(points, BASE, IFG)


SCENARIOS = {
    "confirmed": (lambda: judge(sweep(0.9)), verdicts.CONFIRMED),
    "falsified-flat": (lambda: judge(sweep(0.02)), verdicts.FALSIFIED),
    "falsified-degenerate": (lambda: judge(sweep(1.8, coherent=False)), verdicts.FALSIFIED),
    "partial": (lambda: judge(sweep(0.3)), verdicts.INCONCLUSIVE),
    "one-hook-only": (lambda: judge(sweep(0.9, hooks=(REFERENCE_HOOK,))), verdicts.INCONCLUSIVE),
    "transplanted-direction-only": (lambda: judge(sweep(0.9, direction="base-transplanted")), verdicts.INCONCLUSIVE),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurements", type=Path)
    p.add_argument("--self-test", nargs="?", const="all", choices=["all", *SCENARIOS])
    args = p.parse_args()
    if args.self_test:
        return run_scenarios("probe 3", SCENARIOS, args.self_test)
    if not args.measurements:
        print("Give --measurements FILE or --self-test. This script does not steer a model.", file=sys.stderr)
        return 2
    data = json.loads(args.measurements.read_text())
    points = [SteerPoint(**point) for point in data["points"]]
    outcome = verdicts.steering_reactivation(points, data["emt_base"], data["emt_ifguide"])
    print(f"{outcome.verdict}: {outcome.reason}")
    print(json.dumps(outcome.details, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
