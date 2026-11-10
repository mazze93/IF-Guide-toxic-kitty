#!/usr/bin/env python3
"""Probe 2: does a small fine-tune restore the toxicity IF-Guide suppressed?

This script holds the verdict rule and nothing else. It does not train
anything: the LoRA fine-tune is not implemented here. Run the experiment
with your own tooling, record what you measured, and pass the file in:

    python3 02_lora_retoxification.py --measurements my_run.json
    python3 02_lora_retoxification.py --self-test

The measurements file:

    {"emt_base": 0.585,              EMT of the base model, same prompts and classifier
     "corpus_size": 1000,            toxic examples the LoRA was trained on
     "measurements": [[0, 0.118],    [step, EMT]; step 0 is the unmodified IF-Guide model
                      [100, 0.21], [500, 0.40], [1000, 0.47]]}

The rule is in `lib/verdicts.py` (`lora_retoxification`) and in
knowledge/summary_ifguide.md section 7. It was fixed before any run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lib import verdicts
from lib.selftest import run_scenarios

BASE, IFG = 0.585, 0.118  # the paper's Pythia-1B EMT values, used only to build scenarios


def curve(*recoveries_by_step):
    return [(step, IFG + rec * (BASE - IFG)) for step, rec in recoveries_by_step]


SCENARIOS = {
    "confirmed": (lambda: verdicts.lora_retoxification(
        curve((0, 0), (100, 0.2), (500, 0.6), (1000, 0.8)), BASE, 1000), verdicts.CONFIRMED),
    "falsified": (lambda: verdicts.lora_retoxification(
        curve((0, 0), (500, 0.02), (1000, 0.04)), BASE, 10_000), verdicts.FALSIFIED),
    "partial": (lambda: verdicts.lora_retoxification(
        curve((0, 0), (1000, 0.3)), BASE, 1000), verdicts.INCONCLUSIVE),
    "held-but-small-corpus": (lambda: verdicts.lora_retoxification(
        curve((0, 0), (1000, 0.03)), BASE, 1000), verdicts.INCONCLUSIVE),
    "recovered-but-large-corpus": (lambda: verdicts.lora_retoxification(
        curve((0, 0), (500, 0.7)), BASE, 20_000), verdicts.INCONCLUSIVE),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurements", type=Path)
    p.add_argument("--self-test", nargs="?", const="all", choices=["all", *SCENARIOS])
    args = p.parse_args()
    if args.self_test:
        return run_scenarios("probe 2", SCENARIOS, args.self_test)
    if not args.measurements:
        print("Give --measurements FILE or --self-test. This script does not train a LoRA.", file=sys.stderr)
        return 2
    data = json.loads(args.measurements.read_text())
    outcome = verdicts.lora_retoxification(
        [tuple(m) for m in data["measurements"]], data["emt_base"], data["corpus_size"])
    print(f"{outcome.verdict}: {outcome.reason}")
    print(json.dumps(outcome.details, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
