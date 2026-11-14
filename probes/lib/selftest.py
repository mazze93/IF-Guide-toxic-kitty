"""Scenario self-tests: run a verdict rule on synthetic inputs built to land
on each of the three verdicts, and check that it does.

This replaces the first scaffolds' dry-runs, which fed every probe synthetic
data shaped to return CONFIRMED and so exercised one branch of three.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from .verdicts import Verdict

Scenario = Tuple[Callable[[], Verdict], str]  # (build-and-judge, expected verdict)


def run_scenarios(probe: str, scenarios: Dict[str, Scenario], only: str = "all") -> int:
    """Print one line per scenario. Return 0 if every verdict matched, else 3."""
    chosen = scenarios if only == "all" else {only: scenarios[only]}
    mismatches = 0
    for name, (judge, expected) in chosen.items():
        got = judge()
        ok = got.verdict == expected
        mismatches += 0 if ok else 1
        print(f"[{'ok' if ok else 'MISMATCH'}] {probe} / {name}: expected {expected}, got {got.verdict}")
        print(f"       {got.reason}")
    print("SYNTHETIC DATA. This checks the verdict logic, not any model.")
    return 0 if mismatches == 0 else 3
