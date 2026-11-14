#!/usr/bin/env python3
"""Recompute averages and fractions that IF-Guide v3 reports, from its own tables.

Pure arithmetic on numbers copied from the paper (arXiv 2506.01790v3,
Appendix F Table 9 and Appendix H Table 11). No model, no data files.
Run: python3 knowledge/checks/recompute_paper_numbers.py
"""
from itertools import combinations
from math import comb, sqrt

# ---- Appendix F, Table 9: overlap coefficient per proxy pair (3 decimals) ----
TABLE_9 = {
    ("Pythia-160M", "Pythia-410M"): 0.466,
    ("Pythia-160M", "Pythia-1B"): 0.454,
    ("Pythia-160M", "Llama-3.2-1B"): 0.419,
    ("Pythia-410M", "Pythia-1B"): 0.517,
    ("Pythia-410M", "Llama-3.2-1B"): 0.401,
    ("Pythia-1B", "Llama-3.2-1B"): 0.440,
}
# Averages the Appendix F text states, in percent.
TEXT_MEANS = {"all pairs": 44.61, "within Pythia": 47.22, "cross family": 42.00,
              "within ~500M params": 47.27, "differ by >500M params": 43.61}
ROUNDING = 0.0005  # largest error a 3-decimal table entry can carry


def is_pythia_pair(pair):
    return all(name.startswith("Pythia") for name in pair)


def mean(values):
    return sum(values) / len(values)


def report_table_9():
    groups = {
        "all pairs": list(TABLE_9.values()),
        "within Pythia": [v for p, v in TABLE_9.items() if is_pythia_pair(p)],
        "cross family": [v for p, v in TABLE_9.items() if not is_pythia_pair(p)],
    }
    print("Table 9 means recomputed from the six off-diagonal values")
    for name, vals in groups.items():
        m = 100 * mean(vals)
        gap = abs(m - TEXT_MEANS[name])
        expr = " + ".join(f"{v:.3f}" for v in vals)
        print(f"  {name:14s} ({expr}) / {len(vals)} = {m:.2f}%   "
              f"text {TEXT_MEANS[name]:.2f}%   gap {gap:.2f}pp   "
              f"rounding can explain <= {100 * ROUNDING:.2f}pp")
    report_unmatched_subset_means()


def subset_means(items):
    for k in range(1, len(items) + 1):
        for combo in combinations(items, k):
            yield combo, 100 * mean([v for _, v in combo])


def report_unmatched_subset_means():
    """Can the scale-split figures come from any grouping of the six pairs?

    The paper does not say which pairs fall in each scale group, so every
    subset is tried. The two groups must partition the six pairs.
    """
    items = list(TABLE_9.items())
    near, far = TEXT_MEANS["within ~500M params"], TEXT_MEANS["differ by >500M params"]
    tol = 100 * ROUNDING
    for label, target in (("within ~500M params", near), ("differ by >500M params", far)):
        hits = [c for c, m in subset_means(items) if abs(m - target) <= tol]
        names = ["; ".join(f"{a}/{b}" for (a, b), _ in c) for c in hits]
        print(f"  {label}: text {target:.2f}%; subsets of Table 9 averaging to it "
              f"within rounding: {len(hits)} {names}")
    partitions = [c for c, m in subset_means(items)
                  if abs(m - near) <= tol and len(c) < len(items)
                  and abs(100 * mean([v for p, v in items if (p, v) not in c]) - far) <= tol]
    print(f"  partitions of the six pairs giving both {near:.2f}% and {far:.2f}%: {len(partitions)}")


# ---- Appendix H, Table 11: recognition accuracy, prompted on 100 + 100 ----
TABLE_11 = {"Base": (87.5, 27.0), "IF-Guide": (76.7, 34.0)}
PROMPTED_PER_CLASS = 100


def denominators(percent, max_n=PROMPTED_PER_CLASS):
    """Every n <= max_n for which some k/n rounds to `percent` at 1 decimal."""
    found = []
    for n in range(1, max_n + 1):
        for k in range(n + 1):
            if round(100 * k / n, 1) == percent:
                found.append(f"{k}/{n}")
    return found


def report_table_11():
    print("\nTable 11 cells: which counts out of n <= 100 give the printed percentage")
    for model, (tox, non) in TABLE_11.items():
        tox_n = [d for d in denominators(tox) if d.endswith("/100")]
        print(f"  {model:9s} toxic {tox:.1f}%: k/100 possible? {'yes' if tox_n else 'no'}; "
              f"fractions with n >= {MIN_PLAUSIBLE_N}: {[d for d in denominators(tox) if int(d.split('/')[1]) >= MIN_PLAUSIBLE_N]}")
        print(f"  {model:9s} non-toxic {non:.1f}%: {non:.0f}/100 = {non:.1f}%")
    for model, (tox, non) in TABLE_11.items():
        print(f"  balanced accuracy {model}: ({tox} + {non}) / 2 = {(tox + non) / 2:.2f}%")
    delta_t = TABLE_11['IF-Guide'][0] - TABLE_11['Base'][0]
    delta_n = TABLE_11['IF-Guide'][1] - TABLE_11['Base'][1]
    print(f"  change: toxic {delta_t:+.1f}pp, non-toxic {delta_n:+.1f}pp")
    report_intervals()


MIN_PLAUSIBLE_N = 60  # assumption: at least 60 of the 100 toxic prompts got a usable answer
Z95 = 1.96


def toxic_n_range(model):
    """Smallest and largest n in [MIN_PLAUSIBLE_N, 100] consistent with the toxic cell."""
    ns = [int(d.split("/")[1]) for d in denominators(TABLE_11[model][0])]
    ns = [n for n in ns if n >= MIN_PLAUSIBLE_N]
    return min(ns), max(ns)


def half_width(percent, n):
    """95% Wald half-width in percentage points for a proportion."""
    p = percent / 100
    return 100 * Z95 * sqrt(p * (1 - p) / n)


def report_intervals():
    print("\n95% Wald half-widths (pp); toxic cells at their smallest and largest possible n")
    for model, (tox, non) in TABLE_11.items():
        lo_n, hi_n = toxic_n_range(model)
        print(f"  {model:9s} toxic {tox}%: n=100 {half_width(tox, 100):.1f}, "
              f"n={hi_n} {half_width(tox, hi_n):.1f}, n={lo_n} {half_width(tox, lo_n):.1f}; "
              f"non-toxic {non}% (n=100) {half_width(non, 100):.1f}")
    (bt, bn), (it, inn) = TABLE_11["Base"], TABLE_11["IF-Guide"]
    (b_lo, b_hi), (i_lo, i_hi) = toxic_n_range("Base"), toxic_n_range("IF-Guide")
    for label, b_n, i_n in (("largest n", b_hi, i_hi), ("smallest n", b_lo, i_lo)):
        hw = sqrt(half_width(bt, b_n) ** 2 + half_width(it, i_n) ** 2)
        print(f"  toxic change {it - bt:+.1f}pp, 95% half-width at {label}: {hw:.1f}pp")
    hw = sqrt(half_width(bn, 100) ** 2 + half_width(inn, 100) ** 2)
    print(f"  non-toxic change {inn - bn:+.1f}pp, 95% half-width: {hw:.1f}pp "
          f"(independent samples)")
    report_paired_bounds()


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for b flips one way and c the other."""
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def report_paired_bounds():
    """Appendix H prompts "both models with 200 examples" (read as the same 200), so the
    test is paired. The table fixes only the net flips, not how many went
    each way, so print p across the possible splits for the non-toxic cell."""
    net = round(TABLE_11["IF-Guide"][1] - TABLE_11["Base"][1])
    print(f"\nPaired (same prompts), non-toxic: net {net:+d} of 100; exact McNemar p by flips back")
    for back in (0, 1, 2, 5, 10):
        print(f"  {net + back} to correct, {back} back: p = {mcnemar_exact(net + back, back):.3f}")


if __name__ == "__main__":
    report_table_9()
    report_table_11()
