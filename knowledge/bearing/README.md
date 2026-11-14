# Claim graph for the IF-Guide summary

The claims in [`../summary_ifguide.md`](../summary_ifguide.md), written as
a graph the bearing engine can score. The engine is not in this
repository. It lives in
[mazze93/intentional-fragility](https://github.com/mazze93/intentional-fragility)
under `engine/` (Python, `bearing 0.1.0`). The graph was first scored at
commit `4b3fa67` and rescored on 2026-11-14 at `f8bf361`.

## Run it

```bash
git clone https://github.com/mazze93/intentional-fragility ../intentional-fragility
```

```bash
PYTHONPATH=../intentional-fragility/engine/src python3 -m bearing --root knowledge/bearing validate
```

```bash
PYTHONPATH=../intentional-fragility/engine/src python3 -m bearing --root knowledge/bearing fold
```

```bash
PYTHONPATH=../intentional-fragility/engine/src python3 -m bearing --root knowledge/bearing fold --json > knowledge/figures/fold.json
```

```bash
python3 knowledge/figures/make_figures.py
```

The engine needs PyYAML. Run the commands from the repository root.

## Output at ledger seq 30

```
fragile    1  claim/headline-inflated-by-shared-classifier  [paper-reported]
fragile    1  claim/proxies-select-mostly-different-tokens  [paper-reported]
fragile    1  claim/query-and-test-share-records  [repo-computed]
fragile    1  claim/table-11-toxic-cells-not-whole-counts  [paper-reported]
fragile    1  claim/table-9-means-not-reproduced  [paper-reported]
gone       0  claim/suppression-is-gated-not-deleted  []
```

`bearing audit` lists four unrun checks. It also warns that no chapter
renders the claims; there is no book here, so that warning is expected.

## What is in the graph

| Directory | Holds |
| --- | --- |
| `graph/claims/` | Six claims. `summary_section` points at the prose. |
| `graph/fragments/` | The evidence, each tied to a source. |
| `graph/checks/` | Eight checks. Four are run (all recomputations, two of them by `../checks/recompute_paper_numbers.py`). Four are probes, unrun. |
| `graph/sources/` | The paper, the shipped data, this summary. |
| `graph/ledger.yml` | Every argument edge, append-only. |

Frames: `paper-reported`, `repo-computed`, `probe-run` count.
`own-reading` is recorded and never counts.

## Limits

- The ledger was written on 2026-11-10, after the fact. Entry dates
  follow the commits where each step happened; seq 1 to 11 were not
  recorded as they occurred. Entries 25 to 30 were written on 2026-11-14
  as their checks ran.
- Entries 12 and 13 retract the two arguments once given for the
  lacuna thesis. They are kept so the overreach stays visible.
- Six claims is a selection. The summary's other observations (§5.E,
  §5.G, the label-bias reading in §5.F, §6) are not in the graph yet.
- `paper-reported` is one frame. Two tables from the same paper are not
  independent of each other.
