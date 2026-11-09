# Independent Probes for IF-Guide — Protocol Stubs + Verdict Harness

Scaffolds for the adversarial probes designed in
[`knowledge/summary_ifguide.md`](../knowledge/summary_ifguide.md) §7.
These are **protocol stubs with verdict logic** — not runnable
end-to-end implementations.

**What's actually here.** For each probe: argument parsing, dataclass
schemas for results, falsification rules in `decide_verdict`, a
`--dry-run` mode that exercises the verdict pipeline on synthetic
data, and TODO comments documenting the exact transformers/peft/
Detoxify/Kronfluence calls a real implementation would make.

**What's NOT here.** Live paths — generation, Detoxify scoring,
Perspective/LlamaGuard/OpenAI-moderation scoring, RTP loading, LoRA
training, toxicity-direction computation, steered generation — all
raise `NotImplementedError`. Anyone who wants to actually run a probe
against the real IF-Guide checkpoints will need to implement those
blocks. The scripts are the scaffolding for that work, not a
substitute for it.

**Honest footer.** The claims in `knowledge/summary_ifguide.md` are
upgraded from "argued" to "demonstrated" only when a live
implementation of one of these probes has been run end-to-end against
the released checkpoints. As of this commit, that has not happened.
The scripts' value right now is as a shareable, reviewable *protocol*,
with the falsification criteria chosen in advance — so if someone else
runs them, we can read their results against a specification we
committed to before seeing them.

## The probes

| # | Script                            | Tests                                                                 | Compute (M5 Pro–class) |
| - | --------------------------------- | --------------------------------------------------------------------- | ---------------------- |
| 1 | `01_classifier_triangulation.py`  | Is the 10× effect a joint function of (method, classifier)?           | ~1 day, inference-only |
| 2 | `02_lora_retoxification.py`       | Is the suppression removable with a small LoRA fine-tune?             | Hours                  |
| 3 | `03_activation_reactivation.py`   | Is the toxicity circuit still addressable from activation space?      | Hours                  |

Probes 4, 5, 6 from the summary are not yet scaffolded here — they
need either full retraining (Probe 4) or external benchmark downloads
(Probe 5) or large score-matrix artifacts (Probe 6) that are out of
scope for a standalone-runnable directory.

## Setup

```bash
cd probes/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each script takes `--help` for its full argument surface. Reading
`--help` first is the right first move.

Checkpoints: Pythia and Llama-3 variants are on HuggingFace Hub. The
IF-Guide variants of these models may be released by the paper
authors at `ztcoalson/*` on HF — check there before training your own.
If not available, the training pipeline in `../src/` reproduces them
(~8 hours for Pythia-160M on H100-class hardware; days on MPS).

API keys: Probe 1 can use Perspective API (free with quota) and
OpenAI moderation (free endpoint with the usage policy). Llama Guard
3 runs locally. Detoxify is a pip package. Export:

```bash
export PERSPECTIVE_API_KEY=...        # optional, probe degrades to 3/4 classifiers
export OPENAI_API_KEY=...              # optional
export HF_TOKEN=...                    # required for Llama-3 variants
```

## Falsification output

Each script writes a JSON result to `--output` (default
`./results/<script-stem>.json`) and prints a one-line verdict to
stdout. The exit code reflects the verdict: 0 if the probe's primary
falsification criterion is *met* (i.e., the hypothesis is confirmed —
the paper's claim is undermined), 1 if the criterion is not met
(hypothesis falsified — the paper's claim survives), 2 if the probe
could not run to completion (ambiguous).

> **Note on exit-code semantics.** Exit 0 means "the probe *succeeded
> in its job of testing the hypothesis*," which in this adversarial
> framing means "the hypothesis is supported." This inverts the usual
> "zero means all good." Read the JSON for details before acting.

## Honest footer

Anything about IF-Guide's retention / durability / classifier-dependence
in the summary is a reading of the paper's measurements, strengthened
where the paper's own data supports it, and otherwise pending the
probes here. Nothing in the summary becomes "shown" until one of these
scripts has run end-to-end against the real checkpoints and the result
has been reproduced independently. The scaffolds pin the protocol and
the verdict thresholds in advance; the claims upgrade only when
someone — us or a reviewer — fills in the live paths and runs them.

## Known-open items the current scaffolds don't yet cover

Called out by review on 2026-11-09:

- Dry-runs exercise each verdict path but don't ship a
  `--dry-run-scenario` flag covering CONFIRMED / FALSIFIED /
  INCONCLUSIVE from a single invocation. Probes 1 and 3's current
  dry-runs each hit one verdict; a proper plumbing test matrix is a
  follow-up.
- No `probes/lib/` yet. Generation and scoring will be duplicated
  across probes once anyone actually implements them. Refactor to a
  shared module after the first live implementation lands.
- No unit tests on the `decide_verdict` functions, which are the only
  real logic in these files. Highest-ROI next commit.
