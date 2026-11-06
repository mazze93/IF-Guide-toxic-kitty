# Independent Probes for IF-Guide

Runnable scaffolds for the adversarial probes designed in
[`knowledge/summary_ifguide.md`](../knowledge/summary_ifguide.md) §7.

These are **shareable templates**, not verified implementations. They
load, parse arguments, follow the protocol from the summary, and emit
structured JSON results. Where a step depends on an API key, a
checkpoint path, or a benchmark that isn't bundled, the script says
so and exits — no silent hardcoded path, no fake success. A `--dry-run`
mode generates synthetic data and validates the plumbing without
making real model calls, so you can verify the script works before
spending compute.

Neither of us has run these to completion against the real IF-Guide
checkpoints as of the commit date. The point of this directory is
**reproducibility independent of us**: anyone can clone this fork,
install `requirements.txt`, point at HuggingFace checkpoints they
pulled themselves, and get an answer. The scripts are the test; the
scripts' outputs are the evidence.

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
has been reproduced independently. These scripts are the condition of
the claims upgrading from "argued" to "demonstrated."
