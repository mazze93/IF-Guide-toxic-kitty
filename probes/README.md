# Probes for IF-Guide: verdict rules, tests, and one scoring path

Designs are in [`knowledge/summary_ifguide.md`](../knowledge/summary_ifguide.md) §7.
**No probe has been run on an IF-Guide model.** None is public (summary §10),
so nothing here is evidence about the paper yet.

## What is here

| File | What it does | State |
| --- | --- | --- |
| `lib/verdicts.py` | The three verdict rules, as pure functions. The only place a threshold lives. | Tested |
| `tests/test_verdicts.py` | 32 unit tests pinning the rules. | Pass |
| `01_classifier_triangulation.py` | Re-scores two saved outputs files with other classifiers, applies rule 1. | Runs; see limits below |
| `generate_outputs.py` | Writes a paper-format outputs file for any Hugging Face causal LM. | Runs |
| `02_lora_retoxification.py` | Applies rule 2 to measurements you supply. **Does not train a LoRA.** | Verdict harness only |
| `03_activation_reactivation.py` | Applies rule 3 to measurements you supply. **Does not steer a model.** | Verdict harness only |

Not implemented: LoRA training, direction computation, activation hooks,
perplexity measurement, the OpenAI moderation scorer, the Perspective scorer.
Probes 4 to 6 from the summary have no code.

## Run it

```bash
cd probes
python3 -m unittest discover -s tests -v      # needs only the standard library
```

```bash
python3 01_classifier_triangulation.py --self-test
```

```bash
python3 02_lora_retoxification.py --self-test
```

```bash
python3 03_activation_reactivation.py --self-test
```

Each `--self-test` builds synthetic inputs for every verdict (CONFIRMED,
FALSIFIED, INCONCLUSIVE and the guard cases) and checks the rule returns it.
It tests the logic, not any model.

The scoring path needs the packages in `requirements.txt`:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python 01_classifier_triangulation.py \
    --base-outputs results/base.outputs.json \
    --ifguide-outputs results/ifguide.outputs.json
```

Outputs files come from `src/run_toxicity_eval.py --save_outputs` (the
paper's harness) or from `generate_outputs.py`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Ran and printed a verdict. Any verdict; read it. |
| 2 | Could not run (missing or mismatched inputs). |
| 3 | A self-test scenario returned the wrong verdict. |

## Limits of Probe 1 as it stands

The verdict needs two "deciding" classifiers: ones that measure explicit
toxicity, as Detoxify does, on training data unrelated to Jigsaw. One runs
here (`olid-offensive`). The second (`openai-moderation`) is declared but
not written. So this script can only return INCONCLUSIVE today, plus a
table. ToxiGen and Dynabench are reported but cannot decide: they measure
implicit hate and hate speech, which are different things, and the paper's
own Table 2 already shows a smaller reduction under ToxiGen for every
method (summary §5.A).

Each classifier must score a hostile sentence above a neutral one before
its numbers are used. This catches a wrong label column.

## Pipeline check, 2026-11-10

Run once on CPU to confirm the scoring path works end to end. **Both
models are ordinary public Pythia checkpoints. Neither is detoxified. The
numbers say nothing about IF-Guide.** 24 prompts, 5 completions each
(the paper uses 10,000 prompts and 25).

| classifier | role | pythia-70m EMT | pythia-160m EMT | ratio |
| --- | --- | --- | --- | --- |
| detoxify | anchor | 0.175 | 0.275 | 0.64x |
| olid-offensive | deciding | 0.307 | 0.434 | 0.71x |
| toxigen | reported | 0.166 | 0.197 | 0.84x |
| dynabench-hate | reported | 0.271 | 0.215 | 1.26x |

Verdict: INCONCLUSIVE, "detoxify shows only 0.64x. There is no effect here
to test." That is the correct answer for two undetoxified models.

## How well the tests guard the rules

A separate script changed one threshold or comparison at a time in
`lib/verdicts.py` (23 changes) and reran the tests. All 23 were caught.
One was missed on the first pass; the test added for it is marked in
`tests/test_verdicts.py`. The mutation script is not in this repository.

The thresholds themselves are choices made before any run. The two
coherence limits in rule 3 are untested and should be checked against the
base model's own text first.
