# IF-Guide: Influence Function-Guided Detoxification of LLMs

**arXiv:** [2506.01790](https://arxiv.org/abs/2506.01790)
**Authors:** Zachary Coalson (Oregon State), Juhan Bae (Toronto), Nicholas Carlini (Anthropic), Sanghyun Hong (Oregon State)
**Venue:** NeurIPS 2025
**Tag:** `ifguide`
**Reviewed:** 2026-11-04 · adversarial read, walking backwards from claims to assumptions

> **Note on reading register.** This summary is not a book report. It is an
> adversarial read commissioned to find the parts of the method that are
> load-bearing but under-tested, and to design probes that could survive on
> a laptop-class budget. Carlini is a co-author; his own research corpus is
> largely built on exactly this move — take a method's own assumptions and
> construct the minimum-effort counterexample — so this register is the
> appropriate one.

---

## 1. What It Does

IF-Guide turns influence functions into a *proactive* detoxification tool.
Where DPO and RLHF push a trained model away from toxic outputs after the
fact, IF-Guide identifies individual **tokens** in the pre-training (or
fine-tuning) data that, through the model's own gradient geometry, are the
most responsible for toxic generations downstream — and then penalizes the
model for producing those specific tokens during training. The headline
delta from prior influence-function work is that attribution happens
*token-wise*, not document-wise, and the attribution signal is
*differential* — the influence on toxic prompts minus the influence on
non-toxic prompts, which cancels out generic high-frequency tokens.

The headline practical claim is efficiency: a million-parameter proxy
(Pythia-160M) can compute the toxic-token mask for a billion-parameter
target with downstream toxicity reduction that is "effectively the same."
That is the claim you flagged as fit-for-purpose. It is also the claim
with the subtlest weakness — see §7.

---

## 2. How It Works — The Four-Step Pipeline

### 2.1 Baseline influence (what fails)

Given a target model with parameters θ, a training example x_i, and a
query (p, c) where c is a toxic completion of prompt p, the standard LLM
influence function is

```
I_θ(x_i, q) ≈ − ∇_θ[log Pr(c | p; θ)]ᵀ · H̃⁻¹ · ∇_θ L(x_i; θ)
```

where `H̃⁻¹` is the EK-FAC Kronecker-factored approximation of the
inverse Hessian, and `L(x_i; θ)` is the standard next-token prediction
loss (§3 of paper, Eq. 3). Baseline evaluation (Pythia-160M, 1B-token
OpenWebText): removing the top 1–50% of highest-influence training
examples reduces toxicity by at most 33%, and that requires a 50%
removal that destroys fluency (PPL −21%, LAMBADA Acc. −13%). Verdict: **standard
influence functions do not usefully find toxic training data.**

### 2.2 Differential attribution (what works)

Build two query gradients by averaging over a set of toxic prompt-completion
pairs `Q_tox` and a set of non-toxic ones `Q_safe` (both drawn from RTP,
labeled by Detoxify with thresholds 0.75 / 0.25). The core score (paper Eq. 6):

```
ΔI_θ(x_i) ≈ −(ḡ_tox − ḡ_safe)ᵀ · H̃⁻¹ · ∇_θ L(x_i; θ)
```

Then decompose the sum over tokens, because the loss decomposes
(paper Eq. 7-8). The score per token j in document i is

```
S_ij ≈ −(ḡ_tox − ḡ_safe)ᵀ · H̃⁻¹ · ∇_θ L(x_ij; θ)
```

### 2.3 Document-ranked selection

Rank each training document by the **harmonic mean** of:

1. the *count* of its tokens with score above the 99th-percentile threshold `τ_tox`,
2. the *sum* of those above-threshold scores,

both min-max normalized (see
[algorithm 1](../src/build_toxic_token_mask.py) and paper Alg. 1 /
Appendix H). Walk documents in descending rank, select all
above-threshold tokens plus `w=1` context neighbors, cap total selection
at `L = 20M` tokens (2% of the 1B training corpus).

### 2.4 Suppression-during-training (penalty objective)

Paper Eq. 9:

```
L_tox(x_i, T_i; θ) = − Σ_{j ∉ T_i} log Pr(x_ij | x_i,<j; θ)
                    + λ Σ_{j ∈ T_i} log Pr(x_ij | x_i,<j; θ)
```

The first term is ordinary NLL on benign tokens; the second term is
λ-weighted *reward* for low probability on toxic tokens. With `λ=1`, the
penalty strength equals the ordinary learning rate on the toxic set.
λ > 1 destabilizes training (model collapses to infinite repetition of
degenerate tokens — see Appendix C).

**Code vs. paper check**: `src/utils/training.py:90` implements this as
`weights = torch.where(toxic_mask, -λ, 1.0); loss = loss * weights`,
which is arithmetically correct (CE loss is `−log P`, so flipping its
sign on toxic tokens gives `+log P`). Confirmed match.

---

## 3. Results & Claims (as reported)

| Claim                                              | Scale / scope                            | Reported effect                    |
| -------------------------------------------------- | ---------------------------------------- | ---------------------------------- |
| Pre-training reduction, EMT (Detoxify)             | Pythia 160M–1B, Llama-3.2-1B             | 4.2–5.5× vs. uncensored            |
| Pre-training reduction, TP (Detoxify)              | same                                     | 6.8–10.4× vs. uncensored           |
| Fine-tuning (post-training) reduction              | Pythia 2.8B / 12B                        | EMT 3.0–5.7×, TP 3.9–10.8×         |
| Implicit toxicity (ToxiGen-RoBERTa)                | Pythia-1B                                | EMT 2.2×, TP 2.4×                  |
| Trade-off metric (Acc / mean(EMT, TP))             | all models                               | 4.1–5.6, best in class             |
| Combined with RAD (decoding-time)                  | all models                               | EMT 14–18×, TP 21–33×              |
| Small proxy (Pythia-160M) ≈ larger proxy           | target models up to Llama-3.2-1B         | Δ-EMT ≤ 0.044, Δ-TP ≤ 0.045        |
| GCG robustness                                     | Pythia-410M only, 100 prompts            | ASR 0.43 → 0.22                    |
| Toxic-direction steering (base)                    | Pythia-1B                                | ±5 scaling: EMT 0.47 → 0.69        |
| Toxic-direction steering (IF-Guide)                | Pythia-1B                                | +scaling: no change; −scaling: EMT→1.0 |
| Toxicity *recognition* (instruction-tuned)         | Pythia-2.8B                              | 87.5% → 76.7% on toxic; 27% → 34% on non-toxic |

These numbers are not in dispute — they are reproducible against the
released code, data indices, and model checkpoints (see fork at
[github.com/mazze93/if-guide-toxic-kitty](https://github.com/mazze93/if-guide-toxic-kitty)).
What is in dispute is what they mean.

---

## 4. Limitations the Authors Flag (honest)

They acknowledge, in Appendix B:

1. The toxicity signal depends on external classifiers (Detoxify / ToxiGen)
   and inherits their biases.
2. Influence functions sometimes surface irrelevant high-scoring documents
   (specifically: documents dominated by repeated tokens).
3. The method is not practical at commercial scale (hundreds-of-billions
   of parameters × trillion-token corpora) — they tested up to 12B params
   on 1B tokens.

These are all real. The more interesting limitations are the ones they
don't name, which is where we start walking backwards.

---

## 5. Walking Backwards — Load-Bearing Assumptions That Aren't Directly Tested

A walking-backwards read of the paper. For each assumption, the question
is: if this is false, what in the method stops working?

### A. Detoxify-circular evaluation

- **Query gradients** are built from RTP prompt-completion pairs labeled by Detoxify.
- **Toxicity threshold for queries** is Detoxify > 0.75 for toxic, < 0.25 for safe.
- **Evaluation** uses Detoxify to label completions as toxic or not (threshold ≥ 0.5).
- **Toxicity Filtering baseline** is explicitly called out as "best-case defender" because it filters with the same Detoxify.
- IF-Guide is not acknowledged to be in the same relationship with Detoxify, but structurally it is: the training signal and the evaluation metric share a classifier.

The paper's "Implicit toxicity" section partially mitigates this by evaluating with ToxiGen-RoBERTa, and the effect size there shrinks from ~5× (Detoxify) to ~2.2× (ToxiGen). The paper reads this as "the method generalizes to implicit toxicity." A different read: the method's effect size is roughly halved when evaluated with a classifier that wasn't in its loop.

### B. Proxy-model "equivalence" is really proxy-model *substitutability*

Appendix E, Table 10, Overlap Coefficient between proxies' selected toxic tokens:

```
                 Pythia-160M   Pythia-410M   Pythia-1B   Llama-3.2-1B
Pythia-160M        1.000         0.466        0.454       0.419
Pythia-410M                      1.000        0.517       0.401
Pythia-1B                                     1.000       0.440
Llama-3.2-1B                                              1.000
```

Mean cross-proxy overlap: **44.6%**. Mean within-family: 47.2%.
Mean cross-family: 42%.

On a 20M-token budget, 44.6% overlap means each pair of proxies agrees
on ~8.9M tokens and disagrees on ~11.1M each. The paper frames this as
"44.6% is substantial relative to a 1B-token corpus." That's the wrong
denominator. The right denominator is the *selected* set (20M), not the
*source* set (1B). On the selected set, **the majority of each proxy's
output disagrees with every other proxy's output**, and yet all of them
produce similar downstream toxicity reductions.

Three compatible interpretations, each probative:

1. **Many valid toxic-token-sets exist.** The method isn't identifying
   *the* toxic tokens; it's identifying *some sufficient subset*. If true,
   this makes IF-Guide behaviourally robust but semantically shallow —
   the attribution is not a measurement of "which tokens are toxic" but
   a measurement of "which tokens, when penalized, reduce measured toxicity."
2. **The 8.9M agreed-upon tokens carry the signal; the 11.1M idiosyncratic
   tokens are noise.** If true, the paper is leaving effect on the table
   — the right method is proxy ensembling: use the intersection across
   multiple small proxies, which would be much cheaper than any single
   large proxy and more robust.
3. **Each proxy captures a different toxicity-generating mechanism.** If
   true, ensembling would still improve but by composition rather than
   agreement — different proxies would catch different kinds of toxic
   content.

Option (1) is the most consequential for safety reasoning, because it
means the method is defined by *what it suppresses sufficient to lower
the Detoxify score*, not by *what is causally toxic*.

### C. The toxicity "direction" is gated, not deleted

§6.2 feature steering. Base Pythia-1B: scaling the toxicity direction
from −5 to +5 linearly raises EMT 0.47 → 0.69. IF-Guide Pythia-1B:
scaling in the **positive** direction has no effect on EMT; scaling in
the **negative** direction raises EMT to **1.0**.

The paper reads this as "IF-Guide learns a direction that suppresses
toxicity." The stronger reading the same measurements support: **IF-Guide
moved the operating point of the toxicity circuit to its lower saturation
boundary; it did not delete it.** Positive scaling can't push EMT further
down because it's already at floor. Negative scaling reactivates the
underlying toxic circuit to EMT = 1.0.

Calling this result "deletion" requires calling a floor-at-zero response
"deleted," *and* explaining why a coherent toxicity direction can still
be computed from the IF-Guide model's own activations at all. The
direction in §6.2 is constructed via difference-in-means on 5k toxic and
5k non-toxic prompts *run through the IF-Guide model*. If the
toxicity-generating circuit were actually deleted from that model, you
should not get a stable direction whose negation reconstitutes maximal
toxicity on held-out prompts. You do. The circuit is present, measurable,
identifiable in activation space, and reachable from inference-time
intervention.

The paper treats this as evidence of successful suppression. On the same
measurements, it is evidence of durable retention — and the two readings
are not equally weighted. The retention reading fits every row of the
data; the deletion reading requires reinterpreting the direction's
continued existence away.

### D. The 4,648-record query/test contamination

From the data files (verified in-code via `data/RTP/`):

```
|query toxic|     = 19,785
|query non-toxic| = 64,327
|test toxic|      =  5,000
|test non-toxic|  =  5,000
query ∩ test      =  4,648 records (!)
```

The paper says test examples are "distinct from those in the influence
computation." They're distinct on label×record keys (a record used as
"toxic query" never appears as "toxic test") but *not* on raw records:
4,648 RTP records are reused across the split, with different Detoxify-
determined labels because the thresholds differ (0.75 / 0.25 for queries
vs. 0.5 for test).

This is not a classical train/test violation — the training data is
OpenWebText, not RTP — but the **query signal that selected the training
tokens** is drawn from a pool that has a 46% overlap with the evaluation
prompt pool at the record level. The method is attributed with and
evaluated against text that overlaps in surface content, mediated only
by threshold placement.

### E. The eval harness ignores embeddings and the LM head

`src/utils/tasks.py:64-94` — EK-FAC-tracked modules for attribution are
MLP (`dense_h_to_4h`, `dense_4h_to_h`) and attention (`query_key_value`,
`dense`) only. No embedding matrix, no LM head. Any toxicity signal that
lives in the embedding of a specific toxic token (which is where the
simplest interpretable toxic-token signals live — slurs are
embedding-close neighbours) is invisible to the attribution path. The
method cannot identify "the model learned a toxic word embedding" as a
cause. It can only identify learned-circuit changes in the body of the network.

### F. Capability regression is understated

Appendix F, Table 13:
- Base Pythia-2.8B: 87.5% accuracy on toxic classification, 27% on non-toxic.
- IF-Guide Pythia-2.8B: 76.7% accuracy on toxic, 34% on non-toxic.

The paper's reading: "comparable performance." The arithmetic: **toxicity
recognition drops 10.8 percentage points absolute (12% relative)**. The
proposed explanation — "prompt template biases toward the toxic label" —
doesn't explain why *IF-Guide* is worse at recognizing toxic content than
the base, because the same template bias applies to both.

A simpler explanation: the suppression objective unlearns patterns
needed to both *produce* and *recognize* toxic content, because they
share representations. If true, IF-Guide's safety story contains a
capability tax that the paper is not characterizing. For a security
engineer applying detoxified models to content moderation, threat
analysis, or adversarial text understanding, this is the number that
matters most.

### G. Narrow stability envelope

- λ ∈ {0, 0.5, 1}: trainable. λ ∈ {1.5, 2}: EMT/TP > 0.80, training degenerates to token repetition.
- L (token budget): 5M–20M works. L = 25M (2.5%) degrades PPL and LAMBADA.
- τ_tox = 99th percentile best; 95, 97.5, and 99.5 worse. 99.5 is "too conservative."
- Context window w: 1 is best, 0 and 2 both lose effectiveness.

Every knob is at the edge of a cliff. This isn't necessarily a bug, but
it's worth naming: the operating envelope is tight, hyperparameter
transfer across corpora / tokenizers / fine-tuning regimes is not
tested, and the method's reported effects are *point estimates* at a
single tuned configuration.

---

## 6. What's Suspicious in the Code (minor, but worth knowing)

1. **`num_train_epochs` default is 3 in `train.py`; paper says 4.** The
   script sets 4 explicitly so reproduction is fine, but the Python
   default drifted — a silent reproducibility hazard for anyone running
   `python train.py` without the script.
2. **`context_window=5` in `build_toxic_token_mask`'s signature default,
   `w=1` in CLI/paper.** Dead code path from exploration, visible drift.
3. **`metric_for_best_model="eval_loss"` with an all-False mask on eval**
   (`train.py:200`) selects checkpoints by unmodified NLL. The best
   checkpoint by this metric is not necessarily the best by the actual
   training objective — selection pressure is partly toward
   lower-NLL, less-aggressively-suppressed states.
4. **Batch-mean loss with per-token weights** (`training.py:97`). The
   effective λ at the per-batch level varies with batch composition. On
   a batch that happens to contain an unusually high density of toxic
   tokens, the net batch loss can go substantially negative, producing
   an outsized optimizer step. This likely contributes to the λ > 1
   instability.
5. **The custom Kronfluence fork isn't inspected here.** The actual
   differential-influence math — `−(g_tox − g_safe)ᵀ · H̃⁻¹ · ∇L(x_i)` —
   is delegated to `analyzer.compute_pairwise_scores(..., query_dataset=tox,
   negative_query_dataset=safe, ...)` which lives at
   [github.com/ztcoalson/kronfluence](https://github.com/ztcoalson/kronfluence).
   `score_args.aggregate_query_gradients = True` confirms mean-pooling
   matches paper Eq. 6. But the exact handling of the "negative" dataset
   (sign, normalization, Hessian sharing) should be read before any
   reproduction that departs from the paper's setup.

---

## 7. Probes We Could Actually Run (M5 Pro–class budget)

Ordered from cheapest-and-most-cutting to most-expensive. Each probe
specifies: hypothesis, protocol, falsification criterion, compute
estimate.

### Probe 1 · Classifier triangulation on the 10× claim (1 day, cheapest)

**Hypothesis.** IF-Guide's headline reduction is a joint function of
`(method, classifier)` more than a function of `method` alone. The
effect inflates when the same classifier (Detoxify) is on both ends of
the pipeline.

**Protocol.**
- Use the released IF-Guide and base checkpoints (pull from HuggingFace;
  ztcoalson's org should host them, or we train Pythia-160M ourselves
  in ~8h).
- Generate completions on 10k RTP prompts × 25 completions (nucleus p=0.9),
  reproducing the paper's setup exactly.
- Re-score **all** completions with four classifiers:
  - Detoxify (reproduce paper baseline)
  - Perspective API (independent lineage, Google)
  - Llama Guard 3 (Meta, different family and training data)
  - OpenAI moderation endpoint (independent)
  - Optional: a locally-trained ToxiGen-RoBERTa checkpoint
- Compute EMT and TP under each classifier.

**Falsification.** If IF-Guide's reduction stays within ±20% of the
reported Detoxify effect across all four, the 10× claim is classifier-
robust. If it drops more than 2× under non-Detoxify classifiers, the
circularity is confirmed as load-bearing.

**Compute.** Inference-only. On M5 Pro: generation takes a few hours;
all classifier scoring is API-bound or small-model inference.
End-to-end: ~1 day wall clock. No training required if checkpoints
are available.

**Why this one first.** It is the probe that costs the least and
collapses the most of the paper's claims if it fails.

### Probe 2 · LoRA re-toxification (1-2 days)

**Hypothesis.** IF-Guide's suppression is reversible by a small
fine-tune, because the toxicity circuit is gated (§5.C), not deleted.
If so, IF-Guide gives *initial* safety but not *durable* safety against
an adversary with any fine-tuning access — which is a well-known
pattern for post-hoc alignment methods (Qi et al. 2024).

**Protocol.**
- Take IF-Guide Pythia-1B (released checkpoint).
- Attach a rank-8 LoRA to attention projections.
- Fine-tune on 1k RTP-toxic completions (Detoxify > 0.9).
- Measure EMT at steps {0, 50, 100, 200, 500, 1000}.
- Plot the re-toxification curve.

**Falsification.** If EMT doesn't rise above 0.2 within 1k steps, IF-Guide
is more durable than typical post-hoc alignment. If it rises above
pre-training baseline within 100 steps, the suppression is skin-deep.

**Compute.** LoRA on Pythia-1B is MPS-friendly. Hours. Datasets are ~1MB.

### Probe 3 · Reactivation via activation addition (1 day)

**Hypothesis.** The IF-Guide model's toxicity direction is reachable
from activation space without any weight update, because the circuit
exists (§5.C). An adversary with inference-time hook access — or who
can distill into a model they then LoRA-tune — can re-activate toxicity.

**Protocol.**
- Reproduce the paper's §6.2 steering experiment with more scaling
  resolution and multiple hook points (not just last layer last token).
- Additionally: compute the toxicity direction on the *base* model, use
  it as an inference-time steering vector on the *IF-Guide* model, test
  if this out-of-distribution direction reactivates toxicity.
- Vary scaling from −20 to +20 in steps of 1.

**Falsification.** If negative scaling saturates at EMT=1.0 across hook
points, the circuit exists and is addressable. If only the last-layer
last-token position shows reactivation, the paper's narrow probe was
coincidentally the right one. If base-derived directions have zero
effect on IF-Guide, the method really did delete the circuit.

**Compute.** 200 prompts × 40 scaling values × 3 hook points × 2 models.
~24k generations. Hours. No training.

### Probe 4 · LDNOOBW-sanitization with the IF-Guide suppression loss (fair baseline) (~2 days)

**Hypothesis.** A substantial fraction of IF-Guide's attribution signal
coincides with literal profanity and slur tokens — i.e., tokens the
LDNOOBW word list would also catch. The paper's Word-Filtering baseline
*removes documents* rather than *suppresses tokens with the training
objective*, so it underperforms for a reason unrelated to attribution
quality. A fair baseline: LDNOOBW ± context_window, fed through the
same suppression loss.

**Protocol.**
- Tokenize OWT (reuse the paper's `ztcoalson/openwebtext-gptneox-2048-chunks`).
- Build a mask of all tokens that decode to words in LDNOOBW + w=1 context.
- Cap at 20M tokens.
- Train Pythia-160M with `SuppressToxicTokensTrainer`, λ=1.
- Evaluate EMT/TP on RTP (Detoxify + at least one triangulation classifier
  from Probe 1).

**Falsification.** If LDNOOBW-suppression recovers ≥80% of IF-Guide's
Pythia-160M toxicity reduction, the method's attribution adds marginal
value over a cheap keyword list applied through the same loss. If it
recovers <50%, the attribution signal is doing substantive work beyond
lexical matching.

**Compute.** One Pythia-160M training run from scratch — paper reports
their training setup fits in ~1 day on 8×A40. On M5 Pro: 1-3 days wall
clock, feasible but not trivial. Alternative: fine-tune only (start from
Pythia-160M pretrained), which collapses to Probe 2's budget.

### Probe 5 · Capability regression on security-relevant tasks (~1 day)

**Hypothesis.** The 10.8pp drop in toxicity recognition (§5.F) is the
tip; IF-Guide trades generation capability for suppression capability
across any task that requires the model to reason about norm-violating
content. This is the result that would land in the Carlini/Anthropic
register because it formalizes "alignment-tax on security-relevant
downstream use."

**Protocol.**
- Load IF-Guide Pythia-2.8B and base Pythia-2.8B.
- Evaluate zero-shot (or 5-shot) on:
  - ToxiGen (balanced classification)
  - CVE description → severity classification (CVSS)
  - A small LegalBench subset (argumentation, conflicting claims)
  - MMLU safety/ethics subsets (subset selection kept honest; no train/test cross)
  - A content-moderation-adjacent benchmark (SetFit social-media moderation)

**Falsification.** If all gaps are <3pp, no meaningful tax. If 5-15pp
consistent gap across multiple benchmarks, there is a measurable
capability regression that is independent of the direct toxicity-
recognition artifact.

**Compute.** Inference only. ~1 day wall clock.

### Probe 6 · Proxy-disagreement forensics (dependent on score artifacts)

**Hypothesis.** The 44.6% proxy overlap decomposes into:
- A "universally agreed" core of literal profanity / slurs / lexicon
  that any small model catches.
- A long tail of proxy-specific attribution where each model is seeing
  different latent toxicity-generating structure.

**Protocol.** Decode the ±1-context spans for:
- `agree_set = ∩ over all four proxies` (the intersection)
- `pythia_only = Pythia-{160M, 410M, 1B}.union() \ Llama-1B`
- `llama_only  = Llama-1B \ Pythia_family.union()`
- `small_only  = Pythia-160M \ Pythia-1B`
- `large_only  = Pythia-1B \ Pythia-160M`

For each set, compute:
- Fraction of selected tokens that lie inside a word matching LDNOOBW.
- Detoxify score of the containing OWT document.
- Distribution over GPTNeoX subword types (full word vs. prefix/suffix).

**Falsification.** If `agree_set` is dominated by explicit lexicon and
proxy-specific sets are dominated by implicit/contextual spans, the
proxy claim decomposes into "small models handle the obvious; large
models handle the subtle" — and the paper's narrative of substitutability
is partially misleading.

**Compute.** Requires full score matrices for all four proxies, which
are 1B-token × ~20 modules × bf16 artifacts (hundreds of GB each).
Unlikely to be released by the authors. If they aren't, this probe
requires either regenerating scores for all four proxies (expensive —
7.5h/proxy on H100, meaningfully longer on MPS), or restricting the
analysis to a subset of the training data.

**Compute (bounded version).** Restrict to Pythia-160M vs. Pythia-1B
scores on a 100M-token sub-slice. ~3 days wall clock on M5 Pro for the
score computation.

---

## 8. Relevance to the Residency Application

(Project context: `~/Code/blog/mazze-leczzare-blog` — security engineer
prepping for the Anthropic Residency application. Deadline 2026-11-22.
Fit-for-purpose assessment, honest.)

The paper is a **Carlini-signature artifact**:

- An empirical claim that looks like "we solved X."
- Buried in the appendix, a measurement (Table 10, Appendix F) that
  partially undermines the claim.
- A method that is technically clean, computationally inventive (small
  proxy), and in a tight stability envelope.

If the residency application needs a concrete "what would you work on,"
Probes 1, 2, 3 and 5 are the ones that:

- Fit a 6-week slot.
- Land in the register of recent Carlini publications (adversarial
  robustness of alignment methods; durability of safety training;
  classifier-circular evaluation).
- Produce a paper-shaped artifact at either outcome:
  - Confirmation: "IF-Guide's safety is durable under activation
    steering and LoRA re-toxification; its effects hold across
    independent classifiers." A positive-result paper that helps
    the method land.
  - Falsification: "IF-Guide's effects are classifier-dependent,
    activation-reversible, and LoRA-removable." A negative-result
    paper that strengthens the field's methodology for evaluating
    pre-training-time alignment.

Either direction, the result is publishable. Both directions are
interesting to the team Carlini runs.

**The honest fit note.** Probe 5 (capability regression) is the one
that most directly maps to the author's declared Secure Pride register
— security engineers routinely need models that can reason about
harmful content without producing it. The current paper under-measures
this, so there is room for a clean empirical contribution.

**What this is not.** It is not an attack on the paper. The paper's
headline claims are real within their evaluation harness. The probes
here are the ones that test the harness itself.

---

## 9. Key References to Follow

1. **Grosse et al., "Studying Large Language Model Generalization with Influence Functions"** (Kronfluence paper, arXiv:2308.03296) — the foundation for the EK-FAC approximation IF-Guide builds on. If you want to understand what a "high-influence training example" actually represents, read this.
2. **Qi et al., "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!"** (arXiv:2310.03693) — the empirical result that safety training is removable with small fine-tunes. Directly motivates Probe 2.
3. **Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction"** (arXiv:2406.11717) — the activation-steering paper whose technique is used in IF-Guide's §6.2 and which Probe 3 extends.
4. **Gehman et al., "RealToxicityPrompts"** (arXiv:2009.11462) — the benchmark that defines what "toxicity" means for this paper's query set and evaluation. Re-reading this makes the Detoxify-circular issue tangible.
5. **Khaddaj et al., "Small-to-Large Generalization"** (arXiv:2502.XXXXX, cited as khaddaj2025smalltolarge) — the proxy-model paper IF-Guide builds the small-proxy claim on. Read it next to understand what claim they are extending and what's new.

---

## 10. Reproducibility Notes

- **Checkpoints.** The IF-Guide paper's released model artifacts are
  gated behind HuggingFace; check `ztcoalson/` org for Pythia-160M / 410M
  / 1B / 2.8B IF-Guide variants.
- **The custom Kronfluence fork** at `github.com/ztcoalson/kronfluence`
  is required to compute scores with the differential-influence path
  (upstream Kronfluence doesn't have the `negative_query_dataset`
  parameter).
- **Score artifacts** are large. A full 1B-token score matrix for one
  proxy is ~100s of GB. The paper's team ran on 2TB-memory boxes and
  A40/H100 clusters.
- **M5 Pro feasibility for probes 1–5**: comfortable within a
  2–4 week window, even without H100 access, by reusing released
  checkpoints and running inference-only probes.
- **M5 Pro feasibility for probe 6**: not without substantial
  storage and compute compromises; propose the bounded version.

---

**Session note.** Arxiv source in `~/.cache/knowledge/2506.01790/`.
Repo fork at `/home/user/if-guide-toxic-kitty/`. This summary lives on
branch `claude/inspiring-dijkstra-1d1th3` in the fork. Probe scripts
to be added as a follow-up, separately scoped.
