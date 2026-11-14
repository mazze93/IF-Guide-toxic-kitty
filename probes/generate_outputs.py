#!/usr/bin/env python3
"""Write a paper-format outputs.json for any Hugging Face causal language model.

Use `src/run_toxicity_eval.py --save_outputs` when your checkpoints are in
the repository's registry layout; that is the paper's own harness. This
helper exists for models that are not, and for the pipeline smoke run.

Decoding matches `src/run_toxicity_eval.py`: nucleus sampling with
top_p=0.9, top_k=0, 20 new tokens. Completions are scored with Detoxify
"unbiased", the model the paper's harness uses.

    python3 generate_outputs.py --model EleutherAI/pythia-70m-deduped \\
        --n-toxic 12 --n-nontoxic 12 --completions 5 --out results/base.outputs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lib.rtp import load_test_prompts

DECODING = dict(do_sample=True, top_p=0.9, top_k=0, max_new_tokens=20)


def generate(model_id: str, prompts: list, completions: int, seed: int) -> list:
    """Return, per prompt, a list of completion strings."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    results = []
    with torch.inference_mode():
        for item in prompts:
            inputs = tokenizer([item["prompt"]] * completions, return_tensors="pt", padding=True)
            out = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **DECODING)
            new_tokens = out[:, inputs["input_ids"].shape[-1]:]
            results.append(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return results


def score_with_detoxify(texts_per_prompt: list) -> list:
    from detoxify import Detoxify

    scorer = Detoxify("unbiased")
    return [[float(s) for s in scorer.predict(texts)["toxicity"]] for texts in texts_per_prompt]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--n-toxic", type=int, default=12)
    p.add_argument("--n-nontoxic", type=int, default=12)
    p.add_argument("--completions", type=int, default=25, help="per prompt; the paper uses 25")
    p.add_argument("--seed", type=int, default=1004)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    prompts = load_test_prompts(args.n_toxic, args.n_nontoxic)
    texts = generate(args.model, prompts, args.completions, args.seed)
    scores = score_with_detoxify(texts)
    entries = [
        {"index": item["index"], "prompt": item["prompt"], "split": item["split"],
         "outputs": [{"completion": t, "toxicity": s} for t, s in zip(ts, ss)]}
        for item, ts, ss in zip(prompts, texts, scores)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, indent=1))
    print(f"Wrote {args.out}: {len(entries)} prompts x {args.completions} completions from {args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
