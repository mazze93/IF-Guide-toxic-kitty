"""Read the `outputs.json` files written by `src/run_toxicity_eval.py --save_outputs`.

Each entry is one prompt: {"index", "prompt", "outputs": [{"completion",
"toxicity", ...}, ...]}. The stored "toxicity" values are the paper's own
Detoxify scores, so the anchor classifier never has to be re-run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_outputs(path: Path) -> List[Dict]:
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty list of prompt entries")
    for entry in entries:
        if not entry.get("outputs"):
            raise ValueError(f"{path}: prompt {entry.get('index')} has no completions")
    return entries


def completions_per_prompt(entries: List[Dict]) -> List[List[str]]:
    return [[out["completion"] for out in entry["outputs"]] for entry in entries]


def stored_detoxify_per_prompt(entries: List[Dict]) -> List[List[float]]:
    """The Detoxify scores the paper's harness saved beside each completion."""
    try:
        return [[float(out["toxicity"]) for out in entry["outputs"]] for entry in entries]
    except KeyError as exc:
        raise ValueError("outputs file carries no stored Detoxify 'toxicity' scores") from exc


def same_prompts(a: List[Dict], b: List[Dict]) -> bool:
    """Two runs are comparable only if they answered the same prompts."""
    return [e["index"] for e in a] == [e["index"] for e in b]
