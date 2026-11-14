"""The evaluation prompts the paper uses: the shipped RTP file and test indices."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DATA = Path(__file__).resolve().parents[2] / "data" / "RTP"


def load_test_prompts(n_toxic: int, n_nontoxic: int) -> List[Dict]:
    """First `n` prompts of each test split, as {"index", "prompt", "split"}.

    The index files hold row positions into rtp.json, the same way
    `src/utils/datasets.py` selects them.
    """
    import numpy as np

    rows = json.loads((DATA / "rtp.json").read_text())
    chosen = []
    for split, n in (("toxic", n_toxic), ("nontoxic", n_nontoxic)):
        positions = np.load(DATA / "test_indices" / f"{split}_indices.npy")[:n].tolist()
        chosen += [{"index": rows[p]["index"], "prompt": rows[p]["prompt"]["text"], "split": split}
                   for p in positions]
    return chosen
