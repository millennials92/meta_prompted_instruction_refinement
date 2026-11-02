"""Measure the characters-per-token ratio of the logged generations with the Qwen3 tokenizer (run inside WSL:
`python -m mpir_study.tokens`), used by revision2.cost_per_variant to convert logged characters into tokens."""
from __future__ import annotations

import glob
import json
import random

import numpy as np

from transformers import AutoTokenizer

REV = "/mnt/c/work/ethics_preserving_LLM/meta_prompted_instruction_refinement/results/revision"


def main() -> None:
    texts = []
    for path in sorted(glob.glob(f"{REV}/rounds/*__r1.7b__*.json")):
        run = json.load(open(path, encoding="utf-8"))
        for x in run["rounds"]:
            texts += [t for t in (x.get("judge_text"), x.get("refine_text")) if t]
    random.Random(0).shuffle(texts)
    texts = texts[:1500]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    ratios = [len(t) / max(1, len(tok(t).input_ids)) for t in texts]
    chars = sum(len(t) for t in texts)
    tokens = sum(len(tok(t).input_ids) for t in texts)
    out = {"n": len(texts), "chars_per_token_pooled": chars / tokens, "median_ratio": float(np.median(ratios))}
    open(f"{REV}/chars_per_token.json", "w").write(json.dumps(out))
    print(out)


if __name__ == "__main__":
    main()
