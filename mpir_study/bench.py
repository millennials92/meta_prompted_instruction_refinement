"""Throughput micro-benchmark for a served endpoint: python -m mpir_study.bench [base_url ...]"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from .data import Cell, load_p0, load_split
from .templates import eval_template


def run(urls, n=192, max_tokens=400, workers=192):
    cell = Cell("ruin_names", "ape", 42)
    examples = load_split(cell.task, cell.seed, "test")[:n]
    template, system = eval_template(cell.optimizer)
    prompt = load_p0(cell)
    clients = [OpenAI(base_url=u, api_key="EMPTY", timeout=600) for u in urls]

    def one(i):
        ex = examples[i % len(examples)]
        r = clients[i % len(clients)].chat.completions.create(
            model="Qwen/Qwen3-1.7B", temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": template.format(instruction=prompt, question=ex["question"])}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.usage.completion_tokens

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        tokens = sum(pool.map(one, range(n)))
    elapsed = time.time() - started
    print(f"{len(urls)} endpoint(s): {tokens} tokens in {elapsed:.0f}s = {tokens / elapsed:.0f} tok/s")


if __name__ == "__main__":
    run(sys.argv[1:] or ["http://localhost:8000/v1"])
