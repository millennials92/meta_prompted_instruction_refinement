# Run details

Start with the three steps in the [README](../README.md#how-to-run). They use the saved base prompts and data splits already included in the repository.

## Experiment groups

| Group | Experiment |
| --- | --- |
| `g5-fix` | Protected rubric-guided refinement (V2F) |
| `g7-v3f` | Protected rubric-free control (V3F) |
| `g7-api` | Both variants with larger refiners through OpenRouter |

For example, run the rubric-free control with:

```bash
python -m mpir_study.run_study --group g7-v3f --mseeds 4 --workers 1
```

The `g7-api` group requires `OPENROUTER_API_KEY` in `demos/.env` and a running local target server. It uses refinement seed 1 regardless of `--mseeds`. Other groups are listed in [run_study.py](../mpir_study/run_study.py).

## Published runs

The README uses refinement seed 4 for new trials. These are not the paper's reported runs. Completed runs are skipped, so commands using saved seeds may report `0 runs to do`.

To rerun published seeds, follow the [protocol](REVISION_PROTOCOL.md) in a separate working copy. Move the corresponding saved runs and generation/scoring caches aside first to avoid reusing them.

The study used vLLM 0.28.0 under WSL2. The recorded server settings include:

```bash
vllm serve Qwen/Qwen3-1.7B --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 \
  --max-num-seqs 192 --gpu-memory-utilization 0.86
```

One V2F cell exceeded the paper's 8192-token context limit. Increasing that limit changes the experimental setting.

## Outputs and configuration

- Run records: `results/revision/rounds/`.
- Generation cache: `results/revision/gen_cache/`.
- Scoring cache: `results/revision/score_cache/`.
- Published summaries: `results/revision/`.

Inspect the run JSON files for new results. Recomputing lenient analyses requires raw-output caches that are not all tracked in Git.

The default target endpoint is `http://localhost:8000/v1`; override it with `TARGET_URL`. See [client.py](../mpir_study/client.py) for other endpoints and `python -m mpir_study.run_study --help` for command options.

The README dependencies cover refinement from saved prompts. The older demo notebooks use the broader [requirements.txt](../requirements.txt) environment.
