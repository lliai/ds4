# DS4 Expert-Prune Mask Pipeline

This directory contains helper material for DS4 routed-MoE expert pruning
experiments.

The current implementation is shape-preserving: it writes a
`ds4-expert-mask-v1` JSON keep-list and the runtime avoids pruned experts during
routing. It does not physically delete tensors from the GGUF, reduce model file
size, or change `deepseek4.expert_count`. That structural path would require a
separate GGUF/layout and loader change.

## 1. Build A C4 Calibration File

The expert-prune collector consumes a text file split by visible
`DS4_IMATRIX_PROMPT` markers. Build one from Hugging Face C4 with:

```sh
python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
  --out /tmp/ds4-c4-prune.txt \
  --max-docs 512 \
  --max-chars-per-doc 8192
```

The default source is `allenai/c4`, config `en`, split `train`, loaded in
streaming mode through the optional Python `datasets` package.

If you already have C4 extracted locally as JSONL with a `text` field:

```sh
python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
  --source jsonl \
  --input /path/to/c4.jsonl \
  --out /tmp/ds4-c4-prune.txt \
  --max-docs 512
```

## 2. Collect The 7/8 Keep Mask

For Flash, `--expert-prune-keep-ratio 0.875` keeps `224/256` experts in every
layer. For Pro, it keeps `336/384`. The same command shape works for both
models after the GGUF is loaded:

```sh
./ds4 --cuda \
  -m ds4flash.gguf \
  --expert-prune-dataset /tmp/ds4-c4-prune.txt \
  --expert-prune-out /tmp/ds4-c4-keep7of8.json \
  --expert-prune-keep-ratio 0.875 \
  --expert-prune-max-tokens 131072 \
  --ctx 32768
```

The collector runs the graph prefill path on the calibration text, accumulates
per-layer OMP statistics over routed expert contributions, and writes one
sorted keep-list per layer.

## 3. Run With The Mask

Use the generated mask on CPU/CUDA runtime surfaces:

```sh
./ds4 --cuda \
  -m ds4flash.gguf \
  --expert-mask /tmp/ds4-c4-keep7of8.json \
  --nothink \
  -n 128 \
  -p "Explain the difference between structural pruning and runtime masking."
```

The mask is also exposed on `ds4-eval`, `ds4-bench`, `ds4-server`, and
`ds4-agent` as `--expert-mask FILE`. Masked inference is not currently supported
on Metal.

## 4. Sweep Keep Ratios

To find the lowest coarse keep ratio that still produces readable output, run:

```sh
python3 gguf-tools/expert-prune/sweep_expert_keep_ratios.py \
  --model ds4flash.gguf \
  --dataset gguf-tools/expert-prune/ds4-c4-prune.txt \
  --gpus 0,1,2,3,5,7 \
  --expert-prune-max-tokens 131072
```

The script generates OMP masks for `100%, 90%, ..., 10%`, assigns one ratio at
a time to the listed GPUs, then runs a short masked inference test. It also sets
a per-GPU `DS4_LOCK_FILE` under the run directory so the CLI's default
`/tmp/ds4.lock` singleton guard does not serialize different GPUs. All masks,
logs, generated text, and summaries are written under
`gguf-tools/expert-prune/sweep-runs/`.

If the calibration file does not exist yet, add `--build-c4` to build it first
with `build_c4_prune_dataset.py`.

The pass/fail check is an automatic readability smoke test, not a benchmark or
human quality evaluation. Inspect each ratio's `test.stdout.txt`, especially
near the cutoff reported in `summary.md`.
