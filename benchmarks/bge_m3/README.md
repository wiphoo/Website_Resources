# BGE-M3 ONNX CPU FP32 Benchmarks

Benchmark BGE-M3 embedding inference using ONNX Runtime on CPU (FP32).

## Quick start

```bash
cd src
uv sync
uv run python scripts/bench_onnx_cpu_fp32.py --dataset mixed --batch-size 4 --max-length 128
```

Results are appended to `results/onnx_cpu_fp32.jsonl`.

## Prerequisites

- Python 3.11+ (see `.python-version`)
- [uv](https://github.com/astral-sh/uv)

## Downloading the model

The ONNX model binary (~2.2 GB combined) is **not** included in the repo due to GitHub's 100 MB file size limit. Export it from the source model:

```bash
uv run python scripts/export_model.py
```

Or manually:
```bash
optimum-cli export onnx --model BAAI/bge-m3 --dtype float32 models/bge-m3-fp32
```

This produces the required files under `models/bge-m3-fp32/`:
- `model.onnx` (~433 KB)
- `model.onnx_data` (~2.2 GB)
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json`

## Setup

```bash
cd src
uv sync          # installs all dependencies from uv.lock
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `transformers`, `onnxruntime`, `jupyter`

## Running benchmarks

### Smoke test

```bash
uv run python scripts/bench_onnx_cpu_fp32.py \
  --model-dir models/bge-m3-fp32 \
  --dataset mixed \
  --batch-size 4 \
  --max-length 128 \
  --target-words 64 \
  --warmup 1 \
  --batches 2 \
  --out results/onnx_cpu_fp32.jsonl
```

### Full 36-run matrix

```bash
rm -f results/onnx_cpu_fp32.jsonl

for dataset in en th mixed; do
  for bs in 1 8 16 32; do
    for len in 32 128 512; do
      uv run python scripts/bench_onnx_cpu_fp32.py \
        --model-dir models/bge-m3-fp32 \
        --dataset "$dataset" \
        --batch-size "$bs" \
        --max-length "$len" \
        --target-words "$len" \
        --warmup 5 \
        --batches 20 \
        --out results/onnx_cpu_fp32.jsonl
    done
  done
done

echo "Runs: $(wc -l < results/onnx_cpu_fp32.jsonl)"
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `models/bge-m3-fp32` | Path to ONNX model + tokenizer directory |
| `--dataset` | `mixed` | One of `en`, `th`, `mixed` |
| `--batch-size` | `32` | Batch size |
| `--max-length` | `512` | Max token length |
| `--target-words` | `256` | Approximate word count for synthetic text |
| `--warmup` | `5` | Warmup batches |
| `--batches` | `20` | Measured batches |
| `--out` | `results/onnx_cpu_fp32.jsonl` | Output path (JSONL, appends) |

## Output format

Each run writes one JSON object per line (JSONL):

```json
{
  "model": "BAAI/bge-m3",
  "runtime": "onnxruntime",
  "provider": "CPUExecutionProvider",
  "device": "cpu",
  "precision": "fp32",
  "dataset": "mixed",
  "machine": " hostname",
  "batch_size": 32,
  "max_length": 512,
  "total_items": 640,
  "total_tokens": 18640,
  "avg_tokens_per_item": 29.1,
  "model_tokens_per_sec": 58210.3,
  "e2e_tokens_per_sec": 41200.1,
  "model_embeddings_per_sec": 2000.5,
  "e2e_embeddings_per_sec": 1415.2,
  "model_latency_ms_p95": 1.82,
  "e2e_latency_ms_p95": 2.78
}
```

Key metrics:
- **model_tokens_per_sec** — tokens/sec during ONNX model inference only
- **e2e_tokens_per_sec** — tokens/sec including tokenization overhead
- **model_embeddings_per_sec** — full sequences (batches) per second, model only
- **e2e_embeddings_per_sec** — full sequences per second, end-to-end

## Viewing results

Open the notebook and load results:

```bash
uv run jupyter notebook archived/plot_onnx_cpu_fp32.ipynb
```

Or from Python:

```python
import pandas as pd
df = pd.read_json("results/onnx_cpu_fp32.jsonl", lines=True)
print(df[["dataset", "batch_size", "max_length", "model_tokens_per_sec", "e2e_tokens_per_sec"]].to_string())
```

## Project layout

```text
benchmarks/bge_m3/
├── SPEC.md                         # milestone specification
├── .gitignore
├── archived/                       # previous notebook version
│   └── plot_onnx_cpu_fp32.ipynb
├── results/                        # benchmark output (JSONL)
│   └── onnx_cpu_fp32.jsonl
└── src/                            # uv project
    ├── .python-version
    ├── pyproject.toml
    ├── uv.lock
    ├── models/bge-m3-fp32/         # exported ONNX model + tokenizer
    │   ├── model.onnx
    │   ├── model.onnx_data
    │   ├── config.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── special_tokens_map.json
    ├── scripts/
    │   ├── bench_onnx_cpu_fp32.py  # benchmark script
    │   └── export_model.py         # model export script
    └── notebooks/
        └── plot_onnx_cpu_fp32.ipynb
```

## Exporting the model

If you need to re-export the model:

```bash
uv run python scripts/export_model.py
```

## Notes

- Thai text samples are embedded directly in the benchmark script (no external Thai NLP needed).
- Results append to the output file. Use `rm` before running a fresh matrix.
- `uv.lock` is committed so runs are reproducible across machines.