# BGE-M3 ONNX CPU FP32 Benchmarks

Benchmark BGE-M3 embedding inference using ONNX Runtime on CPU (FP32).

## Quick start

```bash
cd src
uv sync
uv run python scripts/bench_onnx_cpu_fp32.py --dataset mixed --batch-size 4 --max-length 128
```

Results are appended to `../results/onnx_cpu_fp32.jsonl` (from `src/` → project root → `results/`).

## Prerequisites

- Python 3.14 (see `.python-version`)
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
  --validate \
  --out ../results/onnx_cpu_fp32.jsonl
```

### Full 36-run matrix

```bash
rm -f ../results/onnx_cpu_fp32.jsonl

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
        --out ../results/onnx_cpu_fp32.jsonl
    done
  done
done

echo "Runs: $(wc -l < ../results/onnx_cpu_fp32.jsonl)"
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
| `--out` | `../results/onnx_cpu_fp32.jsonl` | Output path (JSONL, appends) |
| `--validate` | `True` | Enable validation (shape, NaN/Inf, norm, reference, retrieval) |
| `--expected-embedding-dim` | `1024` | Expected embedding dimension |
| `--no-normalize` | `False` | Disable L2 normalization of embeddings |

## Output format

Each run writes one JSON object per line (JSONL). With `--validate` (default), each result includes performance metrics plus correctness validation:

```json
{
  "model": "BAAI/bge-m3",
  "runtime": "onnxruntime",
  "provider": "CPUExecutionProvider",
  "device": "cpu",
  "precision": "fp32",
  "dataset": "mixed",
  "batch_size": 32,
  "max_length": 512,
  "total_items": 640,
  "total_tokens": 18640,
  "avg_tokens_per_item": 29.1,
  "tokenize_tokens_per_sec": 85200.4,
  "embedding_tokens_per_sec": 58210.3,
  "end_to_end_tokens_per_sec": 41200.1,
  "tokenize_items_per_sec": 2944.8,
  "embedding_items_per_sec": 2000.5,
  "end_to_end_items_per_sec": 1415.2,
  "tokenize_latency_ms_p95": 0.96,
  "embedding_latency_ms_p95": 1.82,
  "end_to_end_latency_ms_p95": 2.78,
  "machine_metadata": { ... },
  "validation_enabled": true,
  "validation_passed": true,
  "onnx_output_names": ["last_hidden_state"],
  "onnx_output_shapes": [[10, 21, 1024]],
  "embedding_extraction_method": "mean_pooling:last_hidden_state",
  "embedding_shape": [10, 1024],
  "embedding_dim": 1024,
  "embedding_nan_count": 0,
  "embedding_inf_count": 0,
  "embedding_norm_mean": 1.0,
  "embedding_norm_min": 0.9999,
  "embedding_norm_max": 1.0001,
  "embedding_norm_std": 0.00001,
  "reference_validation_enabled": true,
  "reference_cosine_similarity_mean": 1.0,
  "reference_cosine_similarity_min": 0.9999,
  "retrieval_validation_enabled": true,
  "retrieval_top1_overlap": 1.0,
  "retrieval_top5_overlap": 1.0,
  "retrieval_top10_overlap": 1.0,
  "validation_errors": []
}
```

**Validation fields** (present when `--validate` is used):

| Field | Description |
|---|---|
| `validation_passed` | All validation checks passed |
| `onnx_output_names` | Names of ONNX model outputs |
| `onnx_output_shapes` | Shapes of ONNX model outputs |
| `embedding_extraction_method` | How embeddings were extracted (e.g. `mean_pooling:last_hidden_state`) |
| `embedding_dim` | Embedding dimension (expected 1024 for BGE-M3) |
| `embedding_nan_count` | Count of NaN values in embeddings |
| `embedding_inf_count` | Count of Inf values in embeddings |
| `embedding_norm_mean` | Mean L2 norm (should be ~1.0 if normalized) |
| `reference_cosine_similarity_mean` | Self-comparison cosine similarity mean (≥0.999 for FP32) |
| `retrieval_top10_overlap` | Self-retrieval top-10 overlap (≥0.99 for FP32) |
| `validation_errors` | List of validation errors (empty if passed) |

Key metrics:
- **tokenize_tokens_per_sec** — tokens/sec during tokenization only
- **embedding_tokens_per_sec** — tokens/sec during ONNX model inference only
- **end_to_end_tokens_per_sec** — tokens/sec including tokenization + inference
- **tokenize_items_per_sec** — batches/sec, tokenizer only
- **embedding_items_per_sec** — batches/sec, model only
- **end_to_end_items_per_sec** — batches/sec, end-to-end

Latency columns: `*_latency_ms_p95` — p95 latency in milliseconds for each stage.

## Viewing results

Open the active notebook (prefers `notebooks/` if available, falls back to `src/notebooks/`):

```bash
# From project root
cd benchmarks/bge_m3
uv run jupyter notebook notebooks/plot_onnx_cpu_fp32.ipynb
```

Or from Python:

```python
import pandas as pd

result_path = "results/onnx_cpu_fp32.jsonl"
df = pd.read_json(result_path, lines=True)

cols = [
    "dataset", "batch_size", "max_length",
    "tokenize_tokens_per_sec", "embedding_tokens_per_sec", "end_to_end_tokens_per_sec",
]
print(df[[c for c in cols if c in df.columns]].to_string())
```

## Project layout

```text
benchmarks/bge_m3/
├── README.md
├── SPEC.md                         # milestone specification
├── archived/                       # legacy notebook (pre-machine_metadata schema)
│   └── plot_onnx_cpu_fp32.ipynb
├── notebooks/                      # independent uv project for analysis
│   ├── pyproject.toml
│   └── plot_onnx_cpu_fp32.ipynb   # active notebook (machine_metadata-aware)
├── results/                        # benchmark output (JSONL)
│   └── onnx_cpu_fp32.jsonl
└── src/                            # uv project (models + scripts)
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
    └── scripts/
        ├── bench_onnx_cpu_fp32.py  # benchmark script
        ├── export_model.py          # model export script
        ├── machine_metadata.py      # machine metadata collection
        └── validation.py           # embedding validation module
```

## Notebooks

| Notebook | Path | Schema | Description |
|---|---|---|---|
| **Active** | `notebooks/plot_onnx_cpu_fp32.ipynb` | v2 (with `machine_metadata`) | Full analysis with machine metadata, groupby by CPU config |
| **Archived** | `archived/plot_onnx_cpu_fp32.ipynb` | v1 (no `machine_metadata`) | Legacy analysis, same metrics without hardware correlation |

Both notebooks use `!uv pip install` — run `uv sync` in `notebooks/` before launching for faster startup:

```bash
cd notebooks
uv sync
uv run jupyter notebook plot_onnx_cpu_fp32.ipynb
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
- `notebooks/` has its own `pyproject.toml` — it does not share `src/`'s environment.
- Active notebook path to results: `../results/onnx_cpu_fp32.jsonl`
- Archived notebook path to results: `../../results/onnx_cpu_fp32.jsonl`
