# Milestone 1.1: Separate Tokenization and Embedding Benchmarks

## Goal

Extend Milestone 1 so the ONNX CPU FP32 benchmark measures **tokenization** and **embedding inference** separately.

This milestone keeps the same baseline target:

```text
Model: BAAI/bge-m3
Runtime: ONNX Runtime
Provider: CPUExecutionProvider
Precision: FP32
Package manager: uv
Datasets: English, Thai, mixed English/Thai
```

The purpose is to understand whether BGE-M3 benchmark performance is limited by:

```text
1. Tokenization speed
2. ONNX embedding inference speed
3. End-to-end pipeline overhead
```

---

## Scope

### Included

```text
Tokenization-only benchmark
Embedding-only benchmark
End-to-end benchmark
Thai text benchmark dataset
English text benchmark dataset
Mixed English/Thai benchmark dataset
JSONL result output
Notebook plots for throughput and latency breakdown
```

### Excluded

```text
GPU
FP16
INT8
OpenVINO
Apple Silicon MPS/CoreML
Docker
Kubernetes
Prometheus exporter
```

---

## Key Metrics

Milestone 1.1 adds separate metrics for each stage.

```text
tokenize_tokens_per_sec
embedding_tokens_per_sec
end_to_end_tokens_per_sec

tokenize_items_per_sec
embedding_items_per_sec
end_to_end_items_per_sec

tokenize_latency_ms_avg
tokenize_latency_ms_p50
tokenize_latency_ms_p95

embedding_latency_ms_avg
embedding_latency_ms_p50
embedding_latency_ms_p95

end_to_end_latency_ms_avg
end_to_end_latency_ms_p50
end_to_end_latency_ms_p95
```

---

## Token Counting Rule

Use the BGE-M3 tokenizer output only.

Do **not** use PyThaiNLP or other Thai word segmentation tools for token counting in this milestone.

For padded batches, count real model tokens with:

```python
real_tokens = int(encoded["attention_mask"].sum())
```

This avoids counting padding tokens.

---

## Result Schema

Each benchmark run should append one JSON object to:

```text
results/onnx_cpu_fp32_stage_breakdown.jsonl
```

Example schema:

```json
{
  "model": "BAAI/bge-m3",
  "runtime": "onnxruntime",
  "provider": "CPUExecutionProvider",
  "device": "cpu",
  "precision": "fp32",
  "dataset": "mixed",
  "language": "mixed",
  "batch_size": 32,
  "max_length": 512,
  "target_words": 256,
  "warmup": 5,
  "batches": 20,

  "total_items": 640,
  "total_tokens": 327680,
  "avg_tokens_per_item": 512.0,

  "tokenize_tokens_per_sec": 1200000.0,
  "embedding_tokens_per_sec": 250000.0,
  "end_to_end_tokens_per_sec": 210000.0,

  "tokenize_items_per_sec": 2200.0,
  "embedding_items_per_sec": 480.0,
  "end_to_end_items_per_sec": 410.0,

  "tokenize_latency_ms_avg": 3.1,
  "tokenize_latency_ms_p50": 2.9,
  "tokenize_latency_ms_p95": 4.2,

  "embedding_latency_ms_avg": 48.0,
  "embedding_latency_ms_p50": 47.2,
  "embedding_latency_ms_p95": 55.0,

  "end_to_end_latency_ms_avg": 55.0,
  "end_to_end_latency_ms_p50": 54.1,
  "end_to_end_latency_ms_p95": 62.0
}
```

---

## Updated File Layout

```text
bge-m3-perf/
  pyproject.toml
  uv.lock
  README.md
  scripts/
    bench_onnx_cpu_fp32.py
    bench_onnx_cpu_fp32_stage_breakdown.py
  notebooks/
    plot_onnx_cpu_fp32.ipynb
    plot_onnx_cpu_fp32_stage_breakdown.ipynb
  results/
    .gitkeep
  models/
    bge-m3-fp32/
      model.onnx
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
      config.json
```

---

## Task 1: Add Milestone 1.1 Script

Create:

```text
scripts/bench_onnx_cpu_fp32_stage_breakdown.py
```

The script should include three benchmark phases:

```text
1. Tokenization-only
2. Embedding-only
3. End-to-end
```

### Tokenization-only phase

Measure only:

```python
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=args.max_length,
    return_tensors="np",
)
```

Output:

```text
tokenize_tokens_per_sec
tokenize_items_per_sec
tokenize_latency_ms_avg
tokenize_latency_ms_p50
tokenize_latency_ms_p95
```

### Embedding-only phase

Tokenize once outside the timing loop, then measure only:

```python
session.run(None, inputs)
```

Output:

```text
embedding_tokens_per_sec
embedding_items_per_sec
embedding_latency_ms_avg
embedding_latency_ms_p50
embedding_latency_ms_p95
```

### End-to-end phase

Measure the full pipeline:

```text
tokenize -> make ONNX inputs -> session.run
```

Output:

```text
end_to_end_tokens_per_sec
end_to_end_items_per_sec
end_to_end_latency_ms_avg
end_to_end_latency_ms_p50
end_to_end_latency_ms_p95
```

---

## Task 2: Dataset Modes

Add a `--dataset` argument:

```bash
--dataset en
--dataset th
--dataset mixed
```

### English dataset

Example text pattern:

```text
prometheus metric http_request_duration_seconds_bucket service api latency histogram route status code namespace pod thanos query frontend cache alert dashboard runbook
```

### Thai dataset

Example text pattern:

```text
เมตริก prometheus สำหรับวัดเวลา latency ของ api และตรวจสอบสถานะ service namespace pod dashboard alert runbook
```

### Mixed dataset

Example text pattern:

```text
prometheus metric latency ของ api service ตรวจสอบ dashboard alert และ runbook สำหรับทีม platform
```

---

## Task 3: Script Interface

The script should support:

```bash
uv run python scripts/bench_onnx_cpu_fp32_stage_breakdown.py \
  --model-dir models/bge-m3-fp32 \
  --dataset mixed \
  --batch-size 32 \
  --max-length 512 \
  --target-words 256 \
  --warmup 5 \
  --batches 20 \
  --out results/onnx_cpu_fp32_stage_breakdown.jsonl
```

---

## Task 4: Smoke Test

Run:

```bash
uv run python scripts/bench_onnx_cpu_fp32_stage_breakdown.py \
  --model-dir models/bge-m3-fp32 \
  --dataset th \
  --batch-size 4 \
  --max-length 128 \
  --target-words 64 \
  --warmup 1 \
  --batches 2 \
  --out results/onnx_cpu_fp32_stage_breakdown.jsonl
```

Acceptance criteria:

```text
Script exits 0
JSONL file is created
Result includes tokenization-only metrics
Result includes embedding-only metrics
Result includes end-to-end metrics
Result includes dataset=th
```

---

## Task 5: Small Benchmark Matrix

Run:

```bash
rm -f results/onnx_cpu_fp32_stage_breakdown.jsonl

for dataset in en th mixed; do
  for bs in 1 8 16 32; do
    for len in 32 128 512; do
      uv run python scripts/bench_onnx_cpu_fp32_stage_breakdown.py \
        --model-dir models/bge-m3-fp32 \
        --dataset "$dataset" \
        --batch-size "$bs" \
        --max-length "$len" \
        --target-words "$len" \
        --warmup 5 \
        --batches 20 \
        --out results/onnx_cpu_fp32_stage_breakdown.jsonl
    done
  done
done
```

Expected number of rows:

```text
3 datasets × 4 batch sizes × 3 token lengths = 36 rows
```

Check:

```bash
wc -l results/onnx_cpu_fp32_stage_breakdown.jsonl
```

---

## Task 6: Create Plotting Notebook

Create:

```text
notebooks/plot_onnx_cpu_fp32_stage_breakdown.ipynb
```

---

## Notebook Cell 1: Load Results

```python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

result_path = Path("../results/onnx_cpu_fp32_stage_breakdown.jsonl")

df = pd.read_json(result_path, lines=True)
df.head()
```

---

## Notebook Cell 2: Clean Table

```python
cols = [
    "dataset",
    "batch_size",
    "max_length",
    "avg_tokens_per_item",
    "tokenize_tokens_per_sec",
    "embedding_tokens_per_sec",
    "end_to_end_tokens_per_sec",
    "tokenize_items_per_sec",
    "embedding_items_per_sec",
    "end_to_end_items_per_sec",
    "tokenize_latency_ms_p95",
    "embedding_latency_ms_p95",
    "end_to_end_latency_ms_p95",
]

df[cols].sort_values(["dataset", "max_length", "batch_size"])
```

---

## Notebook Cell 3: Throughput Breakdown

```python
plot_df = df.copy()
plot_df["run"] = (
    plot_df["dataset"].astype(str)
    + ", bs=" + plot_df["batch_size"].astype(str)
    + ", len=" + plot_df["max_length"].astype(str)
)

metrics = [
    "tokenize_tokens_per_sec",
    "embedding_tokens_per_sec",
    "end_to_end_tokens_per_sec",
]

ax = plot_df.plot.bar(
    x="run",
    y=metrics,
    figsize=(16, 6),
)

ax.set_title("BGE-M3 ONNX CPU FP32: Tokenization vs Embedding Throughput")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("tokens/sec")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

---

## Notebook Cell 4: Latency Breakdown

```python
latency_metrics = [
    "tokenize_latency_ms_p95",
    "embedding_latency_ms_p95",
    "end_to_end_latency_ms_p95",
]

ax = plot_df.plot.bar(
    x="run",
    y=latency_metrics,
    figsize=(16, 6),
)

ax.set_title("BGE-M3 ONNX CPU FP32: p95 Latency Breakdown")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("p95 latency ms")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

---

## Notebook Cell 5: Tokenization Overhead Ratio

```python
df["tokenization_overhead_ratio"] = (
    df["end_to_end_latency_ms_p95"] - df["embedding_latency_ms_p95"]
) / df["end_to_end_latency_ms_p95"]

plot_df = df.copy()
plot_df["run"] = (
    plot_df["dataset"].astype(str)
    + ", bs=" + plot_df["batch_size"].astype(str)
    + ", len=" + plot_df["max_length"].astype(str)
)

ax = plot_df.plot.bar(
    x="run",
    y="tokenization_overhead_ratio",
    figsize=(16, 5),
    legend=False,
)

ax.set_title("BGE-M3 ONNX CPU FP32: Tokenization Overhead Ratio")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("ratio")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

---

## Notebook Cell 6: Dataset Comparison

```python
summary = (
    df.groupby(["dataset", "max_length", "batch_size"], as_index=False)
      .agg(
          tokenize_tokens_per_sec=("tokenize_tokens_per_sec", "median"),
          embedding_tokens_per_sec=("embedding_tokens_per_sec", "median"),
          end_to_end_tokens_per_sec=("end_to_end_tokens_per_sec", "median"),
          end_to_end_latency_ms_p95=("end_to_end_latency_ms_p95", "median"),
      )
)

summary
```

---

## Notebook Cell 7: Batch Scaling by Dataset

```python
for dataset in sorted(summary["dataset"].unique()):
    subset = summary[summary["dataset"] == dataset]

    pivot = subset.pivot_table(
        index="batch_size",
        columns="max_length",
        values="embedding_tokens_per_sec",
    )

    ax = pivot.plot(figsize=(10, 5), marker="o")
    ax.set_title(f"BGE-M3 ONNX CPU FP32: Embedding tokens/sec - dataset={dataset}")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("embedding_tokens_per_sec")
    plt.tight_layout()
    plt.show()
```

---

## Definition of Done

```text
Milestone 1.1 is complete when:

- scripts/bench_onnx_cpu_fp32_stage_breakdown.py exists.
- Script supports --dataset en|th|mixed.
- Script measures tokenization-only throughput.
- Script measures embedding-only throughput.
- Script measures end-to-end throughput.
- Script writes JSONL output.
- Result includes throughput and latency for all three stages.
- Notebook loads JSONL with pandas.
- Notebook plots throughput breakdown.
- Notebook plots p95 latency breakdown.
- Notebook plots tokenization overhead ratio.
- Notebook compares English, Thai, and mixed datasets.
```

---

## Interpretation Guide

Use these rules when reading results:

```text
If tokenize_tokens_per_sec is much higher than embedding_tokens_per_sec:
  ONNX model inference is the bottleneck.

If tokenize_tokens_per_sec is close to end_to_end_tokens_per_sec:
  tokenization or Python overhead may be the bottleneck.

If Thai has lower tokenize_tokens_per_sec than English:
  Thai text tokenization may be more expensive for this tokenizer.

If embedding_tokens_per_sec is similar across en/th/mixed at the same token count:
  model inference cost is mostly token-count dependent, not language dependent.

If short texts have low end-to-end throughput:
  overhead dominates; increase batch size.

If long texts have low embedding throughput:
  transformer inference dominates; optimize runtime or hardware.
```

---

## Next Milestone

```text
Milestone 1.2:
  Add machine metadata to benchmark harness (CPU model, cores, flags, memory, runtime, container, Kubernetes, GPU placeholder).

Milestone 1.3:
  Add benchmark validation and smoke test.

Milestone 1.4:
  Add ONNX CPU INT8 comparison using the same metadata schema.

Milestone 1.5:
  Add ONNX CUDA FP32/FP16 with GPU metadata enabled.
```
