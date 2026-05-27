# Milestone 1: ONNX CPU FP32 Baseline + Notebook Plot

**Goal:** create the first working MVP that runs **BGE-M3 ONNX Runtime CPU FP32**, generates benchmark results as JSONL, then loads those results in a Python notebook and plots throughput.

This milestone should stay simple: **no Docker, no Kubernetes, no GPU, no FP16/INT8, no PyThaiNLP comparison yet**.

The tech stack for this milestone uses **uv** for Python project/dependency management. `uv` manages Python projects through `pyproject.toml`, creates `.venv`, and writes `uv.lock` when running project commands such as `uv run`, `uv sync`, or `uv lock`.

ONNX Runtime's Python API uses `InferenceSession` with execution providers such as `CPUExecutionProvider`, which fits this baseline. Token counting should use the Hugging Face tokenizer output; `attention_mask` is the right field to count real non-padding tokens in a padded batch.

For Thai support in this milestone, include Thai text samples directly in the benchmark dataset, for example:

```text
สวัสดีครับ ระบบแจ้งเตือนมี latency สูงผิดปกติ
เมตริก http_request_duration_seconds_bucket ใช้วัดเวลา request ของ API
แดชบอร์ดแสดงอัตรา error ของบริการ production
```

We **do not compare against PyThaiNLP** in Milestone 1. PyThaiNLP can tokenize Thai text into words/sentences/subwords, but for BGE-M3 throughput measurement we only need the **BGE-M3 tokenizer token count** because that is what the model actually processes.

---

## Scope

### Included

```text
uv-based Python project
ONNX Runtime CPUExecutionProvider
FP32 ONNX model only
Synthetic BGE-M3-like observability texts
Thai observability text samples
JSONL result output
Notebook loading and plotting
Metrics:
  - model_tokens_per_sec
  - e2e_tokens_per_sec
  - model_embeddings_per_sec
  - e2e_embeddings_per_sec
  - p50/p95 latency
```

### Excluded for Milestone 1

```text
CUDA / GPU
OpenVINO
Apple Silicon MPS/CoreML
FP16
INT8
Docker
Kubernetes Job
Prometheus exporter
PyThaiNLP comparison
Thai word segmentation benchmark
```

---

## Target file layout

```text
bge-m3-perf/
  pyproject.toml
  uv.lock
  README.md
  scripts/
    bench_onnx_cpu_fp32.py
  notebooks/
    plot_onnx_cpu_fp32.ipynb
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

## Task 1 — Initialize uv project

Create the project:

```bash
uv init bge-m3-perf
cd bge-m3-perf
```

Add dependencies:

```bash
uv add numpy pandas matplotlib transformers onnxruntime jupyter
```

Expected generated files:

```text
pyproject.toml
uv.lock
.venv/
```

Acceptance criteria:

```bash
uv run python -c "import onnxruntime, transformers, pandas, matplotlib; print('ok')"
```

---

## Task 2 — Prepare FP32 ONNX model directory

Milestone 1 assumes the ONNX model already exists here:

```text
models/bge-m3-fp32/model.onnx
```

The tokenizer files must also be in the same directory so the script can call:

```python
AutoTokenizer.from_pretrained("models/bge-m3-fp32")
```

Acceptance criteria:

```bash
ls models/bge-m3-fp32/model.onnx
ls models/bge-m3-fp32/tokenizer.json
```

---

## Task 3 — Create benchmark script

Create:

```text
scripts/bench_onnx_cpu_fp32.py
```

The script should:

```text
1. Load tokenizer from model dir.
2. Load ONNX model with CPUExecutionProvider.
3. Generate synthetic English + Thai observability text batches.
4. Tokenize with padding/truncation using the BGE-M3 tokenizer.
5. Count real tokens with attention_mask.sum().
6. Run warmup batches.
7. Run measured batches.
8. Write one JSON object to results/onnx_cpu_fp32.jsonl.
```

The output should be JSONL because it is easy to append multiple benchmark runs and easy to load with pandas using `read_json(..., lines=True)`.

```python
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


ENGLISH_OBSERVABILITY_TEXT = (
    "prometheus metric http_request_duration_seconds_bucket "
    "service api latency histogram route status code namespace pod "
    "thanos query frontend cache alert dashboard runbook "
)

THAI_OBSERVABILITY_TEXTS = [
    "สวัสดีครับ ระบบแจ้งเตือนมี latency สูงผิดปกติ",
    "เมตริก http_request_duration_seconds_bucket ใช้วัดเวลา request ของ API",
    "แดชบอร์ดแสดงอัตรา error ของบริการ production",
    "ระบบ Prometheus เก็บข้อมูลจาก pod และ namespace เพื่อตรวจสอบ service",
    "คู่มือ runbook อธิบายขั้นตอนการแก้ปัญหาเมื่อ API ช้า",
]


def make_texts(batch_size: int, target_words: int, dataset: str) -> list[str]:
    if dataset == "en":
        words = ENGLISH_OBSERVABILITY_TEXT.split()
        text = " ".join(words * ((target_words // len(words)) + 1))
        text = " ".join(text.split()[:target_words])
        return [f"{text} item_{i}" for i in range(batch_size)]

    if dataset == "th":
        # Thai does not use spaces between every word, so target_words is only a rough
        # text-size control here. The real measurement still comes from BGE-M3 tokens.
        joined = " ".join(THAI_OBSERVABILITY_TEXTS)
        text = " ".join([joined] * max(1, target_words // 16))
        return [f"{text} รายการ_{i}" for i in range(batch_size)]

    if dataset == "mixed":
        texts = []
        for i in range(batch_size):
            if i % 2 == 0:
                texts.append(make_texts(1, target_words, "en")[0])
            else:
                texts.append(make_texts(1, target_words, "th")[0])
        return texts

    raise ValueError(f"Unsupported dataset: {dataset}")


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = round((len(values) - 1) * p)
    return values[idx]


def make_inputs(session: ort.InferenceSession, encoded) -> dict[str, np.ndarray]:
    expected_inputs = {item.name for item in session.get_inputs()}
    inputs = {}

    if "input_ids" in expected_inputs:
        inputs["input_ids"] = encoded["input_ids"].astype(np.int64)

    if "attention_mask" in expected_inputs:
        inputs["attention_mask"] = encoded["attention_mask"].astype(np.int64)

    if "token_type_ids" in expected_inputs:
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
        else:
            inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"], dtype=np.int64)

    missing = expected_inputs - set(inputs.keys())
    if missing:
        raise RuntimeError(f"Missing required ONNX inputs: {missing}")

    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/bge-m3-fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--target-words", type=int, default=256)
    parser.add_argument("--dataset", choices=["en", "th", "mixed"], default="mixed")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--out", default="results/onnx_cpu_fp32.jsonl")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "model.onnx"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    texts = make_texts(args.batch_size, args.target_words, args.dataset)

    for _ in range(args.warmup):
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        inputs = make_inputs(session, encoded)
        session.run(None, inputs)

    model_latencies = []
    e2e_latencies = []
    total_items = 0
    total_tokens = 0

    for _ in range(args.batches):
        e2e_start = time.perf_counter()

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )

        real_tokens = int(encoded["attention_mask"].sum())
        inputs = make_inputs(session, encoded)

        model_start = time.perf_counter()
        session.run(None, inputs)
        model_elapsed = time.perf_counter() - model_start

        e2e_elapsed = time.perf_counter() - e2e_start

        model_latencies.append(model_elapsed)
        e2e_latencies.append(e2e_elapsed)
        total_items += args.batch_size
        total_tokens += real_tokens

    model_time = sum(model_latencies)
    e2e_time = sum(e2e_latencies)

    result = {
        "model": "BAAI/bge-m3",
        "runtime": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "precision": "fp32",
        "dataset": args.dataset,
        "machine": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "active_providers": session.get_providers(),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "target_words": args.target_words,
        "warmup": args.warmup,
        "batches": args.batches,
        "total_items": total_items,
        "total_tokens": total_tokens,
        "avg_tokens_per_item": total_tokens / total_items,
        "model_tokens_per_sec": total_tokens / model_time,
        "e2e_tokens_per_sec": total_tokens / e2e_time,
        "model_embeddings_per_sec": total_items / model_time,
        "e2e_embeddings_per_sec": total_items / e2e_time,
        "model_latency_ms_avg": statistics.mean(model_latencies) * 1000,
        "model_latency_ms_p50": percentile(model_latencies, 0.50) * 1000,
        "model_latency_ms_p95": percentile(model_latencies, 0.95) * 1000,
        "e2e_latency_ms_avg": statistics.mean(e2e_latencies) * 1000,
        "e2e_latency_ms_p50": percentile(e2e_latencies, 0.50) * 1000,
        "e2e_latency_ms_p95": percentile(e2e_latencies, 0.95) * 1000,
    }

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

---

## Task 4 — Run one smoke test with uv

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

Acceptance criteria:

```text
- Script exits 0.
- results/onnx_cpu_fp32.jsonl exists.
- JSON contains Thai text safely through ensure_ascii=False output handling.
- JSON contains:
  - dataset
  - model_tokens_per_sec
  - e2e_tokens_per_sec
  - model_embeddings_per_sec
  - e2e_embeddings_per_sec
  - model_latency_ms_p95
```

---

## Task 5 — Run the first CPU FP32 matrix

Keep it small for Milestone 1:

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
```

Acceptance criteria:

```bash
wc -l results/onnx_cpu_fp32.jsonl
```

Expected:

```text
36
```

---

## Task 6 — Create plotting notebook

Create:

```text
archived/plot_onnx_cpu_fp32.ipynb
```

Launch notebook with uv:

```bash
uv run jupyter notebook archived/plot_onnx_cpu_fp32.ipynb
```

### Cell 1 — Load results

```python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

result_path = Path("../results/onnx_cpu_fp32.jsonl")

df = pd.read_json(result_path, lines=True)
df.head()
```

### Cell 2 — Show clean table

```python
cols = [
    "machine",
    "runtime",
    "provider",
    "precision",
    "dataset",
    "batch_size",
    "max_length",
    "avg_tokens_per_item",
    "model_tokens_per_sec",
    "e2e_tokens_per_sec",
    "model_embeddings_per_sec",
    "e2e_embeddings_per_sec",
    "model_latency_ms_p95",
]

df[cols].sort_values(["dataset", "max_length", "batch_size"])
```

### Cell 3 — Plot model tokens/sec

```python
plot_df = df.copy()
plot_df["run"] = (
    plot_df["dataset"]
    + ", bs=" + plot_df["batch_size"].astype(str)
    + ", len=" + plot_df["max_length"].astype(str)
)

ax = plot_df.plot.bar(
    x="run",
    y="model_tokens_per_sec",
    figsize=(16, 5),
    legend=False,
)

ax.set_title("BGE-M3 ONNX CPU FP32: model tokens/sec")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("tokens/sec")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

### Cell 4 — Plot e2e tokens/sec

```python
ax = plot_df.plot.bar(
    x="run",
    y="e2e_tokens_per_sec",
    figsize=(16, 5),
    legend=False,
)

ax.set_title("BGE-M3 ONNX CPU FP32: end-to-end tokens/sec")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("tokens/sec")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

### Cell 5 — Plot embeddings/sec

```python
ax = plot_df.plot.bar(
    x="run",
    y="model_embeddings_per_sec",
    figsize=(16, 5),
    legend=False,
)

ax.set_title("BGE-M3 ONNX CPU FP32: model embeddings/sec")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("embeddings/sec")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

### Cell 6 — Pivot by dataset, token length, and batch size

```python
summary = (
    df.groupby(["dataset", "max_length", "batch_size"], as_index=False)
      .agg(
          model_tokens_per_sec=("model_tokens_per_sec", "median"),
          e2e_tokens_per_sec=("e2e_tokens_per_sec", "median"),
          model_embeddings_per_sec=("model_embeddings_per_sec", "median"),
          model_latency_ms_p95=("model_latency_ms_p95", "median"),
          avg_tokens_per_item=("avg_tokens_per_item", "median"),
      )
)

summary
```

### Cell 7 — Plot batch scaling by dataset

```python
for dataset in sorted(summary["dataset"].unique()):
    subset = summary[summary["dataset"] == dataset]
    pivot = subset.pivot_table(
        index="batch_size",
        columns="max_length",
        values="model_tokens_per_sec",
    )

    ax = pivot.plot(figsize=(10, 5), marker="o")
    ax.set_title(f"BGE-M3 ONNX CPU FP32: batch scaling, dataset={dataset}")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("model_tokens_per_sec")
    plt.tight_layout()
    plt.show()
```

### Cell 8 — Compare English vs Thai vs mixed

```python
pivot = summary.pivot_table(
    index=["max_length", "batch_size"],
    columns="dataset",
    values="model_tokens_per_sec",
)

pivot
```

---

## Milestone 1 Definition of Done

```text
- Repository has pyproject.toml and uv.lock.
- Dependencies are installed with uv add / uv sync.
- Script is run with uv run.
- Repository has scripts/bench_onnx_cpu_fp32.py.
- Script runs ONNX Runtime with CPUExecutionProvider only.
- Script writes JSONL results with ensure_ascii=False.
- Script supports dataset=en, dataset=th, and dataset=mixed.
- Thai benchmark samples are included directly in the script.
- Result includes tokens/sec, embeddings/sec, latency, dataset, batch size, and max length.
- Notebook loads JSONL using pandas.
- Notebook plots:
  - model_tokens_per_sec
  - e2e_tokens_per_sec
  - model_embeddings_per_sec
  - batch scaling by max_length and dataset
- README documents the uv setup, smoke test, and matrix command.
```

---

## Suggested commit split

```text
commit 1: initialize uv project skeleton
commit 2: add ONNX CPU FP32 benchmark script
commit 3: add English/Thai/mixed benchmark datasets
commit 4: add smoke test and small matrix commands to README
commit 5: add plotting notebook
commit 6: add sample result JSONL if acceptable for repo
```

## Next milestone after this

```text
Milestone 2:
  Add ONNX CPU INT8 comparison using the same result schema.

Milestone 3:
  Add ONNX CUDA FP32/FP16.

Milestone 4:
  Add OpenVINO CPU FP32/INT8.

Milestone 5:
  Add Docker image and Kubernetes Job.
```

---

## References

- uv manages Python projects with `pyproject.toml`, virtual environments, and `uv.lock`; project commands include `uv run`, `uv sync`, and `uv lock`.
- PyThaiNLP provides Thai tokenization tools, but Milestone 1 does not compare against it. The benchmark uses BGE-M3 tokenizer counts only.
