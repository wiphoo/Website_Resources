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
uv-based Python project (src/ for scripts + models, notebooks/ for analysis)
ONNX Runtime CPUExecutionProvider
FP32 ONNX model only
Synthetic BGE-M3-like observability texts (English, Thai, mixed)
JSONL result output
Notebook loading and plotting
Stage-separated benchmarks: tokenization, embedding, end-to-end
Machine metadata capture (CPU, memory, runtime, container, Kubernetes, GPU placeholder)
Metrics:
  - tokenize_tokens_per_sec
  - embedding_tokens_per_sec
  - end_to_end_tokens_per_sec
  - tokenize_items_per_sec
  - embedding_items_per_sec
  - end_to_end_items_per_sec
  - tokenize_latency_ms_avg / p50 / p95
  - embedding_latency_ms_avg / p50 / p95
  - end_to_end_latency_ms_avg / p50 / p95
  - machine_metadata (nested object)
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
benchmarks/bge_m3/
├── README.md                         # project overview
├── SPEC.md                          # this spec
├── archived/                        # legacy notebooks (pre-machine_metadata schema)
│   └── plot_onnx_cpu_fp32.ipynb
├── notebooks/                        # independent uv project for analysis
│   ├── pyproject.toml
│   └── plot_onnx_cpu_fp32.ipynb   # active notebook (machine_metadata-aware)
├── results/                         # benchmark JSONL output
│   └── onnx_cpu_fp32.jsonl
└── src/                             # main uv project (scripts + models)
    ├── .python-version
    ├── pyproject.toml
    ├── uv.lock
    ├── models/bge-m3-fp32/          # exported ONNX model + tokenizer
    │   ├── model.onnx
    │   ├── model.onnx_data
    │   ├── config.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── special_tokens_map.json
    └── scripts/
        ├── bench_onnx_cpu_fp32.py            # milestone 1 benchmark
        ├── bench_onnx_cpu_fp32_stage_breakdown.py  # milestone 1.1 stage-separated
        ├── machine_metadata.py                # milestone 1.2 metadata collection
        └── validation.py                     # milestone 1.3 validation
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
4. Pre-tokenize once outside timing loop for embedding-only measurements.
5. Measure three phases per batch: tokenization, embedding, end-to-end.
6. Count real tokens with attention_mask.sum() for each batch.
7. Run warmup batches before measured batches.
8. Write one JSON object to ../results/onnx_cpu_fp32.jsonl (from src/ → project root → results/).
9. Include machine_metadata from collect_machine_metadata() in result.
```

The output should be JSONL because it is easy to append multiple benchmark runs and easy to load with pandas using `read_json(..., lines=True)`.

```python
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from machine_metadata import collect_machine_metadata


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
    parser.add_argument("--out", default="../results/onnx_cpu_fp32.jsonl")
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

    # Pre-tokenize once (outside timing) for embedding-only phase
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="np",
    )
    real_tokens = int(encoded["attention_mask"].sum())
    inputs = make_inputs(session, encoded)

    # Warmup
    for _ in range(args.warmup):
        tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="np")
        session.run(None, inputs)

    tokenize_latencies = []
    embedding_latencies = []
    e2e_latencies = []
    total_items = 0
    total_tokens = 0

    for _ in range(args.batches):
        # Tokenization phase
        tok_start = time.perf_counter()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        real_tokens = int(encoded["attention_mask"].sum())
        inputs = make_inputs(session, encoded)
        tok_elapsed = time.perf_counter() - tok_start

        # Embedding phase (pre-tokenized input)
        emb_start = time.perf_counter()
        session.run(None, inputs)
        emb_elapsed = time.perf_counter() - emb_start

        # End-to-end phase
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
        session.run(None, inputs)
        e2e_elapsed = time.perf_counter() - e2e_start

        tokenize_latencies.append(tok_elapsed)
        embedding_latencies.append(emb_elapsed)
        e2e_latencies.append(e2e_elapsed)
        total_items += args.batch_size
        total_tokens += real_tokens

    tokenize_time = sum(tokenize_latencies)
    embedding_time = sum(embedding_latencies)
    e2e_time = sum(e2e_latencies)

    result = {
        "model": "BAAI/bge-m3",
        "runtime": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "precision": "fp32",
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "target_words": args.target_words,
        "warmup": args.warmup,
        "batches": args.batches,
        "total_items": total_items,
        "total_tokens": total_tokens,
        "avg_tokens_per_item": total_tokens / total_items,
        "tokenize_tokens_per_sec": total_tokens / tokenize_time,
        "embedding_tokens_per_sec": total_tokens / embedding_time,
        "end_to_end_tokens_per_sec": total_tokens / e2e_time,
        "tokenize_items_per_sec": total_items / tokenize_time,
        "embedding_items_per_sec": total_items / embedding_time,
        "end_to_end_items_per_sec": total_items / e2e_time,
        "tokenize_latency_ms_avg": statistics.mean(tokenize_latencies) * 1000,
        "tokenize_latency_ms_p50": percentile(tokenize_latencies, 0.50) * 1000,
        "tokenize_latency_ms_p95": percentile(tokenize_latencies, 0.95) * 1000,
        "embedding_latency_ms_avg": statistics.mean(embedding_latencies) * 1000,
        "embedding_latency_ms_p50": percentile(embedding_latencies, 0.50) * 1000,
        "embedding_latency_ms_p95": percentile(embedding_latencies, 0.95) * 1000,
        "end_to_end_latency_ms_avg": statistics.mean(e2e_latencies) * 1000,
        "end_to_end_latency_ms_p50": percentile(e2e_latencies, 0.50) * 1000,
        "end_to_end_latency_ms_p95": percentile(e2e_latencies, 0.95) * 1000,
        "machine_metadata": collect_machine_metadata(session=session),
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
        --out ../results/onnx_cpu_fp32.jsonl
```

Acceptance criteria:

```text
- Script exits 0.
- ../results/onnx_cpu_fp32.jsonl exists.
- JSON contains Thai text safely through ensure_ascii=False output handling.
- JSON contains:
  - dataset
  - tokenize_tokens_per_sec
  - embedding_tokens_per_sec
  - end_to_end_tokens_per_sec
  - tokenize_latency_ms_p95
  - embedding_latency_ms_p95
  - end_to_end_latency_ms_p95
  - machine_metadata (nested object with cpu, memory, host, runtime, container, kubernetes, gpu)
```

---

## Task 5 — Run the first CPU FP32 matrix

Keep it small for Milestone 1:

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
```

Acceptance criteria:

```bash
wc -l ../results/onnx_cpu_fp32.jsonl
```

Expected:

```text
36
```

---

## Task 6 — Notebooks

The project has two notebooks:

```text
notebooks/plot_onnx_cpu_fp32.ipynb    # active, machine_metadata-aware
archived/plot_onnx_cpu_fp32.ipynb     # legacy, pre-machine_metadata schema
```

Both use `!uv pip install` and `pandas`/`matplotlib` for analysis.

Launch the active notebook:

```bash
cd benchmarks/bge_m3
uv run jupyter notebook notebooks/plot_onnx_cpu_fp32.ipynb
```

The active notebook loads results from `../results/onnx_cpu_fp32.jsonl` (relative to `notebooks/`).

Key notebook sections:
- Machine metadata summary (CPU model, cores, AVX flags, RAM, container/K8s)
- Full metrics table with tokenize / embedding / end-to-end breakdown
- Throughput bar chart (3 metrics per run)
- p95 latency bar chart
- Tokenization overhead ratio
- Grouped summary by machine config + benchmark params
- Batch scaling analysis by dataset

The archived notebook shows the legacy schema without machine metadata.

---

## Milestone 1 Definition of Done

```text
- Repository has pyproject.toml and uv.lock in src/.
- Dependencies are installed with uv add / uv sync.
- Script is run with uv run.
- Repository has scripts/bench_onnx_cpu_fp32.py with stage-separated timing.
- Script runs ONNX Runtime with CPUExecutionProvider only.
- Script writes JSONL results with ensure_ascii=False.
- Script supports dataset=en, dataset=th, and dataset=mixed.
- Thai benchmark samples are included directly in the script.
- Result includes stage-separated metrics:
  - tokenize_tokens_per_sec, embedding_tokens_per_sec, end_to_end_tokens_per_sec
  - tokenize_items_per_sec, embedding_items_per_sec, end_to_end_items_per_sec
  - tokenize_latency_ms_p95, embedding_latency_ms_p95, end_to_end_latency_ms_p95
- Result includes machine_metadata (nested object with cpu, memory, host, runtime, container, kubernetes, gpu).
- notebooks/ directory has its own pyproject.toml.
- Notebooks use !uv pip install (no fallback).
- Active notebook loads results from ../results/onnx_cpu_fp32.jsonl.
- README documents the uv setup, smoke test, matrix command, and project layout.
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

## Next milestones

```text
Milestone 1.3 (DONE):
  Add benchmark validation — scripts/validation.py module with:
    - extract_embeddings() with mean pooling fallback
    - validate_embeddings() for shape/NaN/Inf/norm
    - validate_against_reference() for self-comparison cosine similarity
    - validate_retrieval_overlap() for self-retrieval top-k overlap
    - 10-text English/Thai/mixed validation corpus
    - Integration into bench_onnx_cpu_fp32.py with --validate CLI flag
    - Notebook Step 7: validation summary table + failed rows + norm plot

Milestone 1.4:
  Add ONNX CPU INT8 comparison using the same metadata schema.

Milestone 1.5:
  Add ONNX CUDA FP32/FP16 with GPU metadata enabled.

Milestone 1.6:
  Add Docker image and Kubernetes Job using metadata env vars.
```

---

## References

- `docs/bge_m3_benchmark/bge_m3_milestone_1_1_tokenize_embedding_split.md` — stage-separated benchmark specification
- `docs/bge_m3_benchmark/bge_m3_milestone_1_2_machine_metadata.md` — machine metadata capture specification
- uv manages Python projects with `pyproject.toml`, virtual environments, and `uv.lock`; project commands include `uv run`, `uv sync`, and `uv lock`.
- PyThaiNLP provides Thai tokenization tools, but Milestone 1 does not compare against it. The benchmark uses BGE-M3 tokenizer counts only.
