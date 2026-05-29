# Milestone 1.3: Validation for BGE-M3 ONNX CPU FP32 Benchmark

## Goal

Add a validation layer to the BGE-M3 benchmark harness so every runtime result is checked for correctness, not only speed.

This milestone focuses on validating the current baseline:

```text
Model: BAAI/bge-m3
Runtime: ONNX Runtime
Provider: CPUExecutionProvider
Precision: FP32
Package manager: uv
Benchmark mode:
  - tokenization-only
  - embedding-only
  - end-to-end
Datasets:
  - English
  - Thai
  - mixed English/Thai
```

The purpose is to ensure the benchmark output is a valid embedding and can be trusted before adding:

```text
INT8
FP16
CUDA
OpenVINO
Apple Silicon
Kubernetes
```

---

## Why validation is needed

A benchmark can be fast but wrong.

Common failure cases:

```text
ONNX export returns token embeddings instead of sentence embeddings.
Pooling is missing.
Output dimension is unexpected.
Output contains NaN or Inf.
Output is not normalized.
FP16/INT8 later changes embedding quality.
Runtime silently changes output shape.
Thai/mixed inputs trigger unexpected tokenization behavior.
```

Milestone 1.3 adds validation so every benchmark result includes both:

```text
performance metrics
correctness metrics
```

---

## Scope

### Included

```text
ONNX output name inspection
ONNX output shape inspection
sentence_embedding detection
token_embeddings / last_hidden_state pooling fallback
mean pooling with attention_mask
L2 normalization
NaN / Inf validation
embedding dimension validation
embedding norm validation
reference output capture
cosine similarity validation against reference
small retrieval top-k overlap validation
JSONL schema update
notebook validation summary
```

### Excluded

```text
INT8 validation
FP16 validation
CUDA validation
OpenVINO validation
Apple Silicon validation
Large retrieval benchmark
Production vector database integration
```

---

## Validation design

Validation happens after a benchmark run produces ONNX outputs.

The validation layer should answer:

```text
1. Did the model return any output?
2. What are the output names?
3. What are the output shapes?
4. Can we extract one dense embedding per input item?
5. Is embedding dimension expected?
6. Does the embedding contain NaN or Inf?
7. Are embeddings L2-normalized?
8. Are repeated runs stable?
9. Does output match a reference path closely enough?
10. Does retrieval top-k stay stable for a small test corpus?
```

---

## Expected BGE-M3 dense embedding shape

BGE-M3 dense embeddings are expected to be:

```text
batch_size × 1024
```

So for a batch of 32:

```text
(32, 1024)
```

The benchmark should not hard-fail immediately if the raw ONNX output is token-level, for example:

```text
batch_size × sequence_length × hidden_dim
```

Instead, it should apply pooling and produce final dense embeddings:

```text
batch_size × hidden_dim
```

Then validate:

```text
embedding_dim == 1024
```

---

## Result schema additions

Add these fields to each JSONL result row:

```json
{
  "validation_enabled": true,
  "validation_passed": true,

  "onnx_output_names": ["sentence_embedding"],
  "onnx_output_shapes": [[32, 1024]],

  "embedding_extraction_method": "sentence_embedding",
  "embedding_shape": [32, 1024],
  "embedding_dim": 1024,

  "embedding_nan_count": 0,
  "embedding_inf_count": 0,

  "embedding_norm_mean": 1.0,
  "embedding_norm_min": 0.9999,
  "embedding_norm_max": 1.0001,
  "embedding_norm_std": 0.00001,

  "reference_validation_enabled": true,
  "reference_cosine_similarity_mean": 0.9999,
  "reference_cosine_similarity_min": 0.9998,
  "reference_cosine_similarity_p01": 0.9998,

  "retrieval_validation_enabled": true,
  "retrieval_top1_overlap": 1.0,
  "retrieval_top5_overlap": 1.0,
  "retrieval_top10_overlap": 1.0,

  "validation_errors": []
}
```

If validation fails:

```json
{
  "validation_enabled": true,
  "validation_passed": false,
  "validation_errors": [
    "embedding_dim expected 1024 but got 768"
  ]
}
```

---

## Validation thresholds

For Milestone 1.3 FP32 CPU baseline:

```text
embedding_dim == 1024
embedding_nan_count == 0
embedding_inf_count == 0
embedding_norm_mean between 0.95 and 1.05 if normalization is enabled
reference_cosine_similarity_mean >= 0.999
reference_cosine_similarity_min >= 0.995
retrieval_top10_overlap >= 0.99
```

For later INT8/FP16 milestones, thresholds can be slightly relaxed:

```text
INT8:
  reference_cosine_similarity_mean >= 0.995
  retrieval_top10_overlap >= 0.98

FP16:
  reference_cosine_similarity_mean >= 0.998
  retrieval_top10_overlap >= 0.99
```

---

## Task 1: Add validation module

Create:

```text
scripts/validation.py
```

Main functions:

```python
def extract_embeddings(
    outputs: list,
    output_names: list[str],
    attention_mask,
    normalize: bool = True,
) -> tuple:
    ...

def validate_embeddings(
    embeddings,
    expected_batch_size: int,
    expected_embedding_dim: int = 1024,
    require_normalized: bool = True,
) -> dict:
    ...

def cosine_similarity_matrix(a, b):
    ...

def validate_against_reference(
    candidate_embeddings,
    reference_embeddings,
) -> dict:
    ...

def validate_retrieval_overlap(
    candidate_embeddings,
    reference_embeddings,
    top_k_values: list[int] = [1, 5, 10],
) -> dict:
    ...
```

---

## Task 2: Implement embedding extraction

The extractor should support common ONNX output patterns.

Priority order:

```text
1. sentence_embedding
2. sentence_embeddings
3. pooler_output
4. token_embeddings
5. last_hidden_state
6. fallback to first output
```

Implementation:

```python
import numpy as np


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings.astype(np.float32) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def extract_embeddings(outputs, output_names, attention_mask, normalize: bool = True):
    outputs_by_name = dict(zip(output_names, outputs))

    for name in ["sentence_embedding", "sentence_embeddings", "pooler_output"]:
        if name in outputs_by_name:
            embeddings = outputs_by_name[name].astype(np.float32)
            method = name
            if normalize:
                embeddings = l2_normalize(embeddings)
            return embeddings, method

    for name in ["token_embeddings", "last_hidden_state"]:
        if name in outputs_by_name:
            token_embeddings = outputs_by_name[name].astype(np.float32)
            embeddings = mean_pooling(token_embeddings, attention_mask)
            method = f"mean_pooling:{name}"
            if normalize:
                embeddings = l2_normalize(embeddings)
            return embeddings, method

    first = outputs[0].astype(np.float32)

    if first.ndim == 3:
        embeddings = mean_pooling(first, attention_mask)
        method = "mean_pooling:first_output"
        if normalize:
            embeddings = l2_normalize(embeddings)
        return embeddings, method

    if first.ndim == 2:
        embeddings = first
        method = "first_output"
        if normalize:
            embeddings = l2_normalize(embeddings)
        return embeddings, method

    raise ValueError(
        f"Cannot extract embeddings from outputs. "
        f"output_names={output_names}, shapes={[list(o.shape) for o in outputs]}"
    )
```

---

## Task 3: Implement embedding validation

```python
def percentile(values, p: float):
    values = sorted(values)
    idx = round((len(values) - 1) * p)
    return values[idx]


def validate_embeddings(
    embeddings: np.ndarray,
    expected_batch_size: int,
    expected_embedding_dim: int = 1024,
    require_normalized: bool = True,
) -> dict:
    errors = []

    shape = list(embeddings.shape)

    if embeddings.ndim != 2:
        errors.append(f"Expected 2D embeddings, got shape={shape}")

    if embeddings.shape[0] != expected_batch_size:
        errors.append(
            f"Expected batch size {expected_batch_size}, got {embeddings.shape[0]}"
        )

    embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else None

    if embedding_dim != expected_embedding_dim:
        errors.append(
            f"Expected embedding_dim={expected_embedding_dim}, got {embedding_dim}"
        )

    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())

    if nan_count > 0:
        errors.append(f"Found NaN values: {nan_count}")

    if inf_count > 0:
        errors.append(f"Found Inf values: {inf_count}")

    norms = np.linalg.norm(embeddings, axis=1)
    norm_mean = float(norms.mean())
    norm_min = float(norms.min())
    norm_max = float(norms.max())
    norm_std = float(norms.std())

    if require_normalized and not (0.95 <= norm_mean <= 1.05):
        errors.append(f"Embedding norm mean out of expected range: {norm_mean}")

    return {
        "validation_passed": len(errors) == 0,
        "embedding_shape": shape,
        "embedding_dim": embedding_dim,
        "embedding_nan_count": nan_count,
        "embedding_inf_count": inf_count,
        "embedding_norm_mean": norm_mean,
        "embedding_norm_min": norm_min,
        "embedding_norm_max": norm_max,
        "embedding_norm_std": norm_std,
        "validation_errors": errors,
    }
```

---

## Task 4: Add reference validation

For Milestone 1.3, reference validation can use the same ONNX CPU FP32 runtime run twice on fixed text.

This detects nondeterministic or broken extraction behavior.

Later milestones will compare:

```text
INT8 vs FP32 reference
FP16 vs FP32 reference
CUDA vs CPU reference
OpenVINO vs ONNX reference
```

Reference validation flow:

```text
1. Build fixed validation texts.
2. Run reference embeddings.
3. Run candidate embeddings.
4. Compare row-wise cosine similarity.
```

Implementation:

```python
def rowwise_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return (a * b).sum(axis=1)


def validate_against_reference(candidate_embeddings, reference_embeddings) -> dict:
    sims = rowwise_cosine_similarity(candidate_embeddings, reference_embeddings)

    errors = []

    sim_mean = float(sims.mean())
    sim_min = float(sims.min())
    sim_p01 = float(percentile(list(sims), 0.01))

    if sim_mean < 0.999:
        errors.append(f"reference_cosine_similarity_mean too low: {sim_mean}")

    if sim_min < 0.995:
        errors.append(f"reference_cosine_similarity_min too low: {sim_min}")

    return {
        "reference_validation_enabled": True,
        "reference_cosine_similarity_mean": sim_mean,
        "reference_cosine_similarity_min": sim_min,
        "reference_cosine_similarity_p01": sim_p01,
        "reference_validation_passed": len(errors) == 0,
        "reference_validation_errors": errors,
    }
```

---

## Task 5: Add retrieval overlap validation

Create a tiny validation corpus.

```python
VALIDATION_TEXTS = [
    "http_request_duration_seconds_bucket api latency histogram",
    "container_cpu_usage_seconds_total cpu usage metric",
    "kube_pod_status_phase pod status metric",
    "prometheus_tsdb_head_series time series count",
    "alert high error rate api service",
    "dashboard latency and availability overview",
    "เมตริก latency ของ api service",
    "แจ้งเตือน error rate สูงในระบบ",
    "dashboard สำหรับตรวจสอบ pod และ namespace",
    "runbook สำหรับแก้ไขปัญหา service latency",
]
```

Use self-retrieval:

```text
For each embedding, nearest neighbor should be itself.
Compare top-k neighbors between reference and candidate embeddings.
```

Implementation:

```python
def topk_indices(similarity_matrix: np.ndarray, k: int) -> list[set[int]]:
    topk = np.argsort(-similarity_matrix, axis=1)[:, :k]
    return [set(row.tolist()) for row in topk]


def validate_retrieval_overlap(candidate_embeddings, reference_embeddings, top_k_values=None):
    if top_k_values is None:
        top_k_values = [1, 5, 10]

    candidate = l2_normalize(candidate_embeddings.astype(np.float32))
    reference = l2_normalize(reference_embeddings.astype(np.float32))

    candidate_sim = candidate @ candidate.T
    reference_sim = reference @ reference.T

    result = {
        "retrieval_validation_enabled": True,
    }

    errors = []

    for k in top_k_values:
        candidate_topk = topk_indices(candidate_sim, k)
        reference_topk = topk_indices(reference_sim, k)

        overlaps = []
        for c, r in zip(candidate_topk, reference_topk):
            overlaps.append(len(c.intersection(r)) / k)

        value = float(np.mean(overlaps))
        result[f"retrieval_top{k}_overlap"] = value

        if k == 10 and value < 0.99:
            errors.append(f"retrieval_top10_overlap too low: {value}")

    result["retrieval_validation_passed"] = len(errors) == 0
    result["retrieval_validation_errors"] = errors

    return result
```

---

## Task 6: Integrate validation into benchmark script

Update:

```text
scripts/bench_onnx_cpu_fp32_stage_breakdown.py
```

Add CLI flags:

```bash
--validate
--expected-embedding-dim 1024
--no-normalize
```

Recommended defaults:

```text
--validate enabled by default
expected_embedding_dim = 1024
normalize = true
```

Pseudo-flow:

```python
outputs = session.run(None, inputs)
output_names = [output.name for output in session.get_outputs()]
output_shapes = [list(output.shape) for output in outputs]

embeddings, extraction_method = extract_embeddings(
    outputs=outputs,
    output_names=output_names,
    attention_mask=encoded["attention_mask"],
    normalize=not args.no_normalize,
)

validation_result = validate_embeddings(
    embeddings=embeddings,
    expected_batch_size=args.batch_size,
    expected_embedding_dim=args.expected_embedding_dim,
    require_normalized=not args.no_normalize,
)

result.update({
    "validation_enabled": args.validate,
    "onnx_output_names": output_names,
    "onnx_output_shapes": output_shapes,
    "embedding_extraction_method": extraction_method,
})

result.update(validation_result)
```

Reference validation can run once per benchmark invocation, not per batch, to avoid too much overhead.

---

## Task 7: Add notebook validation summary

Update:

```text
notebooks/plot_onnx_cpu_fp32_stage_breakdown.ipynb
```

Add validation table:

```python
validation_cols = [
    "dataset",
    "batch_size",
    "max_length",
    "validation_passed",
    "embedding_extraction_method",
    "embedding_shape",
    "embedding_dim",
    "embedding_nan_count",
    "embedding_inf_count",
    "embedding_norm_mean",
    "reference_cosine_similarity_mean",
    "reference_cosine_similarity_min",
    "retrieval_top1_overlap",
    "retrieval_top5_overlap",
    "retrieval_top10_overlap",
    "validation_errors",
]

df[validation_cols].sort_values(["dataset", "max_length", "batch_size"])
```

Add failure filter:

```python
failed = df[df["validation_passed"] != True]
failed[validation_cols]
```

Add norm plot:

```python
ax = df.plot.bar(
    x="run",
    y=["embedding_norm_mean", "embedding_norm_min", "embedding_norm_max"],
    figsize=(16, 5),
)

ax.set_title("Embedding norm validation")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("L2 norm")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

---

## Task 8: Smoke test

Run:

```bash
uv run python scripts/bench_onnx_cpu_fp32_stage_breakdown.py \
  --model-dir models/bge-m3-fp32 \
  --dataset mixed \
  --batch-size 4 \
  --max-length 128 \
  --target-words 64 \
  --warmup 1 \
  --batches 2 \
  --validate \
  --expected-embedding-dim 1024 \
  --out results/onnx_cpu_fp32_stage_breakdown.jsonl
```

Inspect validation:

```bash
tail -n 1 results/onnx_cpu_fp32_stage_breakdown.jsonl | jq '{
  validation_passed,
  embedding_extraction_method,
  embedding_shape,
  embedding_dim,
  embedding_nan_count,
  embedding_inf_count,
  embedding_norm_mean,
  reference_cosine_similarity_mean,
  retrieval_top10_overlap,
  validation_errors
}'
```

Expected:

```json
{
  "validation_passed": true,
  "embedding_dim": 1024,
  "embedding_nan_count": 0,
  "embedding_inf_count": 0,
  "validation_errors": []
}
```

---

## Definition of Done

Milestone 1.3 is complete when:

```text
- scripts/validation.py exists.
- Benchmark script can inspect ONNX output names and shapes.
- Benchmark script can extract dense embeddings from sentence_embedding output.
- Benchmark script can fallback to mean pooling for token_embeddings/last_hidden_state.
- Extracted embeddings are L2-normalized by default.
- Benchmark result includes embedding shape and dimension.
- Benchmark result includes NaN/Inf counts.
- Benchmark result includes embedding norm statistics.
- Benchmark result includes validation_passed boolean.
- Benchmark result includes validation_errors list.
- Reference validation is implemented.
- Retrieval overlap validation is implemented with a small English/Thai/mixed validation corpus.
- Notebook includes validation summary table.
- Notebook shows failed validation rows.
```

---

## Interpretation Guide

```text
validation_passed=false:
  Do not trust benchmark speed result yet.

embedding_dim != 1024:
  ONNX export or pooling path may be wrong.

embedding_nan_count > 0 or embedding_inf_count > 0:
  Runtime output is invalid.

embedding_norm_mean far from 1.0:
  Normalization may be missing or disabled.

reference_cosine_similarity_mean < threshold:
  Candidate runtime does not match reference closely enough.

retrieval_top10_overlap < threshold:
  Embeddings may change retrieval results too much.
```

---

## Next Milestone

```text
Milestone 1.4:
  Add ONNX CPU INT8 using the same validation framework.

Milestone 1.5:
  Add ONNX CUDA FP32/FP16 using the same validation framework.

Milestone 1.6:
  Add OpenVINO CPU FP32/INT8 using the same validation framework.
```
