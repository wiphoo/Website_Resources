import numpy as np


VALIDATION_TEXTS = [
    # English
    "http_request_duration_seconds_bucket api latency histogram",
    "container_cpu_usage_seconds_total cpu usage metric",
    "kube_pod_status_phase pod status metric",
    "prometheus_tsdb_head_series time series count",
    "alert high error rate api service",
    "dashboard latency and availability overview",
    "kubernetes pod cpu usage seconds total metric",
    "prometheus alertmanager notification queue full",
    "grpc request duration milliseconds bucket histogram",
    "nginx ingress controller ssl certificate expiry",
    "etcd database compaction and defragmentation operations",
    "kubernetes hpa scaling events and replica count",
    "container memory working set bytes exhausted",
    "istio Envoy proxy access log format and fields",
    "PromQL instant vector range vector selector syntax",
    # Thai
    "สวัสดีครับ ระบบแจ้งเตือนมี latency สูงผิดปกติ",
    "เมตริก http_request_duration_seconds_bucket ใช้วัดเวลา request ของ API",
    "แดชบอร์ดแสดงอัตรา error ของบริการ production",
    "ระบบ Prometheus เก็บข้อมูลจาก pod และ namespace เพื่อตรวจสอบ service",
    "คู่มือ runbook อธิบายขั้นตอนการแก้ปัญหาเมื่อ API ช้า",
    # Mixed
    "prometheus metric latency ของ api service ตรวจสอบ dashboard alert",
]

INT8_PER_LENGTH_THRESHOLDS = {
    32: {
        "cosine_similarity_mean_min": 0.990,
        "cosine_similarity_min_min": 0.985,
        "top10_overlap_min": 0.95,
    },
    128: {
        "cosine_similarity_mean_min": 0.993,
        "cosine_similarity_min_min": 0.990,
        "top10_overlap_min": 0.97,
    },
    512: {
        "cosine_similarity_mean_min": 0.995,
        "cosine_similarity_min_min": 0.992,
        "top10_overlap_min": 0.98,
    },
}


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = round((len(values) - 1) * p)
    return values[idx]


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings.astype(np.float32) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def cls_pooling(token_embeddings: np.ndarray) -> np.ndarray:
    return token_embeddings[:, 0, :].astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def extract_embeddings(
    outputs: list,
    output_names: list[str],
    attention_mask: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, str]:
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
            embeddings = cls_pooling(token_embeddings)
            method = f"cls_pooling:{name}"
            if normalize:
                embeddings = l2_normalize(embeddings)
            return embeddings, method

    first = outputs[0].astype(np.float32)

    if first.ndim == 3:
        embeddings = cls_pooling(first)
        method = "cls_pooling:first_output"
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

    if embeddings.ndim == 2:
        norms = np.linalg.norm(embeddings, axis=1)
        norm_mean = float(norms.mean())
        norm_min = float(norms.min())
        norm_max = float(norms.max())
        norm_std = float(norms.std())

        if require_normalized and not (0.95 <= norm_mean <= 1.05):
            errors.append(f"Embedding norm mean out of expected range: {norm_mean}")
    else:
        norm_mean = norm_min = norm_max = norm_std = None

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


def rowwise_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return (a * b).sum(axis=1)


def validate_against_reference(
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    max_length: int | None = None,
) -> dict:
    sims = rowwise_cosine_similarity(candidate_embeddings, reference_embeddings)

    errors = []

    sim_mean = float(sims.mean())
    sim_min = float(sims.min())
    sim_p01 = float(percentile(list(sims), 0.01))

    threshold_key = None
    if max_length is not None:
        for length_key in sorted(INT8_PER_LENGTH_THRESHOLDS.keys()):
            if max_length <= length_key:
                threshold_key = length_key
                break
        if threshold_key is None:
            threshold_key = max(sorted(INT8_PER_LENGTH_THRESHOLDS.keys()))

    if threshold_key is not None:
        thresholds = INT8_PER_LENGTH_THRESHOLDS[threshold_key]
        cosine_sim_mean_min = thresholds["cosine_similarity_mean_min"]
        ref_sim_min_threshold = thresholds["cosine_similarity_min_min"]
    else:
        cosine_sim_mean_min = 0.999
        ref_sim_min_threshold = 0.995

    if sim_mean < cosine_sim_mean_min:
        errors.append(f"reference_cosine_similarity_mean too low: {sim_mean} < {cosine_sim_mean_min}")

    if sim_min < ref_sim_min_threshold:
        errors.append(f"reference_cosine_similarity_min too low: {sim_min} < {ref_sim_min_threshold}")

    return {
        "reference_validation_enabled": True,
        "reference_cosine_similarity_mean": sim_mean,
        "reference_cosine_similarity_min": sim_min,
        "reference_cosine_similarity_p01": sim_p01,
        "reference_validation_passed": len(errors) == 0,
        "reference_validation_errors": errors,
        "reference_cosine_similarity_threshold": cosine_sim_mean_min,
    }


def topk_indices(similarity_matrix: np.ndarray, k: int) -> list[set[int]]:
    topk = np.argsort(-similarity_matrix, axis=1)[:, :k]
    return [set(row.tolist()) for row in topk]


def validate_retrieval_overlap(
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    max_length: int | None = None,
    top_k_values: list[int] | None = None,
) -> dict:
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    candidate = l2_normalize(candidate_embeddings.astype(np.float32))
    reference = l2_normalize(reference_embeddings.astype(np.float32))

    candidate_sim = candidate @ candidate.T
    reference_sim = reference @ reference.T

    result = {
        "retrieval_validation_enabled": True,
    }

    errors = []

    threshold_key = None
    if max_length is not None:
        for length_key in sorted(INT8_PER_LENGTH_THRESHOLDS.keys()):
            if max_length <= length_key:
                threshold_key = length_key
                break
        if threshold_key is None:
            threshold_key = max(sorted(INT8_PER_LENGTH_THRESHOLDS.keys()))

    for k in top_k_values:
        candidate_topk = topk_indices(candidate_sim, k)
        reference_topk = topk_indices(reference_sim, k)

        overlaps = []
        for c, r in zip(candidate_topk, reference_topk):
            overlaps.append(len(c.intersection(r)) / k)

        value = float(np.mean(overlaps))
        result[f"retrieval_top{k}_overlap"] = value

        if threshold_key is not None:
            top10_threshold = INT8_PER_LENGTH_THRESHOLDS[threshold_key]["top10_overlap_min"]
            if k == 10 and value < top10_threshold:
                errors.append(f"retrieval_top10_overlap too low: {value} < {top10_threshold}")

    if threshold_key is not None:
        result["retrieval_top10_overlap_threshold"] = INT8_PER_LENGTH_THRESHOLDS[threshold_key]["top10_overlap_min"]

    result["retrieval_validation_passed"] = len(errors) == 0
    result["retrieval_validation_errors"] = errors

    return result

