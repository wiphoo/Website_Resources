import warnings
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from machine_metadata import collect_machine_metadata, flatten_machine_metadata

warnings.filterwarnings("ignore", message=".*regex pattern.*")


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

    for name, val in [
        ("batch_size", args.batch_size),
        ("max_length", args.max_length),
        ("target_words", args.target_words),
        ("batches", args.batches),
    ]:
        if val <= 0:
            parser.error(f"--{name} must be > 0 (got {val})")
    if args.warmup < 0:
        parser.error(f"--warmup must be >= 0 (got {args.warmup})")

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

    try:
        machine_metadata = collect_machine_metadata(session=session)
    except Exception as e:
        machine_metadata = {"error": str(e)}

    texts = make_texts(args.batch_size, args.target_words, args.dataset)

    encoded_warmup = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="np",
    )
    inputs_warmup = make_inputs(session, encoded_warmup)

    for _ in range(args.warmup):
        tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        session.run(None, inputs_warmup)

    # Phase 1: Tokenization-only
    tokenize_latencies = []
    for _ in range(args.batches):
        start = time.perf_counter()
        tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        elapsed = time.perf_counter() - start
        tokenize_latencies.append(elapsed)

    # Phase 2: Embedding-only (tokenize once, time embedding only)
    encoded_once = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="np",
    )
    real_tokens_once = int(encoded_once["attention_mask"].sum())
    inputs_once = make_inputs(session, encoded_once)

    embedding_latencies = []
    for _ in range(args.batches):
        start = time.perf_counter()
        session.run(None, inputs_once)
        elapsed = time.perf_counter() - start
        embedding_latencies.append(elapsed)

    # Phase 3: End-to-end
    e2e_latencies = []
    total_items = 0
    total_tokens = 0

    for _ in range(args.batches):
        start = time.perf_counter()

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

        elapsed = time.perf_counter() - start
        e2e_latencies.append(elapsed)
        total_items += args.batch_size
        total_tokens += real_tokens

    tokenize_time = sum(tokenize_latencies)
    embedding_time = sum(embedding_latencies)
    e2e_time = sum(e2e_latencies)

    total_items_phase2 = args.batches * args.batch_size
    total_tokens_phase2 = real_tokens_once * args.batches

    result = {
        "model": "BAAI/bge-m3",
        "runtime": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "precision": "fp32",
        "dataset": args.dataset,
        "language": args.dataset,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "target_words": args.target_words,
        "warmup": args.warmup,
        "batches": args.batches,
        "total_items": total_items,
        "total_tokens": total_tokens,
        "avg_tokens_per_item": total_tokens / total_items,
        "tokenize_tokens_per_sec": total_tokens_phase2 / tokenize_time,
        "embedding_tokens_per_sec": total_tokens_phase2 / embedding_time,
        "end_to_end_tokens_per_sec": total_tokens / e2e_time,
        "tokenize_items_per_sec": total_items_phase2 / tokenize_time,
        "embedding_items_per_sec": total_items_phase2 / embedding_time,
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
        "machine_metadata": machine_metadata,
    }
    result.update(flatten_machine_metadata(machine_metadata))

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
