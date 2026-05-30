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
from validation import (
    extract_embeddings,
    validate_embeddings,
    validate_against_reference,
    validate_retrieval_overlap,
    VALIDATION_TEXTS,
)

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


def build_quantization_metadata(args: argparse.Namespace) -> dict:
    """
    Build the ``quantization`` metadata dict for the result JSON from CLI arguments.

    When ``args.precision == "int8"``, returns a dict with full quantization
    metadata (method, weight_type, activation_type, calibration, source variant).
    When ``args.precision == "fp32"``, returns a dict with all quantization
    fields set to ``None`` (quantization not applicable to FP32 runs).

    :param args: Parsed CLI arguments from :func:`argparse.ArgumentParser.parse_args`.
    :returns: Dict with keys:
        - enabled (bool)
        - method (str | None)
        - weight_type (str | None)
        - activation_type (str | None)
        - calibration_enabled (bool)
        - calibration_dataset (str | None)
        - source_model_variant (str | None)
    """
    if args.precision == "int8":
        return {
            "enabled": True,
            "method": args.quantization_method or "dynamic",
            "weight_type": args.quantization_weight_type or "qint8",
            "activation_type": None,
            "calibration_enabled": False,
            "calibration_dataset": None,
            "source_model_variant": "onnx-cpu-fp32",
        }
    else:
        return {
            "enabled": False,
            "method": None,
            "weight_type": None,
            "activation_type": None,
            "calibration_enabled": False,
            "calibration_dataset": None,
            "source_model_variant": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/bge-m3-fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--target-words", type=int, default=256)
    parser.add_argument("--dataset", choices=["en", "th", "mixed"], default="mixed")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--out", default="../results/onnx_cpu_stage_breakdown.jsonl")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--expected-embedding-dim", type=int, default=1024)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", default=True)
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--model-variant", default="onnx-cpu-fp32")
    parser.add_argument("--quantization-method", default=None)
    parser.add_argument("--quantization-weight-type", default=None)
    parser.add_argument("--reference-model-dir", default=None)
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

    if args.precision == "int8" and (
        not args.reference_model_dir
        or Path(args.reference_model_dir) == Path(args.model_dir)
    ):
        parser.error(
            "--reference-model-dir (FP32) is required and must differ from "
            "--model-dir when --precision int8; otherwise INT8 validates against itself"
        )

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

    validation_result = {}
    reference_embeddings = None
    if args.validate:
        validation_texts = VALIDATION_TEXTS
        encoded_val = tokenizer(
            validation_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        inputs_val = make_inputs(session, encoded_val)

        outputs_candidate = session.run(None, inputs_val)
        output_names = [o.name for o in session.get_outputs()]
        output_shapes = [list(o.shape) for o in outputs_candidate]
        attention_mask_val = encoded_val["attention_mask"]

        cand_embeddings, extraction_method_cand = extract_embeddings(
            outputs=outputs_candidate,
            output_names=output_names,
            attention_mask=attention_mask_val,
            normalize=args.normalize,
        )

        emb_validation = validate_embeddings(
            embeddings=cand_embeddings,
            expected_batch_size=len(validation_texts),
            expected_embedding_dim=args.expected_embedding_dim,
            require_normalized=args.normalize,
        )

        ref_model_dir = Path(args.reference_model_dir) if args.reference_model_dir else None
        if ref_model_dir and ref_model_dir != model_dir:
            ref_tokenizer = AutoTokenizer.from_pretrained(str(ref_model_dir))
            ref_onnx_path = ref_model_dir / "model.onnx"
            ref_session_options = ort.SessionOptions()
            ref_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ref_session = ort.InferenceSession(
                str(ref_onnx_path),
                sess_options=ref_session_options,
                providers=["CPUExecutionProvider"],
            )
            encoded_val_ref = ref_tokenizer(
                validation_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="np",
            )
            inputs_val_ref = make_inputs(ref_session, encoded_val_ref)
            outputs_ref = ref_session.run(None, inputs_val_ref)
            ref_output_names = [o.name for o in ref_session.get_outputs()]
            reference_embeddings, _ = extract_embeddings(
                outputs=outputs_ref,
                output_names=ref_output_names,
                attention_mask=encoded_val_ref["attention_mask"],
                normalize=args.normalize,
            )
        else:
            outputs_ref = session.run(None, inputs_val)
            reference_embeddings, _ = extract_embeddings(
                outputs=outputs_ref,
                output_names=output_names,
                attention_mask=attention_mask_val,
                normalize=args.normalize,
            )

        ref_validation = validate_against_reference(
            cand_embeddings,
            reference_embeddings,
            max_length=args.max_length,
            is_int8=(args.precision == "int8"),
        )

        retrieval_validation = validate_retrieval_overlap(
            cand_embeddings,
            reference_embeddings,
            max_length=args.max_length,
            is_int8=(args.precision == "int8"),
        )

        validation_result = {
            "validation_enabled": True,
            "onnx_output_names": output_names,
            "onnx_output_shapes": output_shapes,
            "embedding_extraction_method": extraction_method_cand,
        }
        validation_result.update(emb_validation)
        validation_result.update(ref_validation)
        validation_result.update(retrieval_validation)

        combined_errors = (
            emb_validation.get("validation_errors", [])
            + ref_validation.get("reference_validation_errors", [])
            + retrieval_validation.get("retrieval_validation_errors", [])
        )
        validation_result["validation_passed"] = (
            emb_validation.get("validation_passed", False)
            and ref_validation.get("reference_validation_passed", False)
            and retrieval_validation.get("retrieval_validation_passed", False)
        )
        validation_result["validation_errors"] = combined_errors

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
        "schema_version": "2.0",
        "run_id": f"{args.model_variant}-{args.dataset}-bs{args.batch_size}-l{args.max_length}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "BAAI/bge-m3",
        "model_variant": args.model_variant,
        "runtime": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "precision": args.precision,
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

    quantization_metadata = build_quantization_metadata(args)
    result["quantization"] = quantization_metadata

    if args.precision == "int8":
        result["reference_model_variant"] = "onnx-cpu-fp32"
        result["candidate_model_variant"] = "onnx-cpu-int8-dynamic"
        result["reference_precision"] = "fp32"
        result["candidate_precision"] = "int8"
    else:
        result["reference_model_variant"] = None
        result["candidate_model_variant"] = None
        result["reference_precision"] = None
        result["candidate_precision"] = None

    if validation_result:
        result.update(validation_result)

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
