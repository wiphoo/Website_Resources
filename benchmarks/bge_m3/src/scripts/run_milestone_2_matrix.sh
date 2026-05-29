#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

rm -f "$PROJECT_DIR/results/onnx_cpu_fp32_stage_breakdown.jsonl"
rm -f "$PROJECT_DIR/results/onnx_cpu_int8_stage_breakdown.jsonl"

for dataset in en th mixed; do
  for bs in 1 8 16 32; do
    for len in 32 128 512; do
      uv run python scripts/bench_onnx_cpu_stage_breakdown.py \
        --model-dir models/bge-m3-fp32 \
        --precision fp32 \
        --model-variant onnx-cpu-fp32 \
        --dataset "$dataset" \
        --batch-size "$bs" \
        --max-length "$len" \
        --target-words "$len" \
        --warmup 5 \
        --batches 20 \
        --validate \
        --out ../results/onnx_cpu_fp32_stage_breakdown.jsonl

      uv run python scripts/bench_onnx_cpu_stage_breakdown.py \
        --model-dir models/bge-m3-int8-dynamic \
        --precision int8 \
        --model-variant onnx-cpu-int8-dynamic \
        --quantization-method dynamic \
        --quantization-weight-type qint8 \
        --reference-model-dir models/bge-m3-fp32 \
        --dataset "$dataset" \
        --batch-size "$bs" \
        --max-length "$len" \
        --target-words "$len" \
        --warmup 5 \
        --batches 20 \
        --validate \
        --out ../results/onnx_cpu_int8_stage_breakdown.jsonl
    done
  done
done
