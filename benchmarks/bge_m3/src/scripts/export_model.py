import argparse
import sys
from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


SUPPORTED_OPTIMIZE = {"O0", "O1", "O2", "O3"}


def export_model(
    model_name: str,
    output_dir: Path,
    optimize: str = "O3",
    dtype: str = "float32",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{model_name}'...")
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
        provider="CPUExecutionProvider",
    )

    print(f"Exporting ONNX model to '{output_dir}' with optimization {optimize}...")
    model.save_pretrained(output_dir, optimize=optimize, dtype=dtype)

    print(f"Exporting tokenizer to '{output_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print("Done.")
    print(f"Files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name} ({size_kb} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BGE-M3 to ONNX FP32")
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output",
        default="models/bge-m3-fp32",
        help="Output directory for ONNX model and tokenizer",
    )
    parser.add_argument(
        "--optimize",
        default="O3",
        choices=SUPPORTED_OPTIMIZE,
        help="ONNX optimization level (O0-O3)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Data type: float32, float16, int8",
    )
    args = parser.parse_args()

    try:
        export_model(args.model, args.output, args.optimize, args.dtype)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
