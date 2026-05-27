import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer


SUPPORTED_OPTIMIZE = {"O0", "O1", "O2", "O3"}


def export_model(
    model_name: str,
    output_dir: Path,
    optimize: str = "O3",
    dtype: str = "float32",
) -> None:
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    cmd = [
        sys.executable, "-m", "optimum-cli", "export", "onnx",
        "--model", model_name,
        "--task", "feature-extraction",
        "--dtype", dtype,
        "--optimize", optimize.lower(),
        output_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"optimum-cli export failed with code {result.returncode}")

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
