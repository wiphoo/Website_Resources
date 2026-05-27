import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer


SUPPORTED_OPTIMIZE = {"O0", "O1", "O2", "O3"}
VALID_DTYPE = {"float32", "float16", "int8"}


def _validate_model_name(name: str) -> str:
    if name.startswith("-") or "\x00" in name:
        raise ValueError(f"Invalid model name: {name!r}")
    return name


def _validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if "\x00" in str(path):
        raise ValueError("output_dir contains null byte")
    return resolved


def _model_name_type(value: str) -> str:
    if value.startswith("-") or "\x00" in value:
        raise argparse.ArgumentTypeError(f"invalid model name: {value!r}")
    return value


def export_model(
    model_name: str,
    output_dir: Path,
    optimize: str = "O3",
    dtype: str = "float32",
) -> None:
    _validate_model_name(model_name)
    output_dir = _validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if optimize.upper() not in SUPPORTED_OPTIMIZE:
        raise ValueError(f"optimize must be one of {SUPPORTED_OPTIMIZE}, got {optimize}")
    if dtype not in VALID_DTYPE:
        raise ValueError(f"dtype must be one of {VALID_DTYPE}, got {dtype}")

    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
    tokenizer.save_pretrained(output_dir)

    exe = shutil.which("optimum-cli")
    cmd = (
        [exe, "export", "onnx", "--model", model_name, "--task", "feature-extraction", "--dtype", dtype, "--optimize", optimize.lower(), str(output_dir)]
        if exe
        else [sys.executable, "-m", "optimum", "export", "onnx", "--model", model_name, "--task", "feature-extraction", "--dtype", dtype, "--optimize", optimize.lower(), str(output_dir)]
    )
    print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(output_dir.parent))
    if result.returncode != 0:
        raise RuntimeError(
            f"optimum export failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

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
        type=_model_name_type,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output",
        default="models/bge-m3-fp32",
        type=Path,
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
    except (OSError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, SystemExit):
        raise


if __name__ == "__main__":
    sys.exit(main())
