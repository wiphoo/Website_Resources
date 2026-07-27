from pathlib import Path
import argparse
import shutil

from onnxruntime.quantization import QuantType, quantize_dynamic


def copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy all tokenizer files from ``src_dir`` into ``dst_dir``.

    Matches: ``*.json``, ``tokenizer*``, ``vocab*``, ``sentencepiece*``, ``*.txt``.
    Skips directories.  Does not overwrite existing files.

    :param src_dir: Source model directory containing tokenizer files.
    :param dst_dir: Destination directory for copied tokenizer files.
    """
    patterns = [
        "*.json",
        "tokenizer*",
        "vocab*",
        "sentencepiece*",
        "*.txt",
    ]

    for pattern in patterns:
        for src in src_dir.glob(pattern):
            if src.is_file():
                shutil.copy2(src, dst_dir / src.name)


def main() -> None:
    """
    CLI entry point: quantize a BGE-M3 FP32 ONNX model to INT8.

    Uses :func:`onnxruntime.quantization.quantize_dynamic` with configurable
    weight type (qint8 / quint8).  Copies tokenizer assets to the output
    directory so the quantized model is self-contained.

    :raises FileNotFoundError: If ``src_model_dir / "model.onnx"`` does not exist.
    :raises ValueError: If ``--weight-type`` is not ``qint8`` or ``quint8``.

    CLI flags
    ---------
    --src-model-dir   : Source FP32 model directory (must contain model.onnx).
    --dst-model-dir   : Destination INT8 model output directory.
    --weight-type     : ``qint8`` (default) or ``quint8``.
    --per-channel    : Quantize weights per output channel (usually recovers
                       accuracy on transformer linear layers).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-model-dir", default="models/bge-m3-fp32")
    parser.add_argument("--dst-model-dir", default="models/bge-m3-int8-dynamic")
    parser.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Quantize weights per output channel instead of per tensor "
        "(usually recovers accuracy on transformer linear layers).",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_model_dir)
    dst_dir = Path(args.dst_model_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_model = src_dir / "model.onnx"
    dst_model = dst_dir / "model.onnx"

    if not src_model.exists():
        raise FileNotFoundError(f"Missing source ONNX model: {src_model}")

    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    quantize_dynamic(
        model_input=str(src_model),
        model_output=str(dst_model),
        weight_type=weight_type,
        per_channel=args.per_channel,
    )

    copy_tokenizer_files(src_dir, dst_dir)

    print(f"Saved INT8 model to: {dst_model}")


if __name__ == "__main__":
    main()
