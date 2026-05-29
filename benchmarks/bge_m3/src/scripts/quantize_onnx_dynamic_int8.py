from pathlib import Path
import argparse
import shutil

from onnxruntime.quantization import QuantType, quantize_dynamic


def copy_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-model-dir", default="models/bge-m3-fp32")
    parser.add_argument("--dst-model-dir", default="models/bge-m3-int8-dynamic")
    parser.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
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
    )

    copy_tokenizer_files(src_dir, dst_dir)

    print(f"Saved INT8 model to: {dst_model}")


if __name__ == "__main__":
    main()
