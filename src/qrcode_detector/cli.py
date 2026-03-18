from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .dataset import export_yolo_dataset, load_manifest, split_records
from .detector import QRCodeDetector
from .synthetic import compose_synthetic_qrcode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QR code detection utility commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run QR code detection on one image.")
    detect_parser.add_argument("--model", required=True, help="Path to a YOLO model weights file.")
    detect_parser.add_argument("--image", required=True, help="Path to the input image.")

    synthesize_parser = subparsers.add_parser("synthesize", help="Paste one QR code onto a background image.")
    synthesize_parser.add_argument("--background", required=True, help="Path to a poster image.")
    synthesize_parser.add_argument("--qrcode", required=True, help="Path to a QR code image.")
    synthesize_parser.add_argument("--output", required=True, help="Path to the output synthetic image.")
    synthesize_parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")

    export_parser = subparsers.add_parser("export-dataset", help="Split a manifest and export YOLO labels.")
    export_parser.add_argument("--manifest", required=True, help="Path to the JSONL manifest.")
    export_parser.add_argument("--output-dir", required=True, help="Directory for the generated dataset.")
    export_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    export_parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    export_parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "detect":
        detector = QRCodeDetector(model_path=arguments.model)
        result = detector.detect(arguments.image)
        print(json.dumps({
            "has_qrcode": result.has_qrcode,
            "score": result.score,
            "boxes": [asdict(box) for box in result.boxes],
        }, ensure_ascii=True, indent=2))
        return

    if arguments.command == "synthesize":
        placement = compose_synthetic_qrcode(
            background_path=arguments.background,
            qrcode_path=arguments.qrcode,
            output_path=arguments.output,
            seed=arguments.seed,
        )
        print(json.dumps(asdict(placement), ensure_ascii=True, indent=2))
        return

    if arguments.command == "export-dataset":
        records = load_manifest(arguments.manifest)
        records_by_split = split_records(
            records=records,
            train_ratio=arguments.train_ratio,
            validation_ratio=arguments.validation_ratio,
            seed=arguments.seed,
        )
        export_yolo_dataset(records_by_split=records_by_split, output_dir=arguments.output_dir)
        counts = {split_name: len(split_records_list) for split_name, split_records_list in records_by_split.items()}
        print(json.dumps(counts, ensure_ascii=True, indent=2))
        return

    raise RuntimeError(f"Unsupported command: {arguments.command}")


if __name__ == "__main__":
    main()
