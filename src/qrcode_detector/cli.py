from __future__ import annotations

import time
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qrcode_detector.dataset_build import build_dataset_from_splits
    from qrcode_detector.download import download_images_from_csv
    from qrcode_detector.dataset import export_yolo_dataset, load_manifest, split_records
    from qrcode_detector.detector import QRCodeDetector
    from qrcode_detector.source_split import SourceSplitConfig, split_training_sources
    from qrcode_detector.synthetic import SyntheticQRCodeConfig, compose_synthetic_qrcode, synthesize_directory
else:
    from .dataset_build import build_dataset_from_splits
    from .download import download_images_from_csv
    from .dataset import export_yolo_dataset, load_manifest, split_records
    from .detector import QRCodeDetector
    from .labelme import export_labelme_directory_to_yolo
    from .source_split import SourceSplitConfig, split_training_sources
    from .synthetic import SyntheticQRCodeConfig, compose_synthetic_qrcode, synthesize_directory
if __package__ in {None, ""}:
    from qrcode_detector.labelme import export_labelme_directory_to_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QR code detection utility commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run QR code detection on one image.")
    detect_parser.add_argument("--model", required=True, help="Path to a YOLO model weights file.")
    detect_parser.add_argument("--image", required=True, help="Path to the input image.")
    detect_parser.add_argument(
        "--output",
        help="Save the image with drawn bounding boxes, labels and scores to this path.",
    )
    detect_parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image in the default system image viewer.",
    )

    synthesize_parser = subparsers.add_parser("synthesize", help="Paste one QR code onto a background image.")
    synthesize_parser.add_argument("--background", required=True, help="Path to a poster image.")
    synthesize_parser.add_argument("--qrcode", required=True, help="Path to a QR code image.")
    synthesize_parser.add_argument("--output", required=True, help="Path to the output synthetic image.")
    synthesize_parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")

    synthesize_dir_parser = subparsers.add_parser(
        "synthesize-directory",
        help="Generate one synthetic QR code image for each input image in a directory.",
    )
    synthesize_dir_parser.add_argument("--input-dir", required=True, help="Directory that contains original images.")
    synthesize_dir_parser.add_argument("--output-dir", required=True, help="Directory for generated synthetic images.")
    synthesize_dir_parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    synthesize_dir_parser.add_argument("--recursive", action="store_true", help="Traverse subdirectories recursively.")
    synthesize_dir_parser.add_argument(
        "--variants-per-image",
        type=int,
        default=1,
        help="How many synthetic variants to generate for each source image.",
    )
    synthesize_dir_parser.add_argument(
        "--write-labelme-json",
        action="store_true",
        help="Write one Labelme rectangle JSON file next to each generated image.",
    )

    export_parser = subparsers.add_parser("export-dataset", help="Split a manifest and export YOLO labels.")
    export_parser.add_argument("--manifest", required=True, help="Path to the JSONL manifest.")
    export_parser.add_argument("--output-dir", required=True, help="Directory for the generated dataset.")
    export_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    export_parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    export_parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")

    download_parser = subparsers.add_parser(
        "download-images",
        help="Download image URLs from a CSV file into the origins directory.",
    )
    download_parser.add_argument("--csv", required=True, help="CSV file where each row contains an image URL.")
    download_parser.add_argument(
        "--output-dir",
        default="auto_generated_test_images/origins",
        help="Directory for downloaded original images.",
    )
    download_parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")

    labelme_parser = subparsers.add_parser(
        "export-labelme-dataset",
        help="Convert a Labelme directory into a YOLO dataset with train/val/test splits.",
    )
    labelme_parser.add_argument("--input-dir", required=True, help="Directory that contains Labelme JSON and images.")
    labelme_parser.add_argument("--output-dir", required=True, help="Directory for the generated YOLO dataset.")
    labelme_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    labelme_parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    labelme_parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")

    split_parser = subparsers.add_parser(
        "split-training-sources",
        help="Split origin negatives and real positives into train/val/test source directories.",
    )
    split_parser.add_argument(
        "--origins-dir",
        default="auto_generated_test_images/origins",
        help="Directory of real negative source images.",
    )
    split_parser.add_argument(
        "--reals-dir",
        default="auto_generated_test_images/reals",
        help="Directory of real positive source images.",
    )
    split_parser.add_argument(
        "--output-dir",
        default="auto_generated_test_images/splits",
        help="Directory for split source datasets.",
    )
    split_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    split_parser.add_argument("--validation-ratio", type=float, default=0.1, help="Validation split ratio.")
    split_parser.add_argument(
        "--synthesis-source-ratio",
        type=float,
        default=0.5,
        help="Ratio of origin images reserved as synthetic positive sources inside each split.",
    )
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")

    build_parser = subparsers.add_parser(
        "build-dataset-from-splits",
        help="Build the final YOLO dataset from processed synthetic data and split source directories.",
    )
    build_parser.add_argument(
        "--splits-dir",
        default="auto_generated_test_images/splits",
        help="Directory that contains train/val/test split source folders.",
    )
    build_parser.add_argument(
        "--processed-dir",
        default="auto_generated_test_images/processed",
        help="Directory that contains processed synthetic positives grouped by split.",
    )
    build_parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Directory for the generated YOLO dataset.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "detect":
        detector = QRCodeDetector(model_path=arguments.model)
        draw_on_image = arguments.output is not None or arguments.show
        out = detector.detect(
            arguments.image,
            draw_on_image=draw_on_image,
            output_path=arguments.output,
            show=arguments.show,
        )
        result = out[0] if isinstance(out, tuple) else out
        print(json.dumps({
            "has_qrcode": result.has_qrcode,
            "score": result.score,
            "elapsed_ms": result.elapsed_ms,
            "read_elapsed_ms": result.read_elapsed_ms,
            "predict_elapsed_ms": result.predict_elapsed_ms,
            "postprocess_elapsed_ms": result.postprocess_elapsed_ms,
            "boxes": [asdict(box) for box in result.boxes],
        }, ensure_ascii=True, indent=2))
        if isinstance(out, tuple) and result.boxes:
            print(f"Visualization saved to: {arguments.output}", file=sys.stderr)
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

    if arguments.command == "synthesize-directory":
        total_start_time = time.perf_counter()
        results = synthesize_directory(
            input_dir=arguments.input_dir,
            output_dir=arguments.output_dir,
            seed=arguments.seed,
            recursive=arguments.recursive,
            write_labelme_json=arguments.write_labelme_json,
            config=SyntheticQRCodeConfig(variants_per_image=arguments.variants_per_image),
        )
        elapsed_ms = (time.perf_counter() - total_start_time) * 1000.0
        print(f"Synthesize directory elapsed time: {elapsed_ms:.2f}ms")
        # print(json.dumps({
        #     "count": len(results),
        #     "items": results,
        # }, ensure_ascii=True, indent=2))
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

    if arguments.command == "download-images":
        summary = download_images_from_csv(
            csv_path=arguments.csv,
            output_dir=arguments.output_dir,
            timeout_seconds=arguments.timeout,
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return

    if arguments.command == "export-labelme-dataset":
        counts = export_labelme_directory_to_yolo(
            input_dir=arguments.input_dir,
            output_dir=arguments.output_dir,
            train_ratio=arguments.train_ratio,
            validation_ratio=arguments.validation_ratio,
            seed=arguments.seed,
        )
        print(json.dumps(counts, ensure_ascii=True, indent=2))
        return

    if arguments.command == "split-training-sources":
        summary = split_training_sources(
            origins_dir=arguments.origins_dir,
            reals_dir=arguments.reals_dir,
            output_dir=arguments.output_dir,
            config=SourceSplitConfig(
                train_ratio=arguments.train_ratio,
                validation_ratio=arguments.validation_ratio,
                synthesis_source_ratio=arguments.synthesis_source_ratio,
                seed=arguments.seed,
            ),
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return

    if arguments.command == "build-dataset-from-splits":
        total_start_time = time.perf_counter()
        summary = build_dataset_from_splits(
            splits_dir=arguments.splits_dir,
            processed_dir=arguments.processed_dir,
            output_dir=arguments.output_dir,
        )
        # print(json.dumps(summary, ensure_ascii=True, indent=2))
        elapsed_ms = (time.perf_counter() - total_start_time) * 1000.0
        print(f"Build dataset from splits elapsed time: {elapsed_ms:.2f}ms")
        return

    raise RuntimeError(f"Unsupported command: {arguments.command}")


if __name__ == "__main__":
    main()
