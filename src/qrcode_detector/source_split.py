from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}


@dataclass(slots=True, frozen=True)
class SourceSplitConfig:
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    synthesis_source_ratio: float = 0.5
    seed: int = 42


def split_training_sources(
    origins_dir: str | Path,
    reals_dir: str | Path,
    output_dir: str | Path = "auto_generated_test_images/splits",
    config: SourceSplitConfig | None = None,
) -> dict[str, object]:
    current_config = config or SourceSplitConfig()
    _validate_config(current_config)

    origins_root = Path(origins_dir)
    reals_root = Path(reals_dir)
    output_root = Path(output_dir)

    origin_images = _list_images(origins_root)
    real_images = _list_images(reals_root)

    if not origin_images:
        raise ValueError(f"No images found under origins directory: {origins_root}")

    rng = random.Random(current_config.seed)
    rng.shuffle(origin_images)
    rng.shuffle(real_images)

    origin_splits = _split_items(origin_images, current_config.train_ratio, current_config.validation_ratio)
    real_splits = _split_items(real_images, current_config.train_ratio, current_config.validation_ratio)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "output_dir": str(output_root.resolve()),
        "config": {
            "train_ratio": current_config.train_ratio,
            "validation_ratio": current_config.validation_ratio,
            "synthesis_source_ratio": current_config.synthesis_source_ratio,
            "seed": current_config.seed,
        },
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        negative_images, synthesis_images = _split_negative_and_synthesis(
            origin_splits[split_name],
            current_config.synthesis_source_ratio,
        )
        real_positive_images = real_splits[split_name]

        split_root = output_root / split_name
        negative_dir = split_root / "negative_origins"
        synthesis_dir = split_root / "synthetic_sources"
        real_dir = split_root / "real_positives"

        _copy_images(negative_images, negative_dir)
        _copy_images(synthesis_images, synthesis_dir)
        _copy_images(real_positive_images, real_dir)

        summary["splits"][split_name] = {
            "negative_origins": len(negative_images),
            "synthetic_sources": len(synthesis_images),
            "real_positives": len(real_positive_images),
        }

    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def _validate_config(config: SourceSplitConfig) -> None:
    if config.train_ratio <= 0 or config.validation_ratio <= 0 or config.train_ratio + config.validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and leave room for test")
    if not 0 < config.synthesis_source_ratio < 1:
        raise ValueError("synthesis_source_ratio must be between 0 and 1")


def _list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    images = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return images


def _split_items(items: list[Path], train_ratio: float, validation_ratio: float) -> dict[str, list[Path]]:
    total_count = len(items)
    if total_count == 0:
        return {"train": [], "val": [], "test": []}

    train_count = int(total_count * train_ratio)
    validation_count = int(total_count * validation_ratio)
    test_count = total_count - train_count - validation_count

    if total_count >= 3:
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)
        test_count = total_count - train_count - validation_count
        if test_count <= 0:
            test_count = 1
            if train_count >= validation_count and train_count > 1:
                train_count -= 1
            elif validation_count > 1:
                validation_count -= 1

    return {
        "train": items[:train_count],
        "val": items[train_count:train_count + validation_count],
        "test": items[train_count + validation_count:],
    }


def _split_negative_and_synthesis(items: list[Path], synthesis_source_ratio: float) -> tuple[list[Path], list[Path]]:
    if not items:
        return [], []

    synthesis_count = int(round(len(items) * synthesis_source_ratio))
    if len(items) >= 2:
        synthesis_count = min(max(1, synthesis_count), len(items) - 1)
    else:
        synthesis_count = 0

    synthesis_images = items[:synthesis_count]
    negative_images = items[synthesis_count:]
    return negative_images, synthesis_images


def _copy_images(images: list[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        shutil.copy2(image_path, destination_dir / image_path.name)
