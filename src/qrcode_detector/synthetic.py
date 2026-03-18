from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass(slots=True, frozen=True)
class SyntheticPlacement:
    x1: int
    y1: int
    x2: int
    y2: int


def _require_pillow() -> None:
    if Image is None:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required for synthetic generation. Install with: pip install -e '.[synthetic]'"
        )


def compose_synthetic_qrcode(
    background_path: str | Path,
    qrcode_path: str | Path,
    output_path: str | Path,
    seed: int | None = None,
    corner_ratio: float = 0.35,
    scale_range: tuple[float, float] = (0.12, 0.22),
) -> SyntheticPlacement:
    _require_pillow()
    rng = random.Random(seed)

    with Image.open(background_path).convert("RGBA") as background:
        with Image.open(qrcode_path).convert("RGBA") as qrcode:
            background_width, background_height = background.size
            min_scale, max_scale = scale_range
            target_scale = rng.uniform(min_scale, max_scale)
            target_size = max(32, int(min(background_width, background_height) * target_scale))
            qrcode = qrcode.resize((target_size, target_size))
            qrcode = _apply_random_style(qrcode, rng)

            left_area_end = int(background_width * corner_ratio)
            right_area_start = int(background_width * (1.0 - corner_ratio))
            bottom_area_start = int(background_height * (1.0 - corner_ratio))

            place_on_left = rng.random() < 0.5
            x_start = 0 if place_on_left else right_area_start
            x_end = max(x_start, left_area_end if place_on_left else background_width)
            y_start = bottom_area_start
            y_end = background_height

            x1 = rng.randint(x_start, max(x_start, x_end - target_size))
            y1 = rng.randint(y_start, max(y_start, y_end - target_size))
            x2 = x1 + target_size
            y2 = y1 + target_size

            canvas = background.copy()
            canvas.alpha_composite(qrcode, (x1, y1))
            output = canvas.convert("RGB")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            output.save(output_path, quality=rng.randint(88, 96))
            return SyntheticPlacement(x1=x1, y1=y1, x2=x2, y2=y2)


def _apply_random_style(image: "Image.Image", rng: random.Random) -> "Image.Image":
    angle = rng.uniform(-12.0, 12.0)
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    alpha = rotated.getchannel("A")
    alpha = alpha.point(lambda pixel: int(pixel * rng.uniform(0.88, 1.0)))
    rotated.putalpha(alpha)
    return rotated
