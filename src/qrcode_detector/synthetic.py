from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass(slots=True, frozen=True)
class SyntheticPlacement:
    image_width: int
    image_height: int
    x1: int
    y1: int
    x2: int
    y2: int


def _require_pillow() -> None:
    if Image is None:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required for synthetic generation. Install with: pip install -e '.[synthetic]'"
        )


def generate_qrcode_image(output_path: str | Path, seed: int | None = None) -> str:
    try:
        import qrcode
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qrcode is required for QR generation. Install with: pip install -e '.[synthetic]'"
        ) from exc

    rng = random.Random(seed)
    qr_content = _random_qrcode_content(rng)
    fill_color = _random_dark_color(rng)
    back_color = _random_light_color(rng)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=16,
        border=2,
    )
    qr.add_data(qr_content)
    qr.make(fit=True)
    image = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGBA")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_file)
    return qr_content


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
            return SyntheticPlacement(
                image_width=background_width,
                image_height=background_height,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )


def synthesize_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    seed: int | None = None,
    recursive: bool = False,
    write_labelme_json: bool = False,
) -> list[dict[str, str | int]]:
    _require_pillow()
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    rng = random.Random(seed)
    image_paths = list(_iter_image_paths(input_root, recursive))
    if not image_paths:
        raise ValueError(f"No images found under: {input_root}")

    results: list[dict[str, str | int]] = []
    for index, image_path in enumerate(image_paths):
        relative_path = image_path.relative_to(input_root)
        output_image_path = output_root / relative_path.parent / f"{image_path.stem}_synthetic{image_path.suffix}"
        generated_qrcode_path = output_root / "_generated_qrcodes" / relative_path.parent / f"{image_path.stem}_qrcode.png"

        current_seed = rng.randint(0, 10**9)
        qr_content = generate_qrcode_image(generated_qrcode_path, seed=current_seed)
        placement = compose_synthetic_qrcode(
            background_path=image_path,
            qrcode_path=generated_qrcode_path,
            output_path=output_image_path,
            seed=current_seed,
        )

        if write_labelme_json:
            labelme_path = output_image_path.with_suffix(".json")
            write_labelme_rectangle(labelme_path, output_image_path, placement)

        results.append({
            "index": index,
            "input_image": str(image_path),
            "output_image": str(output_image_path),
            "qrcode_image": str(generated_qrcode_path),
            "qrcode_content": qr_content,
            "x1": placement.x1,
            "y1": placement.y1,
            "x2": placement.x2,
            "y2": placement.y2,
        })

    return results


def write_labelme_rectangle(
    output_path: str | Path,
    image_path: str | Path,
    placement: SyntheticPlacement,
) -> None:
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [
            {
                "label": "qrcode",
                "points": [
                    [placement.x1, placement.y1],
                    [placement.x2, placement.y2],
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": Path(image_path).name,
        "imageData": None,
        "imageHeight": placement.image_height,
        "imageWidth": placement.image_width,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _apply_random_style(image: "Image.Image", rng: random.Random) -> "Image.Image":
    alpha = image.getchannel("A")
    alpha = alpha.point(lambda pixel: int(pixel * rng.uniform(0.88, 1.0)))
    image.putalpha(alpha)
    return image


def _iter_image_paths(input_root: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield path


def _random_qrcode_content(rng: random.Random) -> str:
    alphabet = string.ascii_letters + string.digits
    token = "".join(rng.choice(alphabet) for _ in range(24))
    return f"https://example.com/event/{token}"


def _random_dark_color(rng: random.Random) -> tuple[int, int, int]:
    return (
        rng.randint(0, 80),
        rng.randint(0, 80),
        rng.randint(0, 80),
    )


def _random_light_color(rng: random.Random) -> tuple[int, int, int]:
    return (
        rng.randint(220, 255),
        rng.randint(220, 255),
        rng.randint(220, 255),
    )
