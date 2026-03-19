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


@dataclass(slots=True, frozen=True)
class SyntheticQRCodeConfig:
    corner_ratio: float = 0.35
    center_region_x_range: tuple[float, float] = (0.25, 0.75)
    center_region_y_range: tuple[float, float] = (0.35, 0.75)
    placement_regions: tuple[str, ...] = (
        "bottom_left",
        "bottom_right",
        "top_left",
        "top_right",
        "center",
    )
    placement_weights: tuple[float, ...] = (0.28, 0.28, 0.12, 0.12, 0.20)
    size_buckets: tuple[tuple[float, float], ...] = (
        (0.08, 0.16),
        (0.16, 0.28),
        (0.28, 0.40),
    )
    size_bucket_weights: tuple[float, ...] = (0.4, 0.35, 0.25)
    error_correction_levels: tuple[str, ...] = ("L", "M", "Q", "H")
    error_correction_weights: tuple[float, ...] = (0.15, 0.35, 0.25, 0.25)
    content_length_ranges: tuple[tuple[int, int], ...] = (
        (12, 24),
        (25, 64),
        (65, 160),
    )
    content_length_weights: tuple[float, ...] = (0.3, 0.5, 0.2)
    variants_per_image: int = 1
    use_colored_qrcode: bool = False
    use_alpha_blend: bool = False

    def validate(self) -> None:
        if not 0.0 < self.corner_ratio <= 1.0:
            raise ValueError("corner_ratio must be in (0, 1]")
        if len(self.placement_regions) != len(self.placement_weights):
            raise ValueError("placement_regions and placement_weights must have the same length")
        if len(self.size_buckets) != len(self.size_bucket_weights):
            raise ValueError("size_buckets and size_bucket_weights must have the same length")
        if len(self.error_correction_levels) != len(self.error_correction_weights):
            raise ValueError("error_correction_levels and error_correction_weights must have the same length")
        if len(self.content_length_ranges) != len(self.content_length_weights):
            raise ValueError("content_length_ranges and content_length_weights must have the same length")
        if self.variants_per_image <= 0:
            raise ValueError("variants_per_image must be positive")
        if not 0.0 <= self.center_region_x_range[0] < self.center_region_x_range[1] <= 1.0:
            raise ValueError("center_region_x_range must be in [0, 1] and ascending")
        if not 0.0 <= self.center_region_y_range[0] < self.center_region_y_range[1] <= 1.0:
            raise ValueError("center_region_y_range must be in [0, 1] and ascending")


def _require_pillow() -> None:
    if Image is None:  # pragma: no cover
        raise RuntimeError(
            "Pillow is required for synthetic generation. Install with: pip install -e '.[synthetic]'"
        )


def generate_qrcode_image(
    output_path: str | Path,
    seed: int | None = None,
    error_correction_level: str = "M",
    content_length_range: tuple[int, int] = (25, 64),
    use_colored_qrcode: bool = False,
) -> str:
    try:
        import qrcode
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qrcode is required for QR generation. Install with: pip install -e '.[synthetic]'"
        ) from exc

    rng = random.Random(seed)
    qr_content = _random_qrcode_content(rng, content_length_range)
    if use_colored_qrcode:
        fill_color = _random_dark_color(rng)
        back_color = _random_light_color(rng)
    else:
        fill_color = (0, 0, 0)
        back_color = (255, 255, 255)

    qr = qrcode.QRCode(
        version=None,
        error_correction=_resolve_error_correction_level(qrcode, error_correction_level),
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
    placement_region: str = "bottom_left",
    corner_ratio: float = 0.35,
    center_region_x_range: tuple[float, float] = (0.25, 0.75),
    center_region_y_range: tuple[float, float] = (0.35, 0.75),
    scale_range: tuple[float, float] = (0.16, 0.28),
    use_alpha_blend: bool = False,
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
            qrcode = _apply_random_style(qrcode, rng, use_alpha_blend)

            x_start, x_end, y_start, y_end = _resolve_placement_bounds(
                placement_region=placement_region,
                image_width=background_width,
                image_height=background_height,
                target_size=target_size,
                corner_ratio=corner_ratio,
                center_region_x_range=center_region_x_range,
                center_region_y_range=center_region_y_range,
            )

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
    config: SyntheticQRCodeConfig | None = None,
) -> list[dict[str, str | int | float]]:
    _require_pillow()
    current_config = config or SyntheticQRCodeConfig()
    current_config.validate()
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    rng = random.Random(seed)
    image_paths = list(_iter_image_paths(input_root, recursive))
    if not image_paths:
        raise ValueError(f"No images found under: {input_root}")

    results: list[dict[str, str | int | float]] = []
    for index, image_path in enumerate(image_paths):
        relative_path = image_path.relative_to(input_root)
        for variant_index in range(current_config.variants_per_image):
            output_stem = f"{image_path.stem}_synthetic_{variant_index}" if current_config.variants_per_image > 1 else f"{image_path.stem}_synthetic"
            output_image_path = output_root / relative_path.parent / f"{output_stem}{image_path.suffix}"
            generated_qrcode_path = output_root / "_generated_qrcodes" / relative_path.parent / f"{output_stem}_qrcode.png"

            current_seed = rng.randint(0, 10**9)
            target_bucket = _choose_weighted_item(rng, current_config.size_buckets, current_config.size_bucket_weights)
            placement_region = _choose_weighted_item(
                rng,
                current_config.placement_regions,
                current_config.placement_weights,
            )
            error_correction_level = _choose_weighted_item(
                rng,
                current_config.error_correction_levels,
                current_config.error_correction_weights,
            )
            content_length_range = _choose_weighted_item(
                rng,
                current_config.content_length_ranges,
                current_config.content_length_weights,
            )

            qr_content = generate_qrcode_image(
                generated_qrcode_path,
                seed=current_seed,
                error_correction_level=error_correction_level,
                content_length_range=content_length_range,
                use_colored_qrcode=current_config.use_colored_qrcode,
            )
            placement = compose_synthetic_qrcode(
                background_path=image_path,
                qrcode_path=generated_qrcode_path,
                output_path=output_image_path,
                seed=current_seed,
                placement_region=placement_region,
                corner_ratio=current_config.corner_ratio,
                center_region_x_range=current_config.center_region_x_range,
                center_region_y_range=current_config.center_region_y_range,
                scale_range=target_bucket,
                use_alpha_blend=current_config.use_alpha_blend,
            )

            if write_labelme_json:
                labelme_path = output_image_path.with_suffix(".json")
                write_labelme_rectangle(labelme_path, output_image_path, placement)

            results.append({
                "index": index,
                "variant_index": variant_index,
                "input_image": str(image_path),
                "output_image": str(output_image_path),
                "qrcode_image": str(generated_qrcode_path),
                "qrcode_content": qr_content,
                "placement_region": placement_region,
                "error_correction_level": error_correction_level,
                "content_length": len(qr_content),
                "target_scale_min": target_bucket[0],
                "target_scale_max": target_bucket[1],
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


def _apply_random_style(image: "Image.Image", rng: random.Random, use_alpha_blend: bool) -> "Image.Image":
    if not use_alpha_blend:
        return image
    alpha = image.getchannel("A")
    alpha = alpha.point(lambda pixel: int(pixel * rng.uniform(0.88, 1.0)))
    image.putalpha(alpha)
    return image


def _resolve_placement_bounds(
    placement_region: str,
    image_width: int,
    image_height: int,
    target_size: int,
    corner_ratio: float,
    center_region_x_range: tuple[float, float],
    center_region_y_range: tuple[float, float],
) -> tuple[int, int, int, int]:
    left_area_end = int(image_width * corner_ratio)
    right_area_start = int(image_width * (1.0 - corner_ratio))
    top_area_end = int(image_height * corner_ratio)
    bottom_area_start = int(image_height * (1.0 - corner_ratio))

    if placement_region == "bottom_left":
        return 0, max(0, left_area_end), bottom_area_start, image_height
    if placement_region == "bottom_right":
        return right_area_start, image_width, bottom_area_start, image_height
    if placement_region == "top_left":
        return 0, max(0, left_area_end), 0, max(0, top_area_end)
    if placement_region == "top_right":
        return right_area_start, image_width, 0, max(0, top_area_end)
    if placement_region == "center":
        center_x_start = int(image_width * center_region_x_range[0])
        center_x_end = int(image_width * center_region_x_range[1])
        center_y_start = int(image_height * center_region_y_range[0])
        center_y_end = int(image_height * center_region_y_range[1])
        return center_x_start, center_x_end, center_y_start, center_y_end
    raise ValueError(f"Unsupported placement_region: {placement_region}")


def _choose_weighted_item(rng: random.Random, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def _iter_image_paths(input_root: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield path


def _random_qrcode_content(rng: random.Random, content_length_range: tuple[int, int]) -> str:
    min_length, max_length = content_length_range
    target_length = rng.randint(min_length, max_length)
    builders = (
        _build_short_url_content,
        _build_query_url_content,
        _build_mini_program_style_content,
        _build_plain_text_content,
    )
    builder = rng.choice(builders)
    return builder(rng, target_length)


def _build_short_url_content(rng: random.Random, target_length: int) -> str:
    token = _random_token(rng, max(6, target_length - 23))
    return _trim_or_pad(f"https://sho.ws/{token}", rng, target_length)


def _build_query_url_content(rng: random.Random, target_length: int) -> str:
    event_token = _random_token(rng, 8)
    user_token = _random_token(rng, 8)
    content = f"https://example.com/event/{event_token}?channel=promo&ref={user_token}"
    return _trim_or_pad(content, rng, target_length)


def _build_mini_program_style_content(rng: random.Random, target_length: int) -> str:
    event_token = _random_token(rng, 10)
    scene_token = _random_token(rng, 12)
    content = f"pages/event/detail?id={event_token}&scene={scene_token}"
    return _trim_or_pad(content, rng, target_length)


def _build_plain_text_content(rng: random.Random, target_length: int) -> str:
    prefix = rng.choice(("EVENT-", "LIVE-", "SHOW-"))
    token = _random_token(rng, max(8, target_length - len(prefix)))
    return _trim_or_pad(f"{prefix}{token}", rng, target_length)


def _trim_or_pad(content: str, rng: random.Random, target_length: int) -> str:
    if len(content) > target_length:
        return content[:target_length]
    if len(content) < target_length:
        return content + _random_token(rng, target_length - len(content))
    return content


def _random_token(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def _resolve_error_correction_level(qrcode_module, level: str):
    mapping = {
        "L": qrcode_module.constants.ERROR_CORRECT_L,
        "M": qrcode_module.constants.ERROR_CORRECT_M,
        "Q": qrcode_module.constants.ERROR_CORRECT_Q,
        "H": qrcode_module.constants.ERROR_CORRECT_H,
    }
    normalized_level = level.upper()
    if normalized_level not in mapping:
        raise ValueError(f"Unsupported error correction level: {level}")
    return mapping[normalized_level]


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
