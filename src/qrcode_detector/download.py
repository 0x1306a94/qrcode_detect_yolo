from __future__ import annotations

import csv
import hashlib
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(slots=True, frozen=True)
class DownloadResult:
    url: str
    status: str
    output_path: str
    message: str = ""


def download_images_from_csv(
    csv_path: str | Path,
    output_dir: str | Path = "auto_generated_test_images/origins",
    timeout_seconds: int = 20,
) -> dict[str, object]:
    csv_file = Path(csv_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    urls = _load_urls(csv_file)
    results: list[DownloadResult] = []

    for url in urls:
        try:
            output_path, status = _download_one(url, output_root, timeout_seconds)
            results.append(DownloadResult(url=url, status=status, output_path=str(output_path)))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            results.append(DownloadResult(url=url, status="failed", output_path="", message=str(exc)))

    summary = {
        "total": len(results),
        "downloaded": sum(1 for item in results if item.status == "downloaded"),
        "skipped": sum(1 for item in results if item.status == "skipped"),
        "failed": sum(1 for item in results if item.status == "failed"),
        "items": [asdict(result) for result in results],
    }
    return summary


def _load_urls(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    seen_urls: set[str] = set()
    urls: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            url = row[0].strip()
            if not url or url.lower() == "url":
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            urls.append(url)
    return urls


def _download_one(url: str, output_root: Path, timeout_seconds: int) -> tuple[Path, str]:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {url}")

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; qrcode-detect-yolo/0.1.0)",
        },
    )

    with urlopen(request, timeout=timeout_seconds) as response:
        content_type = response.headers.get_content_type()
        url_suffix = Path(parsed_url.path).suffix.lower()
        suffix = url_suffix or mimetypes.guess_extension(content_type) or ".jpg"
        file_name = _build_file_name(url, suffix)
        output_path = output_root / file_name

        if output_path.exists():
            return output_path, "skipped"

        data = response.read()
        output_path.write_bytes(data)
        return output_path, "downloaded"


def _build_file_name(url: str, suffix: str) -> str:
    parsed_url = urlparse(url)
    stem = Path(parsed_url.path).stem or "image"
    stem = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in stem)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}{suffix}"
