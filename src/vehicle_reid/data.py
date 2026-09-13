from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


ALLOWED_SPLITS = {"train", "query", "gallery"}
ALLOWED_PLATE_STATUSES = {"absent", "unreadable", "masked", "visible"}
REQUIRED_COLUMNS = {"image_path", "vehicle_id", "camera_id", "split", "plate_status"}


@dataclass(frozen=True)
class Record:
    image_path: Path
    vehicle_id: str
    camera_id: str
    split: str
    plate_status: str
    plate_bbox: tuple[int, int, int, int] | None = None
    vehicle_bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class AuditReport:
    records: int
    identities: int
    cameras: int
    queries: int
    gallery: int
    positive_queries: int
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "identities": self.identities,
            "cameras": self.cameras,
            "queries": self.queries,
            "gallery": self.gallery,
            "positive_queries": self.positive_queries,
            "warnings": list(self.warnings),
        }


def _parse_bbox(value: str, field: str, row_number: int) -> tuple[int, int, int, int] | None:
    value = value.strip()
    if not value:
        return None
    try:
        parts = tuple(int(part) for part in value.split(":"))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must contain integers x1:y1:x2:y2") from exc
    if len(parts) != 4 or parts[0] < 0 or parts[1] < 0 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise ValueError(f"row {row_number}: invalid {field}={value!r}")
    return parts


def load_manifest(path: str | Path) -> list[Record]:
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    records: list[Record] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            image_path = Path(row["image_path"].strip())
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            split = row["split"].strip().lower()
            plate_status = row["plate_status"].strip().lower()
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"row {row_number}: unsupported split {split!r}")
            if plate_status not in ALLOWED_PLATE_STATUSES:
                raise ValueError(f"row {row_number}: unsupported plate_status {plate_status!r}")
            record = Record(
                image_path=image_path,
                vehicle_id=row["vehicle_id"].strip(),
                camera_id=row["camera_id"].strip(),
                split=split,
                plate_status=plate_status,
                plate_bbox=_parse_bbox(row.get("plate_bbox", ""), "plate_bbox", row_number),
                vehicle_bbox=_parse_bbox(row.get("vehicle_bbox", ""), "vehicle_bbox", row_number),
            )
            if not record.vehicle_id or not record.camera_id:
                raise ValueError(f"row {row_number}: vehicle_id and camera_id are required")
            if record.plate_status == "visible" and record.plate_bbox is None:
                raise ValueError(f"row {row_number}: visible plate requires plate_bbox")
            if not record.image_path.is_file():
                raise FileNotFoundError(f"row {row_number}: image not found: {record.image_path}")
            records.append(record)
    if not records:
        raise ValueError("manifest contains no records")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_records(records: Iterable[Record]) -> AuditReport:
    rows = list(records)
    train_ids = {row.vehicle_id for row in rows if row.split == "train"}
    eval_ids = {row.vehicle_id for row in rows if row.split in {"query", "gallery"}}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(f"identity leakage between train and evaluation: {', '.join(overlap[:10])}")

    seen_hashes: dict[str, Record] = {}
    for row in rows:
        digest = _sha256(row.image_path)
        previous = seen_hashes.get(digest)
        if previous and {previous.split, row.split} <= {"query", "gallery"}:
            raise ValueError(f"duplicate evaluation image: {previous.image_path} and {row.image_path}")
        seen_hashes[digest] = row

    queries = [row for row in rows if row.split == "query"]
    gallery = [row for row in rows if row.split == "gallery"]
    positives = sum(
        any(candidate.vehicle_id == query.vehicle_id and candidate.camera_id != query.camera_id for candidate in gallery)
        for query in queries
    )
    warnings: list[str] = []
    if not queries:
        warnings.append("no query rows")
    if not gallery:
        warnings.append("no gallery rows")
    if queries and positives < len(queries):
        warnings.append(f"{len(queries) - positives} queries have no cross-camera positive in gallery")
    return AuditReport(
        records=len(rows),
        identities=len({row.vehicle_id for row in rows}),
        cameras=len({row.camera_id for row in rows}),
        queries=len(queries),
        gallery=len(gallery),
        positive_queries=positives,
        warnings=tuple(warnings),
    )


def load_vehicle_image(record: Record) -> Image.Image:
    image = Image.open(record.image_path).convert("RGB")
    if record.plate_bbox is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(record.plate_bbox, fill=(127, 127, 127))
    if record.vehicle_bbox is not None:
        image = image.crop(record.vehicle_bbox)
    return image


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0
