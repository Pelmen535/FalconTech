from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw


VEHICLES = {
    "car_red": (195, 42, 50),
    "car_blue": (32, 84, 190),
    "car_green": (42, 150, 88),
}


def _draw_car(path: Path, color: tuple[int, int, int], camera: int, visible_plate: bool) -> None:
    image = Image.new("RGB", (240, 150), (45 + camera * 7, 48 + camera * 5, 52 + camera * 3))
    draw = ImageDraw.Draw(image)
    offset = 6 * camera
    draw.rounded_rectangle((30 + offset, 48, 205 + offset // 2, 115), radius=15, fill=color, outline=(230, 230, 230), width=3)
    draw.polygon([(65 + offset, 48), (95 + offset, 24), (160, 24), (185, 48)], fill=tuple(min(255, value + 35) for value in color))
    draw.ellipse((55 + offset, 101, 88 + offset, 134), fill=(20, 20, 20))
    draw.ellipse((165, 101, 198, 134), fill=(20, 20, 20))
    draw.rectangle((105, 91, 158, 108), fill=(245, 245, 245) if visible_plate else (127, 127, 127))
    if visible_plate:
        draw.text((113, 93), "A123", fill=(20, 20, 20))
    image.save(path)


def make_demo(out_dir: str | Path) -> Path:
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for vehicle_id, color in VEHICLES.items():
        for camera in (1, 2):
            filename = f"{vehicle_id}_cam{camera}.png"
            visible = camera == 1
            _draw_car(root / filename, color, camera, visible)
            rows.append(
                {
                    "image_path": filename,
                    "vehicle_id": vehicle_id,
                    "camera_id": f"cam{camera}",
                    "split": "query" if camera == 1 else "gallery",
                    "plate_status": "visible" if visible else "unreadable",
                    "plate_bbox": "105:91:158:108" if visible else "",
                    "vehicle_bbox": "20:15:225:140",
                }
            )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest
