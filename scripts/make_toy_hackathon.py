"""Игрушечные данные в формате организаторов (ТЗ §5) — для проверки всей цепочки без реальных данных.

    images/<random>.jpg   полные кадры: фон «камеры» + 1–3 машины
    train.csv             image_id,x,y,w,h,vehicle_id
    test.csv              image_id,x,y,w,h
    _truth_test.csv       (для себя) vehicle_id и camera_id тестовых bbox — организаторы этого не дают

    python scripts/make_toy_hackathon.py --root data/toy_hack --ids 80 --cams 5
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_toy_dataset import draw_car  # noqa: E402

PALETTE = [(200, 200, 200), (30, 30, 30), (180, 20, 20), (20, 60, 180), (240, 240, 240),
           (120, 120, 130), (20, 120, 40), (220, 180, 30), (90, 40, 20), (150, 90, 160)]


def make_camera(rng: random.Random, i: int) -> dict:
    W, H = rng.choice([(640, 480), (800, 450), (960, 540)])
    bg = Image.new("RGB", (W, H), (rng.randint(50, 180), rng.randint(50, 180), rng.randint(50, 180)))
    d = ImageDraw.Draw(bg)
    for _ in range(12):  # «здания», чтобы фон у камер различался
        x0, y0 = rng.randint(0, W - 40), rng.randint(0, H // 2)
        d.rectangle([x0, y0, x0 + rng.randint(20, 120), y0 + rng.randint(20, 120)],
                    fill=(rng.randint(40, 200),) * 3)
    d.rectangle([0, H * 2 // 3, W, H], fill=(rng.randint(60, 90),) * 3)  # дорога
    return {"bg": bg, "W": W, "H": H, "scale": rng.uniform(0.7, 1.0), "light": rng.uniform(0.7, 1.2),
            "mirror": i % 2 == 1, "id": i + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/toy_hack")
    ap.add_argument("--ids", type=int, default=80)
    ap.add_argument("--cams", type=int, default=5)
    ap.add_argument("--views", type=int, default=3, help="сколько камер видит каждую машину")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    cams = [make_camera(rng, i) for i in range(a.cams)]

    specs = {}
    for vid in range(1, a.ids + 1):
        specs[vid] = {"color": rng.choice(PALETTE), "height": rng.uniform(0.8, 1.2),
                      "sticker": rng.choice([None, None, (255, 255, 0), (0, 255, 255), (255, 0, 255)]),
                      "sticker_pos": rng.uniform(0.3, 0.6)}
    ids = list(specs)
    rng.shuffle(ids)
    n_test = int(len(ids) * a.test_frac)
    test_ids, train_ids = set(ids[:n_test]), set(ids[n_test:])

    # каждая машина появляется на a.views камерах; в кадре 1–3 машины с одной камеры
    appearances = []  # (vid, cam_idx)
    for vid in ids:
        for c in rng.sample(range(a.cams), min(a.views, a.cams)):
            appearances.append((vid, c))
    rng.shuffle(appearances)
    by_cam = {c: [] for c in range(a.cams)}
    for vid, c in appearances:
        by_cam[c].append(vid)

    rows_train, rows_test, rows_truth = [], [], []
    for c, vids in by_cam.items():
        cam = cams[c]
        i = 0
        while i < len(vids):
            group = vids[i:i + rng.randint(1, 3)]
            i += len(group)
            # тест и трейн не смешиваем в одном кадре — так проще, у организаторов может быть иначе
            frame_split = "test" if group[0] in test_ids else "train"
            group = [v for v in group if (v in test_ids) == (frame_split == "test")]
            frame = cam["bg"].copy()
            image_id = uuid.uuid4().hex[:12]
            slots = rng.sample(range(3), len(group))
            for v, slot in zip(group, slots):
                car = draw_car(rng, specs[v], {"bg": (0, 0, 0), "scale": cam["scale"],
                                               "light": cam["light"], "mirror": cam["mirror"]},
                               size=(128, 96))
                # вырезаем машину из чёрного фона по маске
                mask = car.convert("L").point(lambda p: 255 if p > 8 else 0)
                bw, bh = car.size
                x = int(cam["W"] * (0.08 + 0.3 * slot) + rng.randint(-10, 10))
                y = int(cam["H"] * 0.55 + rng.randint(-10, 10))
                frame.paste(car, (x, y), mask)
                row = {"image_id": image_id, "x": x, "y": y, "w": bw, "h": bh}
                if frame_split == "train":
                    rows_train.append({**row, "vehicle_id": v, "camera_id": cam["id"]})
                else:
                    rows_test.append(row)
                    rows_truth.append({**row, "vehicle_id": v, "camera_id": cam["id"]})
            frame.save(root / "images" / f"{image_id}.jpg", quality=85)

    # как у организаторов: test_query / test_gallery (в галерее у части машин пары нет)
    by_v = {}
    for r in rows_truth:
        by_v.setdefault(r["vehicle_id"], []).append(r)
    rows_q, rows_g = [], []
    for k, (v, rs) in enumerate(sorted(by_v.items())):
        rows_q.append(rs[0])
        if k % 5 != 0:            # каждая пятая машина — без пары в галерее
            rows_g.extend(rs[1:])
    for name, rows, cols in (("train.csv", rows_train, ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]),
                             ("test_query.csv", rows_q, ["image_id", "x", "y", "w", "h"]),
                             ("test_gallery.csv", rows_g, ["image_id", "x", "y", "w", "h"]),
                             ("test.csv", rows_test, ["image_id", "x", "y", "w", "h"]),
                             ("_truth_test.csv", rows_truth, ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"])):
        with open(root / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"toy_hack: train {len(rows_train)} bbox / {len(train_ids)} id, test {len(rows_test)} bbox / {len(test_ids)} id, "
          f"кадров {len(list((root / 'images').iterdir()))} → {root}")


if __name__ == "__main__":
    main()
