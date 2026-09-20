"""Игрушечный датасет для проверки стенда без GPU и без скачивания.

"Машина" = цветной прямоугольник кузова с окнами, колёсами и маленькой наклейкой,
"камера" = свой фон, масштаб, яркость и, у половины камер, зеркальный ракурс.
Файлы: <vid>_c<cam>_<k>.png. Формат такой же, как у VeRi (vid, cam в имени).

    python scripts/make_toy_dataset.py --root data/toy --ids 60 --cams 6
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def draw_car(rng: random.Random, spec: dict, cam: dict, size=(128, 96)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, cam["bg"])
    d = ImageDraw.Draw(img)
    s = cam["scale"]
    cw, ch = int(W * 0.8 * s), int(H * 0.45 * s * spec["height"])
    x0, y0 = (W - cw) // 2 + rng.randint(-4, 4), H // 2 - ch // 2 + rng.randint(-3, 3)
    body = tuple(min(255, max(0, int(c * cam["light"]))) for c in spec["color"])
    d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=body)
    # крыша/окна
    rx0, rx1 = x0 + int(cw * 0.25), x0 + int(cw * 0.7)
    d.rectangle([rx0, y0 - int(ch * 0.5), rx1, y0], fill=body)
    d.rectangle([rx0 + 4, y0 - int(ch * 0.45), rx1 - 4, y0 - 3], fill=(40, 60, 90))
    # колёса
    r = max(4, int(ch * 0.35))
    for cx in (x0 + int(cw * 0.2), x0 + int(cw * 0.8)):
        d.ellipse([cx - r, y0 + ch - r, cx + r, y0 + ch + r], fill=(20, 20, 20))
    # наклейка / отличительный признак
    if spec["sticker"]:
        sx, sy = x0 + int(cw * spec["sticker_pos"]), y0 + int(ch * 0.3)
        d.rectangle([sx, sy, sx + 8, sy + 8], fill=spec["sticker"])
    if cam["mirror"]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/toy")
    ap.add_argument("--ids", type=int, default=60)
    ap.add_argument("--cams", type=int, default=6)
    ap.add_argument("--per-cam", type=int, default=2, help="кадров на машину на камеру")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.root)
    (root / "query").mkdir(parents=True, exist_ok=True)
    (root / "gallery").mkdir(parents=True, exist_ok=True)

    palette = [(200, 200, 200), (30, 30, 30), (180, 20, 20), (20, 60, 180), (240, 240, 240),
               (120, 120, 130), (20, 120, 40), (220, 180, 30), (90, 40, 20), (150, 90, 160)]
    cams = []
    for c in range(a.cams):
        cams.append({"bg": (rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200)),
                     "scale": rng.uniform(0.7, 1.0), "light": rng.uniform(0.7, 1.2),
                     "mirror": c % 2 == 1})
    n = 0
    for vid in range(1, a.ids + 1):
        spec = {"color": rng.choice(palette), "height": rng.uniform(0.8, 1.2),
                "sticker": rng.choice([None, None, (255, 255, 0), (0, 255, 255), (255, 0, 255)]),
                "sticker_pos": rng.uniform(0.3, 0.6)}
        q_cam = rng.randrange(a.cams)
        for cam in range(a.cams):
            for k in range(a.per_cam):
                img = draw_car(rng, spec, cams[cam])
                split = "query" if (cam == q_cam and k == 0) else "gallery"
                img.save(root / split / f"{vid:04d}_c{cam + 1:03d}_{k}.png")
                n += 1
    print(f"toy dataset: {n} изображений, {a.ids} id, {a.cams} камер → {root}")


if __name__ == "__main__":
    main()
