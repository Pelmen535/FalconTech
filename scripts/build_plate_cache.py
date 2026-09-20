"""Один раз прогнать детектор зоны анонимизации номера по всем кропам и сложить в кеш.

    python scripts/build_plate_cache.py --config configs/hackathon.yaml --workers 8

Детекция стоит ~50 мс на кроп — это дороже всего боевого инференса (33 мс на ТС), поэтому
в бою она не нужна и не используется: кеш нужен для абляции («модель опирается на номер?»)
и для обучения с аугментацией «номер закрыт» (train.py --mask-p).

Боксы хранятся в долях кропа, так что кеш не зависит ни от разрешения кадра, ни от
ускоренного декодирования.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vreid.hackathon_data import load_test, read_annotations   # noqa: E402
from vreid.plate import build_cache, cache_key, save_cache      # noqa: E402


def _chunk(records, pad):
    return build_cache(records, pad=pad, out=None, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hackathon.yaml")
    ap.add_argument("--out", default="results/plate_boxes.json")
    ap.add_argument("--pad", type=float, default=None, help="по умолчанию — crop.pad из конфига")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="только первые N записей (для проверки)")
    a = ap.parse_args()

    import yaml
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    pad = a.pad if a.pad is not None else cfg.get("crop", {}).get("pad", 0.08)
    root = Path(ds["root"])
    recs = read_annotations(root / ds.get("train_csv", "train.csv"),
                            root / ds.get("images_dir", "images"),
                            ds.get("cols"), ds.get("bbox_format", "xywh"))
    for name, split in load_test(ds).items():
        recs += list(split.records)
    # один и тот же кроп может встретиться в train и в тесте — считаем его один раз
    seen, uniq = set(), []
    for r in recs:
        k = cache_key(r.path, r.bbox)
        if k not in seen:
            seen.add(k); uniq.append(r)
    if a.limit:
        uniq = uniq[: a.limit]
    print(f"[plate] записей {len(uniq)} (train + тест, без повторов), pad={pad}, воркеров {a.workers}")

    t0 = time.perf_counter()
    boxes: dict = {}
    if a.workers > 1:
        parts = [uniq[i::a.workers] for i in range(a.workers)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, d in enumerate(ex.map(_chunk, parts, [pad] * len(parts))):
                boxes.update(d)
                print(f"[plate] часть {i + 1}/{len(parts)} готова, всего {len(boxes)}")
    else:
        boxes = build_cache(uniq, pad=pad, out=None)
    dt = time.perf_counter() - t0
    found = sum(1 for v in boxes.values() if v)
    save_cache(boxes, a.out, pad)
    print(f"[plate] {len(boxes)} записей за {dt:.0f} с ({dt / max(len(boxes), 1) * 1000:.0f} мс/кроп), "
          f"зона найдена на {found / max(len(boxes), 1) * 100:.1f}% → {a.out}")


if __name__ == "__main__":
    main()
