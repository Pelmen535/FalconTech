"""Сборка демонстрационной галереи из выданного организаторами набора.

Зачем так, а не архивом в репозитории. Демонстрация «за два клика» раньше держалась на
`demo/gallery.zip` с двадцатью настоящими кадрами организаторов. Выкладывать их публично
нельзя: это их данные, выданные участникам под их условия, и на снимках уличных камер
различимы машины и люди. Но у проверяющего набор есть — он его и выдавал. Поэтому архив не
поставляется, а собирается на месте одной командой из той же папки, что подаётся в
`vreid.predict`.

    python scripts/make_demo_gallery.py --data Данные

После этого в интерфейсе работает «Галерея → Загрузить демо-галерею» и «Поиск → Взять
демонстрационный кадр». Ничего скачивать не нужно.

Что собирается. Берутся первые `--count` строк `test_gallery.csv` и один кадр из
`test_query.csv` под запрос. Порядок строк не перемешивается: так набор воспроизводим, и
два человека на одних данных получат одинаковый архив. Это демонстрация исполнения, а не
измерение точности — совпадения в этой выборке никто не гарантирует.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="папка организаторов: images/, test_query.csv, test_gallery.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "demo")
    parser.add_argument("--count", type=int, default=20, help="сколько кадров в галерее")
    parser.add_argument("--query-index", type=int, default=0,
                        help="какую строку test_query.csv взять под демонстрационный запрос")
    args = parser.parse_args()

    images = args.data / "images"
    gallery_csv = args.data / "test_gallery.csv"
    query_csv = args.data / "test_query.csv"
    for path in (images, gallery_csv, query_csv):
        if not path.exists():
            print(f"[demo] не нашёл {path}. Укажи --data на папку организаторов: в ней должны "
                  f"лежать images/, test_query.csv и test_gallery.csv")
            return 1

    gallery = read_rows(gallery_csv)[:args.count]
    queries = read_rows(query_csv)
    if not gallery or args.query_index >= len(queries):
        print("[demo] во входных CSV не хватает строк")
        return 1
    query = queries[args.query_index]

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "gallery.zip"
    missing = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # CSV внутри архива — тот же формат, что у организаторов: сервис разбирает его
        # тем же кодом, что и конкурсный прогон, и подменять формат ради демо нельзя.
        rows = ["image_id,x,y,w,h"]
        for row in gallery:
            source = images / f"{row['image_id']}.jpg"
            if not source.is_file():
                missing.append(row["image_id"])
                continue
            rows.append(f"{row['image_id']},{row['x']},{row['y']},{row['w']},{row['h']}")
            zf.write(source, f"images/{row['image_id']}.jpg")
        zf.writestr("test_gallery.csv", "\n".join(rows) + "\n")

    query_image = images / f"{query['image_id']}.jpg"
    if not query_image.is_file():
        print(f"[demo] нет кадра запроса {query_image}")
        return 1
    shutil.copyfile(query_image, args.out / "query.jpg")
    (args.out / "query.json").write_text(json.dumps({
        "image": "query.jpg",
        "image_id": query["image_id"],
        "bbox": [int(float(query["x"])), int(float(query["y"])),
                 int(float(query["w"])), int(float(query["h"]))],
        "scope": ("демонстрация исполнения на данных организаторов; "
                  "совпадение в этой выборке не гарантировано и точность этим не меряется"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    size = archive.stat().st_size / 2 ** 20
    print(f"[demo] {archive}: {len(gallery) - len(missing)} кадров, {size:.1f} МиБ")
    if missing:
        print(f"[demo] пропущено {len(missing)} строк — не нашлись файлы кадров")
    print(f"[demo] {args.out / 'query.jpg'} и query.json: запрос {query['image_id']}")
    print("[demo] готово. В интерфейсе: «Галерея» → «Загрузить демо-галерею», "
          "затем «Поиск» → «Взять демонстрационный кадр»")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
