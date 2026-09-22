"""Синтетические входные CSV для тестов и CI — вместо настоящих данных организаторов.

Зачем. Тесты и проверка формата сдачи читают test_query.csv и test_gallery.csv. Раньше там
лежали НАСТОЯЩИЕ файлы организаторов: их image_id и их размеченные рамки. Публиковать это
нельзя — это часть их датасета. Но проверяется в этих тестах только ФОРМА: покрытие всех
запросов, порядок строк embeddings.npy, ровно десять кандидатов на запрос, вид candidates.csv.
Для формы достаточно идентификаторов и рамок правильного вида.

Числа строк (1110 и 750) взяты такими же, как в опубликованном наборе, чтобы проверки
работали на той же размерности. Сами идентификаторы и рамки случайные, seed фиксирован —
файлы воспроизводятся побайтово.

    python scripts/make_test_fixtures.py
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260922
FRAME = (1920, 1080)


def make_rows(rng: random.Random, count: int) -> list[tuple]:
    rows = []
    for _ in range(count):
        image_id = "".join(rng.choice("0123456789abcdef") for _ in range(32))
        w = rng.randint(180, 980)
        h = rng.randint(140, int(w * 0.95))
        x = rng.randint(0, FRAME[0] - w)
        y = rng.randint(0, FRAME[1] - h)
        rows.append((image_id, x, y, w, h))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "tests/service/fixtures/inputs_csv")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, count in (("test_query.csv", 1110), ("test_gallery.csv", 750)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows(make_rows(rng, count))
        print(f"[fixtures] {name}: {count} строк")
    print(f"[fixtures] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
