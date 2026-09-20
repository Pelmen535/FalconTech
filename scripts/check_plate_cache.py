"""Проверка кеша зон анонимизации: есть ли в нём вообще боксы и похожи ли они на номера.

Код возврата 1, если кеша нет или он пуст — очередь по этому коду решает, пересобирать ли.
Нужен потому, что однажды кеш собрался из 11416 ПУСТЫХ записей (не было opencv), и вся
абляция молча показала «маска ничего не меняет».
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vreid.plate import load_cache      # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "results/plate_boxes.json"
cache = load_cache(path)
if not cache:
    print(f"кеша нет или он пуст: {path}")
    raise SystemExit(1)
found = sum(1 for v in cache.values() if v)
n_box = sum(len(v) for v in cache.values())
print(f"{path}: записей {len(cache)}, зона найдена на {found / len(cache) * 100:.1f}%, боксов {n_box}")
if found == 0:
    print("ни одного бокса — маскировать нечего")
    raise SystemExit(1)
import numpy as np
a = np.array([b for v in cache.values() for b in v])
print(f"медианный бокс: центр y={np.median(a[:, 1] + a[:, 3] / 2):.2f}, "
      f"{np.median(a[:, 2]) * 100:.0f}% ширины и {np.median(a[:, 3]) * 100:.0f}% высоты кропа")
raise SystemExit(0)
