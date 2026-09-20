# -*- coding: utf-8 -*-
"""Контрпример аудита: независимость запросов на РАВНЫХ эмбеддингах.

    python scripts/check_tie_independence.py

Аудит нашёл, что ре-ранжирование меняло выдачу одного запроса, если рядом в матрице
появлялись другие запросы. Причина — нестабильная сортировка соседей: при точно равных
расстояниях порядок решал introsort, а он зависит от раскладки строки в памяти, которая
сдвигается вместе с числом запросов. На 666 реальных запросах это не проявлялось: там
точных ничьих нет. Проверка строит их специально.

Три проверки:
  1. дубликаты в галерее: один запрос против того же запроса плюс 15 посторонних;
  2. перестановка запросов: та же выдача при любом порядке;
  3. пачки: результат не зависит от размера пачки (k_reciprocal_chunked).
Код возврата 1, если хоть одна не прошла.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vreid.rerank import k_reciprocal, k_reciprocal_chunked   # noqa: E402

K1, K2, LAM = 6, 2, 0.3


def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def build(seed=0, G=20, dup=4, D=16):
    rng = np.random.default_rng(seed)
    g = _norm(rng.normal(size=(G, D)).astype(np.float32))
    # точные повторы: несколько галерейных записей буквально совпадают
    for i in range(1, dup):
        g[i] = g[0]
    g[10] = g[0]
    q = _norm((g[0] + 0.01 * rng.normal(size=(1, D))).astype(np.float32))
    other = _norm(rng.normal(size=(15, D)).astype(np.float32))
    return q, other, g


def top10(d):
    return np.argsort(d[0], kind="stable")[:10]


def main():
    ok = True
    q, other, g = build()

    # 1. один запрос против «запрос + 15 посторонних»
    d_alone = k_reciprocal(q, g, K1, K2, LAM)
    d_crowd = k_reciprocal(np.concatenate([q, other]), g, K1, K2, LAM)
    same = np.array_equal(top10(d_alone), top10(d_crowd))
    delta = float(np.abs(d_alone[0] - d_crowd[0]).max())
    print(f"1. дубликаты в галерее: топ-10 {'СОВПАЛ' if same else 'РАЗОШЁЛСЯ'}, "
          f"макс. расхождение расстояний {delta:.3e}")
    print(f"   один:          {top10(d_alone).tolist()}")
    print(f"   +15 чужих:     {top10(d_crowd).tolist()}")
    ok &= same and delta < 1e-6

    # 2. перестановка остальных запросов
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(other))
    d_perm = k_reciprocal(np.concatenate([q, other[perm]]), g, K1, K2, LAM)
    same2 = np.array_equal(top10(d_crowd), top10(d_perm))
    d2 = float(np.abs(d_crowd[0] - d_perm[0]).max())
    print(f"2. перестановка чужих запросов: топ-10 {'СОВПАЛ' if same2 else 'РАЗОШЁЛСЯ'}, "
          f"макс. расхождение {d2:.3e}")
    ok &= same2 and d2 < 1e-6

    # 3. пачки не меняют результат
    Q = np.concatenate([q, other])
    full = k_reciprocal(Q, g, K1, K2, LAM)
    worst = 0.0
    for c in (1, 3, 7, 16):
        ch = k_reciprocal_chunked(Q, g, K1, K2, LAM, chunk=c)
        worst = max(worst, float(np.abs(full - ch).max()))
    print(f"3. пачки 1/3/7/16 против одного куска: макс. расхождение {worst:.3e}")
    ok &= worst < 1e-6

    print("\n" + ("ВЫВОД: запросы независимы и на точных ничьих, и при любом размере пачки"
                  if ok else "ВЫВОД: ПРОВАЛ — выдача зависит от остальных запросов"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
