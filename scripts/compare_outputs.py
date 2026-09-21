"""Сравнение двух папок сдачи: доказательство, что прогон воспроизводится.

    python scripts/compare_outputs.py out_mini_release out_mini_dev

Сравнивается то, что идёт в метрику: множество отвеченных запросов и пары
(query_id, gallery_id) в candidates.csv, полный top-10 в submission.csv и сами векторы.
Численная разница confidence печатается отдельно; её причина не устанавливается этим скриптом.
Код возврата 1, если разошлось то, что влияет на баллы.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


def rows(p: Path):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def main(a: str, b: str) -> int:
    A, B = Path(a), Path(b)
    bad = False

    ca, cb = rows(A / "candidates.csv")[1:], rows(B / "candidates.csv")[1:]
    qa, qb = {r[0] for r in ca}, {r[0] for r in cb}
    print(f"candidates: строк {len(ca)} и {len(cb)}, отвеченных запросов {len(qa)} и {len(qb)}")
    if qa != qb:
        print(f"  РАСХОЖДЕНИЕ: разные множества отказов, различий {len(qa ^ qb)}"); bad = True
    pa = {(r[0], r[1]) for r in ca}; pb = {(r[0], r[1]) for r in cb}
    print(f"  пары (query, gallery) совпадают: {len(pa & pb)}/{max(len(pa), len(pb))}")
    if pa != pb:
        bad = True
    if len(ca) == len(cb) and ca:
        d = np.abs(np.array([float(r[2]) for r in ca]) - np.array([float(r[2]) for r in cb]))
        print(f"  уверенность: среднее расхождение {d.mean():.2e}, максимум {d.max():.2e}")

    # submission.csv идёт без заголовка (официальный формат); строку-заголовок из
    # файлов прежних версий пропускаем, чтобы сравнивать одно и то же.
    drop_header = lambda r: r[1:] if r and r[0][0] == "query_id" else r
    sa = drop_header(rows(A / "submission.csv"))
    sb = drop_header(rows(B / "submission.csv"))
    t1 = sum(1 for x, y in zip(sa, sb) if x[1] == y[1])
    cells = sum(1 for x, y in zip(sa, sb) for u, v in zip(x, y) if u != v)
    print(f"submission: top-1 совпадает у {t1}/{len(sa)} запросов, различается ячеек {cells}")
    if sa != sb:
        print("  РАСХОЖДЕНИЕ: полный top-10/порядок query не совпал; влияние на mAP требует меток"); bad = True

    ea, eb = np.load(A / "embeddings.npy"), np.load(B / "embeddings.npy")
    if ea.shape != eb.shape:
        print(f"embeddings: РАСХОЖДЕНИЕ формы {ea.shape} и {eb.shape}"); return 1
    if not np.isfinite(ea).all() or not np.isfinite(eb).all():
        print("Nonfinite embeddings"); return 1
    dd = np.abs(ea - eb)
    cos = (ea * eb).sum(1) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-12)
    print(f"embeddings: форма {ea.shape}, среднее расхождение {dd.mean():.2e}, "
          f"максимум {dd.max():.2e}, минимальный косинус {cos.min():.6f}")
    if cos.min() < 0.999:
        print("  РАСХОЖДЕНИЕ: векторы разошлись больше, чем на численный шум"); bad = True

    print("\nВЫВОД: " + ("есть расхождения, влияющие на метрику" if bad else
                         "проверенные файлы совпали по указанным проверкам; accuracy без меток не измерялась"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
