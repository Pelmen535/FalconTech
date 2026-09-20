"""Доказательство независимости запросов (требование организаторов, ответ 14).

    python scripts/check_query_independence.py --submission submission_soup --data Данные

Берём выданные эмбеддинги, прогоняем ре-ранжирование дважды: по полному набору запросов и по
случайному ПОДНАБОРУ (плюс перестановка порядка). Для запросов, попавших в оба прогона, топ-10
обязан совпасть до последнего кандидата. Если совпал — ранжирование одного запроса не зависит
от наличия, отсутствия и порядка остальных, то есть кластеризации по test_query нет.

Код возврата 1, если хоть один топ-10 разошёлся.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vreid.rerank import k_reciprocal          # noqa: E402
from vreid.rerank import frame_block_mask      # noqa: E402


def keys(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["image_id"] for r in csv.DictReader(f)]


def rank(q, g, qk, gk, rec, topk=10):
    block = frame_block_mask(list(qk) + list(gk), list(qk) + list(gk))
    np.fill_diagonal(block, True)
    qg = block[: len(qk), len(qk):]
    sims = q @ g.T
    sims[qg] = -np.inf
    if rec.get("kr"):
        d = k_reciprocal(q, g, int(rec["k1"]), int(rec["k2"]), 0.3, same_mask=block)
        d[~np.isfinite(sims)] = np.inf
        sims = -d
    return np.argsort(-sims, axis=1, kind="stable")[:, :topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--release", default=None, help="по умолчанию рецепт берётся из run_info.json")
    ap.add_argument("--frac", type=float, default=0.6, help="доля запросов во втором прогоне")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sub, data = Path(a.submission), Path(a.data)
    # Рецепт ищем по порядку: явный --release, recipe.json рядом со сдачей, run_info.json.
    # Чужая сдача (например из release_calibrated_v1) run_info.json не содержит.
    rec = None
    for cand in ([Path(a.release) / "recipe.json"] if a.release else []) + [sub / "recipe.json"]:
        if cand.exists():
            rec = json.loads(cand.read_text(encoding="utf-8")); break
    if rec is None and (sub / "run_info.json").exists():
        rec = json.loads((sub / "run_info.json").read_text(encoding="utf-8"))["recipe"]
    if rec is None:
        raise SystemExit(f"не нашёл рецепт: ни --release, ни {sub}/recipe.json, ни {sub}/run_info.json")
    E = np.load(sub / "embeddings.npy")
    qk, gk = keys(data / "test_query.csv"), keys(data / "test_gallery.csv")
    q, g = E[: len(qk)], E[len(qk):]
    print(f"запросов {len(qk)}, галерея {len(gk)}, k-reciprocal={rec.get('kr')} "
          f"({rec.get('k1')}/{rec.get('k2')})")

    full = rank(q, g, qk, gk, rec)
    rng = np.random.default_rng(a.seed)
    sel = rng.permutation(len(qk))[: int(a.frac * len(qk))]
    part = rank(q[sel], g, [qk[i] for i in sel], gk, rec)

    bad = 0
    for row, i in enumerate(sel):
        if not np.array_equal(full[i], part[row]):
            bad += 1
            if bad <= 3:
                print(f"  РАСХОЖДЕНИЕ у запроса {qk[i]}: полный {full[i][:5]} против {part[row][:5]}")
    print(f"второй прогон: {len(sel)} запросов из {len(qk)} в другом порядке")
    print(f"топ-10 совпал у {len(sel) - bad}/{len(sel)} запросов")
    if bad:
        print("\nВЫВОД: ранжирование запроса зависит от других запросов — это кластеризация по test_query")
        return 1
    print("\nВЫВОД: запросы независимы — удаление и перестановка остальных не меняют выдачу")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
