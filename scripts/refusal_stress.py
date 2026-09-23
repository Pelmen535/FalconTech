# -*- coding: utf-8 -*-
"""Сколько стоит порог отказа, если в закрытом тесте не будет одно-камерных дубликатов.

    python scripts/refusal_stress.py --run runs/hack/ft_soup_r392dw40_fit --release release

ОТКУДА ВОПРОС. В нашей валидации (и в открытом тесте) галерея собрана как один кадр на трек
«машина × камера». Поэтому у запроса часто лежит в галерее кадр ТОЙ ЖЕ машины с ТОЙ ЖЕ
камеры — почти дубликат с высоким косинусом. Оптимум на таком распределении может
плохо переноситься на закрытый тест; замороженный порог читайте из рецепта.

Организаторы про закрытый тест гарантируют другое: у matched-запросов есть хотя бы один
КРОСС-КАМЕРНЫЙ положительный (ответ 17). Про одно-камерные кадры не сказано ничего. Если их
там не окажется, уверенность у правильных ответов упадёт примерно с 0.91 до 0.51, и высокий
порог начнёт отказывать тем, кому отвечать надо.

Скрипт считает балл отказа 0.7·F1 + 0.3·TNR в двух сценариях на одних и тех же эмбеддингах:
  A — как сейчас: одно-камерные кадры в галерее есть;
  B — стресс: у каждого запроса убраны кадры своей машины со своей камеры, ЧУЖИЕ машины
      с этой камеры остаются (то есть трудные негативы никуда не делись).
и печатает таблицу по сетке порогов с ожиданием при разной вере в сценарий A.

Это не измерение закрытого теста, а оценка цены гипотезы. Решение о пороге принимается по
этой таблице и записывается в recipe.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vreid.artifacts import release_twin_run  # noqa: E402 - двойник текущего релиза
from vreid.rerank import frame_block_mask   # noqa: E402


def block_score(s, has, gv, qv, thr):
    top1 = s.max(axis=1)
    corr = gv[np.argmax(s, axis=1)] == qv
    acc = top1 >= thr
    tp = int((acc & has & corr).sum()); fp = int((acc & ~(has & corr)).sum())
    fn = int((~acc & has).sum()); tn = int((~acc & ~has).sum()); neg = int((~has).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12); tnr = tn / max(neg, 1)
    return dict(score=100 * (0.7 * f1 + 0.3 * tnr), f1=f1 * 100, tnr=tnr * 100,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(release_twin_run()))
    ap.add_argument("--release", default="release")
    ap.add_argument("--beliefs", type=float, nargs="+", default=[0.70, 0.85, 0.95])
    ap.add_argument("--out", default="results/refusal_stress.json")
    a = ap.parse_args()

    run = Path(a.run)
    qz, gz = np.load(run / "val_query_track.npz"), np.load(run / "val_gallery_track.npz")
    q, g = qz["emb"], gz["emb"]
    qv, qc, gv, gc = qz["vids"], qz["cams"], gz["vids"], gz["cams"]
    qk, gk = list(qz["keys"]), list(gz["keys"])
    blk = frame_block_mask(qk + gk, qk + gk); np.fill_diagonal(blk, True)
    raw = q @ g.T; raw[blk[: len(qk), len(qk):]] = -np.inf

    same_v = gv[None, :] == qv[:, None]
    same_c = gc[None, :] == qc[:, None]
    has_a = same_v.any(axis=1)
    has_b = (same_v & ~same_c).any(axis=1)
    raw_b = np.where(same_v & same_c, -np.inf, raw)
    n_dup = int((has_b & (same_v & same_c).any(axis=1)).sum())
    print(f"запросов {len(qv)}; с парой {int(has_a.sum())}, из них с одно-камерным дубликатом "
          f"{n_dup} ({n_dup / max(int(has_b.sum()), 1) * 100:.1f}%)")
    print(f"медиана уверенности у запросов с парой: A {np.median(raw.max(axis=1)[has_a]):.3f}, "
          f"B {np.median(raw_b.max(axis=1)[has_b]):.3f}")

    thr_now = float(json.loads((Path(a.release) / "recipe.json").read_text(encoding="utf-8"))["threshold"])
    grid = list(np.round(np.arange(0.40, 0.72, 0.01), 3)) + [round(thr_now, 6)]
    rows = []
    for t in sorted(set(grid)):
        A = block_score(raw, has_a, gv, qv, t); B = block_score(raw_b, has_b, gv, qv, t)
        rows.append({"threshold": float(t), "A": A, "B": B, "worst": min(A["score"], B["score"]),
                     **{f"E{p}": p * A["score"] + (1 - p) * B["score"] for p in a.beliefs}})
    hdr = f"{'порог':>8}{'A: как сейчас':>15}{'B: без дублей':>15}{'худший':>9}"
    hdr += "".join(f"{'E p=' + str(p):>11}" for p in a.beliefs)
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        if abs(r["threshold"] * 100 - round(r["threshold"] * 100)) > 1e-6 or int(round(r["threshold"] * 100)) % 2 == 0:
            line = f"{r['threshold']:8.4f}{r['A']['score']:15.2f}{r['B']['score']:15.2f}{r['worst']:9.2f}"
            print(line + "".join(f"{r['E' + str(p)]:11.2f}" for p in a.beliefs))
    best_mm = max(rows, key=lambda r: r["worst"])
    print("-" * len(hdr))
    print(f"минимакс: порог {best_mm['threshold']:.4f}, худший балл {best_mm['worst']:.2f}")
    for p in a.beliefs:
        b = max(rows, key=lambda r: r[f"E{p}"])
        print(f"максимум ожидания при вере p={p:.2f}: порог {b['threshold']:.4f}, ожидание {b[f'E{p}']:.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"n_query": int(len(qv)), "threshold_in_release": thr_now,
                                       "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[stress] → {a.out}")


if __name__ == "__main__":
    main()
