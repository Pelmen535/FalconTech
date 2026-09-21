# -*- coding: utf-8 -*-
"""Предполётная проверка трёх файлов сдачи. Запускать ПЕРЕД отправкой, каждый раз.

    python scripts/check_submission.py --submission submission_soup --data Данные

Официального evaluate.py у нас нет, поэтому это единственное, что стоит между нами и
«файл не распарсился». Проверяется только форма, не качество: правила формата взяты из
письменных ответов организаторов 16.09 (номера ответов указаны у каждой проверки).

Код возврата: 0 — всё чисто, 1 — есть ошибки, 2 — только предупреждения.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ERR, WARN = [], []
def err(msg): ERR.append(msg)
def warn(msg): WARN.append(msg)


def read_ids(p: Path) -> list[str]:
    with open(p, newline="", encoding="utf-8") as f:
        return [str(r["image_id"]) for r in csv.DictReader(f)]


def main():
    ERR.clear(); WARN.clear()
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()
    sub, data = Path(a.submission), Path(a.data)

    qids = read_ids(data / "test_query.csv")
    gids = read_ids(data / "test_gallery.csv")
    qset, gset = set(qids), set(gids)
    print(f"[check] test_query {len(qids)} строк, test_gallery {len(gids)}")
    if len(qset) != len(qids):
        err(f"в test_query.csv повторяются image_id ({len(qids) - len(qset)} шт.)")
    if len(gset) != len(gids):
        err(f"в test_gallery.csv повторяются image_id ({len(gids) - len(gset)} шт.)")

    # --- embeddings.npy: строго query в порядке файла, затем gallery (ответ 27) ---
    ep = sub / "embeddings.npy"
    if not ep.exists():
        err(f"нет {ep}")
    else:
        E = np.load(ep)
        print(f"[check] embeddings.npy {E.shape} {E.dtype}")
        if E.ndim != 2:
            err(f"embeddings.npy должен быть (N, D), а он {E.shape}")
        elif E.shape[0] != len(qids) + len(gids):
            err(f"строк {E.shape[0]}, а должно быть len(query)+len(gallery) = {len(qids) + len(gids)} (ответ 27)")
        if E.dtype != np.float32:
            warn(f"тип {E.dtype}; организаторы ожидают float32 (ответ 27)")
        if not np.isfinite(E).all():
            err("в embeddings.npy есть NaN или inf")
        else:
            n = np.linalg.norm(E, axis=1)
            print(f"[check] норма строк: мин {n.min():.4f}, макс {n.max():.4f}")
            if abs(n.mean() - 1.0) > 0.01:
                warn("строки не L2-нормированы; организатор нормирует сам, но лучше сдавать готовое")
            if (n == 0).any():
                err(f"{int((n == 0).sum())} нулевых векторов — для каждого image_id строка обязана быть осмысленной (ответ 49)")

    # --- submission.csv: РОВНО topk id на каждый запрос, включая отказавшие (ответы 10/12/21/22) ---
    sp = sub / "submission.csv"
    if not sp.exists():
        err(f"нет {sp}")
    else:
        with open(sp, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
        # Официальный формат — БЕЗ заголовка (organizer/evaluate.py: «Формат
        # submission.csv (без заголовка)»). Строку заголовка из файлов прежних версий
        # пропускаем и говорим об этом вслух: сдавать её не надо.
        if rows and rows[0][0] == "query_id":
            warn("в submission.csv есть строка заголовка; официальный формат без неё "
                 "(organizer/evaluate.py разберёт её как лишний запрос и предупредит)")
            rows = rows[1:]
        body = rows
        print(f"[check] submission.csv: строк {len(body)}, ячеек в строке {len(body[0]) if body else 0}")
        if body and len(body[0]) != a.topk + 1:
            err(f"ячеек в строке {len(body[0])}, ожидается 1 + {a.topk}")
        if len(body) != len(qids):
            err(f"строк {len(body)}, а запросов {len(qids)} — строка нужна на КАЖДЫЙ запрос")
        seen = []
        for i, r in enumerate(body):
            if len(r) != a.topk + 1:
                err(f"строка {i + 1}: ячеек {len(r)}, ожидается {a.topk + 1}"); break
            seen.append(r[0])
            empty = [j for j, x in enumerate(r[1:], 1) if x.strip() == ""]
            if empty:
                err(f"строка {i + 1} ({r[0]}): пустые ячейки {empty} — нужно ровно {a.topk} id (ответы 10/12/21/22)"); break
            bad = [x for x in r[1:] if x not in gset]
            if bad:
                err(f"строка {i + 2} ({r[0]}): id не из галереи: {bad[:3]}"); break
            if len(set(r[1:])) != a.topk:
                warn(f"строка {i + 2} ({r[0]}): в топ-{a.topk} есть повторы")
        if seen and seen != qids:
            if set(seen) == qset:
                warn("порядок строк submission.csv не совпадает с test_query.csv (по составу всё на месте)")
            else:
                err("набор query_id в submission.csv не совпадает с test_query.csv")

    # --- candidates.csv: заголовок фиксирован, отказ = отсутствие строк (ответы 18/20/29) ---
    cp = sub / "candidates.csv"
    if not cp.exists():
        err(f"нет {cp}")
    else:
        with open(cp, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        head, body = rows[0], rows[1:]
        if head != ["query_id", "gallery_id", "confidence"]:
            err(f"заголовок candidates.csv = {head}, требуется ['query_id','gallery_id','confidence'] (ответы 18/29)")
        answered, per_q = set(), {}
        for i, r in enumerate(body):
            if len(r) != 3:
                err(f"candidates.csv строка {i + 2}: {len(r)} полей вместо 3"); break
            q, g, c = r
            if q not in qset:
                err(f"candidates.csv строка {i + 2}: query_id {q!r} не из test_query.csv"); break
            if g not in gset:
                err(f"candidates.csv строка {i + 2}: gallery_id {g!r} не из test_gallery.csv"); break
            if c.strip() == "":
                err(f"candidates.csv строка {i + 2}: пустая confidence — отказ кодируется ОТСУТСТВИЕМ строк (ответ 18)"); break
            try:
                cv = float(c)
            except ValueError:
                err(f"candidates.csv строка {i + 2}: confidence {c!r} не число"); break
            # float('nan') разбирается без ошибки, поэтому одного try мало: строка с nan
            # проходила проверку и выглядела как уверенный ответ.
            if not np.isfinite(cv):
                err(f"candidates.csv строка {i + 2}: confidence {c!r} не конечна (NaN/inf)"); break
            answered.add(q); per_q[q] = per_q.get(q, 0) + 1
        refused = len(qset) - len(answered)
        dup = sum(1 for v in per_q.values() if v > 1)
        print(f"[check] candidates.csv: строк {len(body)}, отвечено {len(answered)}, "
              f"отказов {refused} ({refused / max(len(qset), 1) * 100:.1f}%), запросов с >1 строкой: {dup}")
        if dup:
            warn(f"у {dup} запросов больше одной строки — в метрику идёт только строка с максимальной confidence (ответы 19/23/25)")
        if refused == 0:
            warn("отказов нет вовсе: на закрытом тесте ~20% запросов без пары (ответ 17), ноль отказов там даст FP")
        if refused == len(qset):
            # Не ошибка формата: на наборе, где ни у одного запроса нет пары, отказ по всем —
            # правильный ответ модели. Диагностируем, но не объявляем нарушением.
            warn("отказ по ВСЕМ запросам — проверь порог; на наборе без пар это может быть верно")

    for m in WARN:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] {m}")
    for m in ERR:
        print(f"[ОШИБКА] {m}")
    if ERR:
        print(f"\n[check] ОШИБОК {len(ERR)} — так отправлять нельзя")
        return 1
    if WARN:
        print(f"\n[check] ошибок нет, предупреждений {len(WARN)}")
        return 2
    print("\n[check] все три файла по форме в порядке")
    return 0


if __name__ == "__main__":
    sys.exit(main())
