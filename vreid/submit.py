"""Артефакты сдачи (ТЗ §8):

  embeddings.npy   float32 [N_test, D] строго в порядке test.csv
  submission.csv   query_id, gallery_id_1 ... gallery_id_10 без заголовка (топ-10 по убыванию)
  candidates.csv   query_id, gallery_id, confidence — только принятые; ОТКАЗ = полное отсутствие
                   строк для этого query_id (ответ организаторов Q-18/Q-20/Q-29, 16.09)
Формат по README организаторов: embeddings.npy — сначала все query (по порядку файла), затем gallery.

Точные имена столбцов и правила «кто запрос, кто галерея» в test.csv уточним по README
организаторов — здесь параметры: id_col, query_mask.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def save_embeddings(emb: np.ndarray, path: str | Path) -> None:
    """float32 [N, D] в порядке «сначала все query, затем вся gallery» (ответ 27).

    Нулевая или нечисловая строка — это не «плохой признак», а сбой: косинус к ней не
    определён, а строка обязана быть для каждого image_id (ответ 49). Лучше упасть здесь,
    чем отдать жюри матрицу, по которой ничего нельзя посчитать.
    """
    emb = np.ascontiguousarray(emb.astype(np.float32))
    if emb.ndim != 2 or not emb.shape[1]:
        raise ValueError('Embedding matrix must be two-dimensional with at least one column')
    if not np.isfinite(emb).all():
        raise ValueError('Embedding matrix must be finite')
    if np.any(np.linalg.norm(emb, axis=1) <= 1e-8):
        raise ValueError('Embedding matrix must not contain zero rows')
    np.save(path, emb)
    print(f"[submit] embeddings.npy {emb.shape} → {path}")


def write_submission(q_keys, g_keys, sims: np.ndarray, path: str | Path, topk: int = 10,
                     exclude_self: bool = True) -> None:
    """sims [Q, G]. Если запросы и галерея — одно и то же множество (test.csv целиком),
    exclude_self убирает совпадение записи с самой собой."""
    # -inf допустим: так помечаются запрещённые пары. NaN и +inf — нет: по ним нельзя
    # построить порядок, и стабильная сортировка молча вернула бы произвольный.
    if sims.shape != (len(q_keys), len(g_keys)):
        raise ValueError('Ranking matrix shape must be (queries, gallery)')
    if np.isnan(sims).any() or np.isposinf(sims).any():
        raise ValueError('Ranking matrix must not contain NaN or +inf')
    if len(set(q_keys)) != len(q_keys) or len(set(g_keys)) != len(g_keys):
        raise ValueError('Unique query and gallery IDs required')
    if topk < 1 or len(g_keys) < topk:
        raise ValueError('Not enough distinct gallery candidates; padding duplicates is invalid')
    # Стабильная сортировка: на точных ничьих порядок задаётся порядком строк
    # test_gallery.csv, а не внутренним состоянием argsort. Иначе два одинаковых прогона
    # могли бы выдать разные файлы.
    order = np.argsort(-sims, axis=1, kind="stable")
    rows = []
    for i, query_key in enumerate(q_keys):
        selected = [g_keys[j] for j in order[i]
                    if np.isfinite(sims[i, j]) and not (exclude_self and g_keys[j] == query_key)]
        selected = selected[:topk]
        if len(selected) != topk:
            # Ровно десять идентификаторов нужны всегда (ответы 10/12/21/22), но добивать
            # список повторами нельзя: это был бы ответ, которого модель не давала.
            raise ValueError(f'Not enough eligible distinct candidates for {query_key}')
        rows.append([query_key] + selected)
    # БЕЗ строки заголовка: так задан официальный формат (organizer/evaluate.py, шапка:
    # «Формат submission.csv (без заголовка)») и так устроен organizer/example_submission.zip.
    # Заголовок эталонный scorer не отвергает, но разбирает как ещё один запрос с десятью
    # неизвестными идентификаторами и печатает предупреждение — сдавать такое незачем.
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[submit] submission.csv ({len(q_keys)} запросов, top-{topk}) → {path}")


def write_candidates(q_keys, g_keys, sims: np.ndarray, conf: np.ndarray, threshold: float,
                     path: str | Path, topk: int = 1, exclude_self: bool = True,
                     per_candidate_min_sim: float | None = None,
                     conf_sims: np.ndarray | None = None) -> dict:
    """Принятые кандидаты: если уверенность запроса ≥ threshold — топ-k (и, опционально,
    только те, чьё сходство ≥ per_candidate_min_sim); иначе строк для запроса нет вовсе.

    В метрику у организаторов идёт ТОЛЬКО кандидат с максимальной confidence, поэтому по умолчанию
    topk=1: лишние строки ни на что не влияют, а однозначность дороже.

    sims задаёт ПОРЯДОК кандидатов (после ре-ранжирования это −расстояние k-reciprocal),
    а в столбец confidence пишется conf_sims — косинусное сходство в [-1, 1], которое
    человек и жюри могут прочитать. Без conf_sims столбец берётся из sims.
    per_candidate_min_sim отсекает кандидатов по conf_sims (по косинусу, не по расстоянию)."""
    if conf_sims is None:
        conf_sims = sims
    # Стабильная сортировка: при равных оценках порядок определяется порядком строк в
    # test_gallery.csv, а не внутренним состоянием argsort. Иначе два одинаковых прогона
    # могут выдать разных кандидатов на ничьей.
    order = np.argsort(-sims, axis=1, kind="stable")
    n_ref = n_bad = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qk in enumerate(q_keys):
            # Решение об отказе принимается по косинусу ТОГО кандидата, который реально уйдёт
            # в файл, а не по сырому топ-1. У нашей конфигурации они совпадают (измерено:
            # ре-ранжирование меняет топ-1 у 0 из 1213 запросов), но привязка к выдаче
            # закрывает целый класс претензий: «порог считался про другого кандидата».
            sel = None
            for j in order[i]:
                if exclude_self and g_keys[j] == qk:
                    continue
                if per_candidate_min_sim is not None and conf_sims[i, j] < per_candidate_min_sim:
                    continue
                sel = j
                break
            decide = conf[i] if sel is None else float(conf_sims[i, sel])
            # NaN < порог — ЛОЖЬ по IEEE, то есть без явной проверки численный сбой превращался
            # в УВЕРЕННЫЙ ответ с confidence=nan. Нечисло — это отсутствие оснований отвечать,
            # значит отказ.
            if not np.isfinite(decide) or decide < threshold:
                if not np.isfinite(decide):
                    n_bad += 1
                n_ref += 1          # отказ: не пишем ни одной строки для этого query_id
                continue
            written = 0
            for j in order[i]:
                if exclude_self and g_keys[j] == qk:
                    continue
                if per_candidate_min_sim is not None and conf_sims[i, j] < per_candidate_min_sim:
                    continue
                c = float(conf_sims[i, j])
                if not np.isfinite(c):
                    n_bad += 1
                    continue        # не пишем строку с nan/inf в confidence
                w.writerow([qk, g_keys[j], f"{c:.5f}"]); written += 1
                if written >= topk:
                    break
            if written == 0:
                n_ref += 1
    print(f"[submit] candidates.csv: отказов {n_ref}/{len(q_keys)} → {path}")
    if n_bad:
        print(f"[submit] ВНИМАНИЕ: {n_bad} нечисловых (NaN/inf) значений уверенности — эти запросы "
              f"ушли в отказ. Такого быть не должно: разберись с причиной ДО отправки.")
    return {"n_refused": n_ref, "n_query": len(q_keys), "threshold": threshold,
            "n_nonfinite": n_bad}
