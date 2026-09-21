"""Стандартный протокол re-ID: CMC (rank-k) и mAP.

Правила (как в Market-1501 / VeRi-776):
  * для каждого запроса галерея ранжируется по сходству;
  * из галереи выкидываются кадры того же id с той же камеры (это не "переидентификация");
  * запросы без ни одного валидного совпадения в галерее не учитываются;
  * AP считается по всем валидным совпадениям, CMC — по позиции первого.
ВНИМАНИЕ. cross_camera_only=True выкидывает из галереи ВСЕ кадры с камеры запроса, включая
ЧУЖИЕ машины. Это НЕ правило жюри: организаторы (ответы 5 и 38) удаляют только пары, у которых
совпали И vehicle_id, И camera_id. Строгий режим выбрасывает самые трудные негативы — чужие
машины с той же точки, в том же свете и ракурсе — и потому завышает mAP на 1-3 пункта.
Он оставлен как диагностика «та же машина на другой точке», но метрика жюри — это
cross_camera_only=False с cutoff=10.
"""
from __future__ import annotations

import numpy as np


def evaluate(sims: np.ndarray, q_vids: np.ndarray, q_cams: np.ndarray,
             g_vids: np.ndarray, g_cams: np.ndarray, ranks=(1, 5, 10),
             cross_camera_only: bool = False, cutoff: int | None = None) -> dict:
    """sims: [Q, G] сходства (больше = ближе). Возвращает dict с mAP, rank-k, n_valid.

    cutoff=K — метрика организаторов mAP@K. Считается ровно так, как это делает эталонный
    organizer/evaluate.py, а это НЕ то же самое, что «убрать junk и взять первые K»:

      1. берутся первые K кандидатов ИСХОДНОГО ранжирования — именно они уходят в
         submission.csv, и другого списка у жюри нет;
      2. из этих K удаляются junk-пары (тот же vehicle_id И та же camera_id);
      3. AP считается по тому, что осталось (может быть меньше K строк), и нормируется на
         min(n_валидных_позитивов_во_всей_галерее, K).

    Разница с прежней реализацией не косметическая: раньше junk удалялся из ПОЛНОГО
    ранжирования, и на освободившееся место поднимался кандидат с 11-й позиции. У жюри он
    не поднимется — место просто пропадает. На нашей валидации это 74.33 против 74.21,
    то есть мы завышали на 0.12 п.п. Сверено построчно: scripts/score_validation_with_official.py
    даёт побайтово те же 0.742072, что и эталонный скрипт.

    Запросы без валидных совпадений после junk-фильтра из расчёта исключаются (а не
    получают AP=0) — это ответы 11/13/22 и так же сделано в эталоне."""
    Q, G = sims.shape
    max_rank = max(ranks)
    order = np.argsort(-sims, axis=1, kind='stable')
    cmc_hits = np.zeros(max_rank, dtype=np.float64)
    aps = []
    n_valid = 0
    def junk_mask(candidates, i):
        """Пары, которые жюри вычёркивает из ранжирования (ответ 11)."""
        if cross_camera_only:
            return g_cams[candidates] == q_cams[i]
        return (g_vids[candidates] == q_vids[i]) & (g_cams[candidates] == q_cams[i])

    for i in range(Q):
        # Число валидных позитивов считается по ВСЕЙ галерее: именно им нормируется AP
        # и по нему решается, участвует ли запрос в метрике.
        full = order[i]
        kept_full = full[~junk_mask(full, i)]
        n_match = int((g_vids[kept_full] == q_vids[i]).sum())
        if n_match == 0:
            continue
        n_valid += 1

        if cutoff is None:
            matches = (g_vids[kept_full] == q_vids[i]).astype(np.int32)
            first = int(np.argmax(matches))
            if first < max_rank:
                cmc_hits[first:] += 1
            cum = np.cumsum(matches)
            hit_pos = np.nonzero(matches)[0]
            aps.append(float((cum[hit_pos] / (hit_pos + 1)).mean()))
            continue

        # Режим организаторов: сначала обрезаем до K (это и есть submission.csv),
        # потом вычёркиваем junk. Освободившееся место НЕ занимает кандидат с K+1.
        submitted = full[:cutoff]
        clean = submitted[~junk_mask(submitted, i)]
        relevant = (g_vids[clean] == q_vids[i]).astype(np.int32)
        if relevant.any():
            first = int(np.argmax(relevant))
            if first < max_rank:
                cmc_hits[first:] += 1
            cum = np.cumsum(relevant)
            precision = cum / (np.arange(len(relevant)) + 1)
            aps.append(float((precision * relevant).sum() / min(n_match, cutoff)))
        else:
            aps.append(0.0)
    if n_valid == 0:
        raise RuntimeError("ни один запрос не имеет валидного совпадения в галерее — проверь vid/cam")
    out = {f"rank{r}": float(cmc_hits[r - 1] / n_valid) for r in ranks}
    out["mAP"] = float(np.mean(aps))
    out["cutoff"] = cutoff
    out["n_query_valid"] = int(n_valid)
    out["n_query_total"] = int(Q)
    out["n_gallery"] = int(G)
    return out


def format_metrics(m: dict, title: str = "") -> str:
    parts = [f"{('mAP@' + str(m['cutoff'])) if k == 'mAP' and m.get('cutoff') else k}={v * 100:5.1f}%"
             for k, v in m.items() if k.startswith("rank") or k == "mAP"]
    head = f"{title}: " if title else ""
    return head + "  ".join(parts) + f"  (queries {m['n_query_valid']}/{m['n_query_total']}, gallery {m['n_gallery']})"
