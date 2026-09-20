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

    cutoff=K — метрика организаторов mAP@K (ответы Q-10/Q-13 от 16.09): кадры той же машины
    с камеры запроса удаляются ДО обрезания до K, AP считается по первым K и нормируется на
    min(n_совпадений, K), усреднение макро по запросам; запросы без валидных совпадений
    после фильтрации из расчёта исключаются (а не получают AP=0)."""
    Q, G = sims.shape
    max_rank = max(ranks)
    order = np.argsort(-sims, axis=1, kind='stable')
    cmc_hits = np.zeros(max_rank, dtype=np.float64)
    aps = []
    n_valid = 0
    for i in range(Q):
        idx = order[i]
        if cross_camera_only:
            keep = g_cams[idx] != q_cams[i]
        else:
            keep = ~((g_vids[idx] == q_vids[i]) & (g_cams[idx] == q_cams[i]))
        idx = idx[keep]
        matches = (g_vids[idx] == q_vids[i]).astype(np.int32)
        n_match = matches.sum()
        if n_match == 0:
            continue
        n_valid += 1
        first = np.argmax(matches)
        if first < max_rank:
            cmc_hits[first:] += 1
        cum = np.cumsum(matches)
        hit_pos = np.nonzero(matches)[0]
        precision = cum[hit_pos] / (hit_pos + 1)
        if cutoff is None:
            aps.append(precision.mean())
        else:
            inside = hit_pos < cutoff
            aps.append(float(precision[inside].sum() / min(int(n_match), cutoff)))
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
