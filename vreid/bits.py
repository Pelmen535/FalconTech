"""Информационная ёмкость наблюдения в битах.

Идея: вместо "похожесть 0.87" — сколько бит неопределённости снимает наблюдение
относительно парка (галереи):

    bits = -log2(N_кандидатов_в_радиусе / N_галереи)

Радиус (порог сходства tau) НЕ задаётся руками, а калибруется на валидации:
берём такой tau, чтобы истинное совпадение (та же машина, другая камера) попадало
в радиус у recall% запросов. Тогда "кандидаты в радиусе" — это множество, в котором
искомая машина лежит с вероятностью ≈ recall, а bits честно отражает, насколько оно узкое.

Несколько наблюдений одной машины (трасса по камерам) НЕ складываются по битам:
берётся ПЕРЕСЕЧЕНИЕ множеств кандидатов, и биты считаются от его размера.
Пять одинаковых ракурсов дадут почти то же, что один; разные ракурсы — больше.
"""
from __future__ import annotations

import numpy as np

from .index import GalleryIndex


def calibrate_tau(sims: np.ndarray, q_vids: np.ndarray, q_cams: np.ndarray,
                  g_vids: np.ndarray, g_cams: np.ndarray, recall: float = 0.95) -> dict:
    """tau — такой порог сходства, что у recall% запросов лучшее истинное
    (кросс-камерное) совпадение имеет sim ≥ tau."""
    best = []
    for i in range(sims.shape[0]):
        mask = (g_vids == q_vids[i]) & (g_cams != q_cams[i])
        if mask.any():
            best.append(sims[i, mask].max())
    best = np.asarray(best)
    if len(best) == 0:
        raise RuntimeError("нет кросс-камерных пар для калибровки радиуса")
    tau = float(np.quantile(best, 1.0 - recall))
    return {"tau": tau, "recall": recall, "n_pairs": int(len(best)),
            "best_match_sim_median": float(np.median(best))}


def bits_from_count(n_cand: np.ndarray | int, n_gallery: int) -> np.ndarray:
    n = np.maximum(np.asarray(n_cand, dtype=np.float64), 1.0)
    return -np.log2(n / float(n_gallery))


def bits_single(index: GalleryIndex, q: np.ndarray, tau: float) -> dict:
    """Биты для одного/нескольких независимых запросов. → dict(bits [Q], n_cand [Q], n_gallery)."""
    n_cand = index.count_within(q, tau)
    return {"bits": bits_from_count(n_cand, index.n), "n_cand": n_cand, "n_gallery": index.n,
            "max_bits": float(np.log2(index.n))}


def bits_joint(index: GalleryIndex, observations: np.ndarray, tau: float) -> dict:
    """Совместные биты для нескольких наблюдений ОДНОЙ машины (трасса).
    observations: [M, D]. Кандидаты = пересечение радиусов всех наблюдений."""
    sets = index.ids_within(observations, tau)
    cand = None
    per_obs = []
    for s in sets:
        s = set(s.tolist())
        per_obs.append(len(s))
        cand = s if cand is None else (cand & s)
    n = len(cand) if cand is not None else index.n
    return {"bits": float(bits_from_count(n, index.n)), "n_cand": n, "n_gallery": index.n,
            "per_observation_n_cand": per_obs, "candidates": np.array(sorted(cand)) if cand else np.array([], dtype=np.int64),
            "max_bits": float(np.log2(index.n))}


def bits_calibration_table(sims: np.ndarray, q_vids: np.ndarray, q_cams: np.ndarray,
                           g_vids: np.ndarray, g_cams: np.ndarray, tau: float,
                           n_bins: int = 6, cross_camera_only: bool = True) -> list[dict]:
    """Для защиты: разбиваем запросы по набранным битам и смотрим точность rank-1 в каждой корзине.
    Ожидаем монотонность: больше бит → выше точность. Если нет — радиус/эмбеддинг плохие."""
    Q = sims.shape[0]
    n_g = sims.shape[1]
    bits = np.zeros(Q)
    correct = np.zeros(Q, dtype=bool)
    valid = np.zeros(Q, dtype=bool)
    for i in range(Q):
        if cross_camera_only:
            keep = g_cams != q_cams[i]
        else:
            keep = ~((g_vids == q_vids[i]) & (g_cams == q_cams[i]))
        row = sims[i, keep]
        gv = g_vids[keep]
        if not (gv == q_vids[i]).any():
            continue
        valid[i] = True
        n_cand = int((row >= tau).sum())
        bits[i] = bits_from_count(n_cand, n_g)
        correct[i] = gv[np.argmax(row)] == q_vids[i]
    bits, correct = bits[valid], correct[valid]
    edges = np.quantile(bits, np.linspace(0, 1, n_bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (bits >= lo) & (bits <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bits_lo": float(lo), "bits_hi": float(hi), "n": int(m.sum()),
                     "rank1_acc": float(correct[m].mean())})
    return rows
