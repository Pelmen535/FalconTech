"""Каскад «лёгкая модель ищет — тяжёлая проверяет шортлист».

Зачем. Основная модель ViT-B укладывается в бюджет производительности (33 мс на ТС,
149 FPS), но по точности отстаёт от ViT-L на несколько пунктов. Сдавать ViT-L основной
моделью нельзя: форвард втрое дороже, и это минус около 12 баллов из 20 за
производительность против плюс 2 за точность. Каскад берёт от обеих: шортлист строит
лёгкая модель, а порядок внутри шортлиста уточняет тяжёлая.

Почему это разрешено. Правила разводят инференс и ре-ранжирование:
  * «latency_b1 включает чтение и декодирование, crop по BBox, preprocessing, forward,
    postprocessing и L2-нормализацию. Поиск по gallery и rerank не входят» (ответы 31, 32);
  * лимит 2 ГБ считается по сумме весов «основного backbone, auxiliary-моделей,
    классификаторов, ensemble и reranker» (ответы 34, 37) — отдельная модель-ре-ранкер
    правилами предусмотрена прямо, и её вес мы в лимит закладываем;
  * «submission.csv может быть результатом per-query каскада: cosine shortlist,
    pairwise/local-feature проверка или per-query k-reciprocal» (ответ 28);
  * embeddings.npy у организаторов идёт в справочный раздел, балл считается по
    submission.csv — расхождение порядка с косинусом по embeddings.npy штатно.

Чего мы НЕ делаем и о чём говорим вслух: настоящая стоимость одного запроса при каскаде
выше замеряемой. Числа обеих — в README и docs/SELF_REVIEW.md, прятать разницу нечестно.

Каждый запрос обрабатывается независимо: шортлист берётся из его собственной строки,
пересчёт использует только его кандидатов. Проверяется scripts/check_query_independence.py.
"""
from __future__ import annotations

import numpy as np


def shortlist(rank: np.ndarray, topk: int) -> np.ndarray:
    """Индексы топ-K галереи для каждого запроса; -1 там, где кандидат закрыт маской.

    Сортировка стабильная: на ничьей порядок задаёт порядок строк test_gallery.csv
    (ответ 16), а не внутреннее состояние argsort."""
    order = np.argsort(-rank, axis=1, kind="stable")[:, :topk]
    rows = np.arange(rank.shape[0])[:, None]
    return np.where(np.isfinite(rank[rows, order]), order, -1)


def rescore(rank: np.ndarray, order: np.ndarray, heavy: np.ndarray, alpha: float) -> np.ndarray:
    """Новый порядок: внутри шортлиста место определяет alpha·heavy + (1−alpha)·rank.

    У k-reciprocal (−расстояние) и косинуса разные масштабы, поэтому обе величины
    приводятся к [0,1] минимаксом ПО СТРОКЕ: сравниваются кандидаты одного запроса, а не
    запросы между собой — иначе решение по запросу зависело бы от остальных (ответ 38).

    Хвост ниже шортлиста сохраняет порядок лёгкой модели: шортлисту выдаётся полоса
    значений выше K-го значения базы, а оно не ниже любого хвостового по построению
    шортлиста. alpha=0 обязан воспроизводить базу с точностью до порядка — на этом
    держится проверка самой функции (tests/test_cascade.py).

    rank меняется не на месте: вызывающая сторона вправе сравнить до и после."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha должна быть в [0, 1], получено {alpha}")
    rows = np.arange(rank.shape[0])[:, None]
    picked = rank[rows, order]
    finite = np.isfinite(picked) & (order >= 0)

    def to_unit(values: np.ndarray) -> np.ndarray:
        masked = np.where(finite, values, np.nan)
        low = np.nanmin(masked, axis=1, keepdims=True)
        high = np.nanmax(masked, axis=1, keepdims=True)
        return np.nan_to_num((masked - low) / np.clip(high - low, 1e-9, None))

    mixed = alpha * to_unit(heavy) + (1 - alpha) * to_unit(picked)
    kth = np.where(finite, picked, np.inf).min(axis=1, keepdims=True)
    out = rank.copy()
    out[rows.repeat(order.shape[1], 1)[finite], order[finite]] = (kth + 1.0 + mixed)[finite]
    return out


def heavy_scores(q_emb: np.ndarray, g_emb: np.ndarray, order: np.ndarray,
                 blocked: np.ndarray | None = None) -> np.ndarray:
    """Косинусы тяжёлой модели для кандидатов шортлиста: [Q, K].

    Считается только по шортлисту, а не по всей галерее: так и работает каскад, и так
    видно, что в решение по запросу входят только его собственные кандидаты."""
    q_unit = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True).clip(1e-12)
    g_unit = g_emb / np.linalg.norm(g_emb, axis=1, keepdims=True).clip(1e-12)
    out = np.full(order.shape, -np.inf, dtype=np.float32)
    for i in range(order.shape[0]):
        picked = order[i]
        valid = picked >= 0
        if blocked is not None:
            valid = valid & ~blocked[i, np.where(picked >= 0, picked, 0)]
        if valid.any():
            out[i, valid] = g_unit[picked[valid]] @ q_unit[i]
    return out
