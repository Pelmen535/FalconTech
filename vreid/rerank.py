"""Постобработка ранжирования.

  k_reciprocal(q, g, k1, k2, lam)  — Zhong et al., CVPR 2017. Возвращает матрицу расстояний [Q, G]
                                      (меньше = ближе), учитывающую взаимных соседей.
  dba(emb, k, alpha)               — database-side augmentation: каждый вектор заменяется
                                      взвешенным средним с k ближайшими ВЗАИМНЫМИ соседями.
                                      Результат — снова векторы: можно писать в embeddings.npy.
  aqe(q, g, k, alpha)              — alpha query expansion для запросов.

Конкурсный CLI использует только k-reciprocal с независимыми запросами.
DBA/AQE и shared-режим сохранены для исследований; конкурсный recipe их не включает.
"""
from __future__ import annotations

import numpy as np


def _l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


# --------------------------------------------------------------------------- k-reciprocal
# Значение для «соседом быть нельзя»: строго больше максимальной возможной дистанции 4.0.
_FAR = 8.0


def k_reciprocal(q_emb: np.ndarray, g_emb: np.ndarray, k1: int = 20, k2: int = 6, lam: float = 0.3,
                 same_mask: np.ndarray | None = None, shared: bool = False,
                 isolate_queries: bool = True) -> np.ndarray:
    """Расстояния после ре-ранжирования, [Q, G]. same_mask (bool) — пары, которые нельзя считать
    соседями (та же запись / тот же кадр); по умолчанию только диагональ.
    shared=True: q_emb и g_emb — одно и то же множество (тест целиком), same_mask тогда [N, N],
    иначе same_mask [Q+G, Q+G].

    isolate_queries=True: запросы не участвуют в окрестностях ВООБЩЕ — ни своих, ни галерейных.
    Окрестность любого элемента строится только по галерее. Организаторы (ответ 38) разрешили k-reciprocal «в топ-K одного
    query», но запретили кластеризацию по всей test_query; без этой изоляции ранжирование одного
    запроса зависит от остальных запросов теста, то есть это как раз кластеризация."""
    if shared:
        feats = _l2n(q_emb.astype(np.float32))
        N, Q = len(feats), len(feats)
    else:
        feats = _l2n(np.concatenate([q_emb, g_emb]).astype(np.float32))
        N, Q = len(feats), len(q_emb)
    # Ограничиваем k1 РАЗМЕРОМ ГАЛЕРЕИ, а не N = Q+G. С N это была скрытая зависимость
    # от числа остальных запросов: при изоляции соседей берём только из галереи, и если бы k1
    # упирался в потолок, потолок сдвигался вместе с Q.
    bound = (N if shared else len(g_emb)) - 1
    k1 = max(1, min(k1, bound))
    k2 = max(1, min(k2, k1))
    if isolate_queries and not shared:
        # Fixed BLAS operand shapes: a full (Q+G) GEMM can round ties differently
        # when Q changes, even with stable argsort. Query-query distances are unused.
        gallery=feats[Q:]
        orig=np.full((N,N),_FAR,dtype=np.float32)
        orig[Q:,Q:]=2.0-2.0*(gallery @ gallery.T)
        for i in range(Q):orig[i,Q:]=2.0-2.0*(gallery @ feats[i])
    else:
        orig = 2.0 - 2.0 * (feats @ feats.T)
    # Запрещённые пары ставим СТРОГО дальше любой настоящей дистанции (4.0 достижимо
    # для противоположных векторов, и тогда маска бы с ними сравнялась в ничьей).
    if same_mask is not None:
        orig[same_mask] = _FAR
    if isolate_queries and not shared:
        # Мало запретить запросам быть соседями ДРУГ ДРУГА: k-reciprocal строит окрестности и для
        # галерейных элементов, а они могут брать в соседи запросы — и тогда нормировка
        # галерейных строк, а с ней и итоговое расстояние, зависит от состава test_query.
        # Проверка независимости это и поймала: топ-10 совпадал лишь у 156 запросов из 666.
        # Поэтому запрос не может быть соседом НИКОМУ: обнуляем весь столбец запросов.
        orig[:, :Q] = _FAR
    np.fill_diagonal(orig, 0.0)
    # СТАБИЛЬНАЯ сортировка. Без неё на РАВНЫХ расстояниях (два одинаковых кадра
    # в галерее, дубликат файла) порядок соседей решает introsort, а он зависит от раскладки
    # строки в памяти — а она сдвигается, когда рядом в матрице появляются другие запросы.
    # То есть без stable выдача одного запроса могла зависеть от остальных. Со стабильной
    # ничья решается номером столбца, а порядок галерейных столбцов между собой от Q не зависит.
    rank = np.argsort(orig, axis=1, kind="stable")     # [N, N], первый — сам элемент

    def k_recip(i, k):
        fwd = rank[i, : k + 1]
        back = rank[fwd, : k + 1]
        return fwd[(back == i).any(axis=1)]

    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        R = k_recip(i, k1)
        expanded = set(R.tolist())
        for j in R:
            Rj = k_recip(j, max(1, k1 // 2))
            if len(np.intersect1d(R, Rj, assume_unique=True)) > 2 / 3 * len(Rj):
                expanded.update(Rj.tolist())
        idx = np.array(sorted(expanded), dtype=np.int64)
        w = np.exp(-orig[i, idx])
        V[i, idx] = w / w.sum()
    if k2 > 1:
        V = np.stack([V[rank[i, :k2]].mean(axis=0) for i in range(N)])
    inv = [np.nonzero(V[:, j])[0] for j in range(N)]
    jaccard = np.zeros((Q, N), dtype=np.float32)
    for i in range(Q):
        tmin = np.zeros(N, dtype=np.float32)
        nz = np.nonzero(V[i])[0]
        for j in nz:
            ids = inv[j]
            tmin[ids] += np.minimum(V[i, j], V[ids, j])
        jaccard[i] = 1.0 - tmin / (2.0 - tmin)
    final = jaccard * (1 - lam) + orig[:Q] * lam
    return final if shared else final[:, Q:]


def k_reciprocal_chunked(q_emb: np.ndarray, g_emb: np.ndarray, k1: int = 20, k2: int = 6,
                         lam: float = 0.3, same_mask: np.ndarray | None = None,
                         chunk: int | None = None, isolate_queries: bool = True,
                         q_keys=None, g_keys=None) -> np.ndarray:
    """То же самое, но запросы обрабатываются пачками по chunk штук.

    ПРИ isolate_queries=True РЕЗУЛЬТАТ НЕ ЗАВИСИТ ОТ РАЗМЕРА ПАЧКИ, и это не пожелание,
    а свойство алгоритма: галерейные строки не используют запросы, а строка запроса
    использует свою self-компоненту и галерею; чужие запросы не участвуют.
    Зачем так: раньше при нехватке памяти код вообще ОТКЛЮЧАЛ ре-ранжирование, а решение
    зависело от числа ВСЕХ запросов — то есть добавление чужих запросов меняло алгоритм
    для нашего. Теперь память ограничивает размер пачки, а не выбор алгоритма."""
    Q, G = len(q_emb), len(g_emb)
    if (q_keys is None)!=(g_keys is None):raise ValueError('Supply both query and gallery keys')
    if q_keys is not None:
        if same_mask is not None:raise ValueError('Use keys or a full mask, not both')
        if len(q_keys)!=Q or len(g_keys)!=G:raise ValueError('Key lengths mismatch')
        step=chunk if chunk and chunk>0 else max(Q,1)
        out=np.empty((Q,G),dtype=np.float32)
        for start in range(0,Q,step):
            end=min(start+step,Q);keys=list(q_keys[start:end])+list(g_keys)
            mask=frame_block_mask(keys,keys)
            out[start:end]=k_reciprocal(q_emb[start:end],g_emb,k1,k2,lam,same_mask=mask,isolate_queries=isolate_queries)
        return out
    if not chunk or chunk >= Q:
        return k_reciprocal(q_emb, g_emb, k1, k2, lam, same_mask=same_mask,
                            isolate_queries=isolate_queries)
    if not isolate_queries:
        raise ValueError("пачки без изоляции запросов дали бы другой результат, чем одним куском")
    out = np.empty((Q, G), dtype=np.float32)
    g_rows = np.arange(Q, Q + G)
    for start in range(0, Q, chunk):
        sel = np.arange(start, min(start + chunk, Q))
        sm = None
        if same_mask is not None:
            rows = np.concatenate([sel, g_rows])
            sm = same_mask[np.ix_(rows, rows)]
        out[sel] = k_reciprocal(q_emb[sel], g_emb, k1, k2, lam, same_mask=sm,
                                isolate_queries=True)
    return out


# --------------------------------------------------------------------------- DBA / QE
def mutual_neighbors(emb: np.ndarray, k: int, block_mask: np.ndarray | None = None):
    """Индексы взаимных top-k соседей для каждого вектора (без себя). → list[np.ndarray]"""
    e = _l2n(emb.astype(np.float32))
    s = e @ e.T
    np.fill_diagonal(s, -np.inf)
    if block_mask is not None:
        s[block_mask] = -np.inf
    k = min(k, len(e) - 1)
    top = np.argpartition(-s, k, axis=1)[:, :k]
    sets = [set(t.tolist()) for t in top]
    return [np.array([j for j in top[i] if i in sets[j]], dtype=np.int64) for i in range(len(e))], s


def dba(emb: np.ndarray, k: int = 3, alpha: float = 1.0, mutual: bool = True, min_sim: float = 0.0,
        block_mask: np.ndarray | None = None) -> np.ndarray:
    """Уточнённые векторы: v' = norm(v + Σ w_j v_j) по k (взаимным) соседям с w = max(sim,0)^alpha.
    block_mask — пары, которые нельзя усреднять (например, один и тот же кадр)."""
    e = _l2n(emb.astype(np.float32))
    nbrs, s = mutual_neighbors(e, k, block_mask)
    out = e.copy()
    for i, idx in enumerate(nbrs):
        if not mutual:
            idx = np.argpartition(-s[i], k)[:k]
        idx = idx[s[i, idx] >= min_sim]
        if len(idx) == 0:
            continue
        w = np.clip(s[i, idx], 0, None) ** alpha
        out[i] = e[i] + (w[:, None] * e[idx]).sum(axis=0)
    return _l2n(out)


def aqe(q_emb: np.ndarray, g_emb: np.ndarray, k: int = 3, alpha: float = 3.0) -> np.ndarray:
    q, g = _l2n(q_emb.astype(np.float32)), _l2n(g_emb.astype(np.float32))
    s = q @ g.T
    k = min(k, g.shape[0])
    top = np.argpartition(-s, k - 1, axis=1)[:, :k]
    out = q.copy()
    for i in range(len(q)):
        w = np.clip(s[i, top[i]], 0, None) ** alpha
        out[i] = q[i] + (w[:, None] * g[top[i]]).sum(axis=0)
    return _l2n(out)


# --------------------------------------------------------------------------- удобные обёртки
def frame_block_mask(keys_a, keys_b) -> np.ndarray:
    """[A, B] True там, где записи из одного кадра (key = '<image_id>#<n>')."""
    fa = np.array([str(k).split("#")[0] for k in keys_a])
    fb = np.array([str(k).split("#")[0] for k in keys_b])
    return fa[:, None] == fb[None, :]
