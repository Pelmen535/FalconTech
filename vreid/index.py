"""FAISS-индекс галереи.

Эмбеддинги L2-нормированы, поэтому inner product == косинусное сходство.
Расстояние d = 1 - cos ∈ [0, 2].
"""
from __future__ import annotations

import numpy as np


class GalleryIndex:
    def __init__(self, emb: np.ndarray, vids: np.ndarray | None = None,
                 cams: np.ndarray | None = None, use_gpu: bool = False):
        import faiss
        emb = np.ascontiguousarray(emb.astype(np.float32))
        self.dim = emb.shape[1]
        self.n = emb.shape[0]
        self.index = faiss.IndexFlatIP(self.dim)
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        self.index.add(emb)
        self.emb = emb
        self.vids = vids if vids is not None else np.arange(self.n)
        self.cams = cams if cams is not None else np.zeros(self.n, dtype=np.int64)

    def search(self, q: np.ndarray, k: int = 100):
        """→ (sims [Q,k], idx [Q,k]) по убыванию сходства."""
        q = np.ascontiguousarray(q.astype(np.float32))
        if q.ndim == 1:
            q = q[None]
        k = min(k, self.n)
        sims, idx = self.index.search(q, k)
        return sims, idx

    def all_sims(self, q: np.ndarray) -> np.ndarray:
        """Полная матрица сходств [Q, N] (для метрик; на 50k×50k — ок в памяти по кускам)."""
        q = np.ascontiguousarray(q.astype(np.float32))
        if q.ndim == 1:
            q = q[None]
        return q @ self.emb.T

    def count_within(self, q: np.ndarray, min_sim: float) -> np.ndarray:
        """Сколько элементов галереи имеет сходство ≥ min_sim с каждым запросом. → [Q]"""
        import faiss
        q = np.ascontiguousarray(q.astype(np.float32))
        if q.ndim == 1:
            q = q[None]
        # range_search работает на CPU-индексах; для GPU считаем через матрицу
        if isinstance(self.index, faiss.IndexFlatIP):
            lims, _, _ = self.index.range_search(q, float(min_sim))
            return np.diff(lims).astype(np.int64)
        return (self.all_sims(q) >= min_sim).sum(axis=1)

    def ids_within(self, q: np.ndarray, min_sim: float) -> list[np.ndarray]:
        """Индексы галереи в радиусе для каждого запроса (для совместного радиуса)."""
        import faiss
        q = np.ascontiguousarray(q.astype(np.float32))
        if q.ndim == 1:
            q = q[None]
        if isinstance(self.index, faiss.IndexFlatIP):
            lims, _, labels = self.index.range_search(q, float(min_sim))
            return [labels[lims[i]:lims[i + 1]] for i in range(len(q))]
        sims = self.all_sims(q)
        return [np.nonzero(row >= min_sim)[0] for row in sims]
