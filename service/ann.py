"""Приближённый поиск ближайших соседей для больших галерей (ТЗ §10).

Зачем это в сервисе, если в конкурсной галерее 750 записей. Затем, что вопрос жюри —
не «работает ли на 750», а «что будет на 10^6». Отвечать на него презентацией нельзя,
поэтому механизм встроен в боевой путь и включается параметром запроса:

    POST /v1/search?mode=exact   полный перебор, ответ совпадает с конкурсным CLI
    POST /v1/search?mode=ann     HNSW-шортлист, затем точный косинус и тот же ре-ранжировщик

На демо-галерее оба режима видно рядом, и разницу (или её отсутствие) можно показать
мышкой, а не слайдом. Измерения на 10^4 / 10^5 / 10^6 — `scripts/ann_benchmark.py`,
результат лежит в `results/ann_scalability.json` и показывается в интерфейсе.

Важное свойство: ANN меняет только то, КАКИЕ кандидаты дошли до ре-ранжирования, и
никогда — как считается сходство. Шортлист строится по одному запросу и всей галерее,
без участия других запросов (ответ 38). Индекс — функция только от галереи.
"""
from __future__ import annotations

import threading
import time

import numpy as np

# Ниже этого размера ANN бессмысленен: полный перебор дешевле, чем обход графа.
MIN_ITEMS_FOR_ANN = 256


class AnnUnavailable(RuntimeError):
    """faiss не установлен или галерея слишком мала для осмысленного индекса."""


class ShortlistIndex:
    """HNSW поверх галереи, пересобирается при изменении состава.

    Ключ кеша — количество записей и хэш их идентификаторов в порядке выдачи хранилища.
    Этого достаточно: вектор в хранилище неизменяем, запись можно только добавить
    или удалить, и оба события меняют ключ.
    """

    def __init__(self, ef_construction: int = 80, ef_search: int = 128, neighbours: int = 32):
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.neighbours = int(neighbours)
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._index = None
        self._built_seconds = 0.0
        self._size = 0

    @staticmethod
    def available() -> bool:
        try:
            import faiss  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _key_for(ids: list[str]) -> tuple:
        import hashlib
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        return len(ids), digest

    def ensure(self, ids: list[str], vectors: np.ndarray):
        if not self.available():
            raise AnnUnavailable("faiss не установлен в этом окружении")
        if len(ids) < MIN_ITEMS_FOR_ANN:
            raise AnnUnavailable(
                f"в галерее {len(ids)} записей: ANN включается от {MIN_ITEMS_FOR_ANN}, "
                f"ниже полный перебор быстрее и точнее")
        import faiss

        key = self._key_for(ids)
        with self._lock:
            if key != self._key:
                index = faiss.IndexHNSWFlat(int(vectors.shape[1]), self.neighbours,
                                            faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = self.ef_construction
                started = time.perf_counter()
                index.add(np.ascontiguousarray(vectors, dtype=np.float32))
                self._built_seconds = time.perf_counter() - started
                self._index, self._key, self._size = index, key, len(ids)
            self._index.hnsw.efSearch = self.ef_search
            return self._index

    def shortlist(self, ids: list[str], vectors: np.ndarray, query: np.ndarray,
                  size: int) -> tuple[np.ndarray, float]:
        """Индексы кандидатов, отсортированные графом. Дальше их считают точно."""
        index = self.ensure(ids, vectors)
        probe = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        started = time.perf_counter()
        _, found = index.search(probe, int(min(size, len(ids))))
        seconds = time.perf_counter() - started
        picked = found[0]
        return picked[picked >= 0].astype(np.int64), seconds

    def stats(self) -> dict:
        return {
            "available": self.available(),
            "kind": "faiss IndexHNSWFlat (inner product)",
            "min_items_for_ann": MIN_ITEMS_FOR_ANN,
            "neighbours": self.neighbours,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "built_for_items": self._size,
            "build_seconds": round(self._built_seconds, 4),
        }
