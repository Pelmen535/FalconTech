"""Приближённый поиск ближайших соседей для больших галерей (ТЗ §10).

Зачем это в сервисе, если в тестовой галерее 750 записей. Затем, что важный вопрос —
не «работает ли на 750», а «что будет на 10^6». Отвечать на него презентацией нельзя,
поэтому механизм встроен в боевой путь и включается параметром запроса:

    POST /v1/search?mode=exact   полный перебор, ответ совпадает с пакетным прогоном
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
from dataclasses import dataclass
import hashlib,json

import numpy as np

# Ниже этого размера ANN бессмысленен: полный перебор дешевле, чем обход графа.
MIN_ITEMS_FOR_ANN = 256


class AnnUnavailable(RuntimeError):
    """faiss не установлен или галерея слишком мала для осмысленного индекса."""

def _content_key(ids,shape,payload):
    digest=hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(',',':')).encode('utf-8'))
    digest.update(str(shape).encode());digest.update(payload)
    return len(ids),digest.hexdigest()

@dataclass(frozen=True,eq=False,init=False)
class PublishedGallery:
    """Own immutable bytes; input mutation cannot change a published generation."""
    ids:tuple
    vectors:np.ndarray
    generation:str
    identity:str
    content_key:tuple
    def __init__(self,ids,vectors,*,generation,identity):
        ids=tuple(ids);value=np.ascontiguousarray(vectors,dtype='<f4')
        if value.ndim!=2 or len(ids)!=len(value) or value.shape[1]<1 or len(set(ids))!=len(ids) or not np.isfinite(value).all():
            raise ValueError('Published ANN gallery requires unique IDs and finite vectors')
        if not all(isinstance(x,str) and x for x in (generation,identity)):
            raise ValueError('Published ANN generation and model context required')
        payload=value.tobytes();sealed=np.frombuffer(payload,dtype='<f4').reshape(value.shape)
        for name,item in [('ids',ids),('vectors',sealed),('generation',generation),('identity',identity),
                          ('content_key',_content_key(ids,value.shape,payload))]:
            object.__setattr__(self,name,item)
    def rebind(self,generation):
        if not isinstance(generation,str) or not generation:raise ValueError('Gallery generation required')
        other=object.__new__(type(self))
        for name in ('ids','vectors','identity','content_key'):object.__setattr__(other,name,getattr(self,name))
        object.__setattr__(other,'generation',generation)
        return other


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

    def clear(self):
        with self._lock:
            self._key=None;self._index=None;self._built_seconds=0.;self._size=0

    @staticmethod
    def available() -> bool:
        try:
            import faiss  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _key_for(ids: list[str], vectors: np.ndarray) -> tuple:
        value=np.ascontiguousarray(vectors,dtype='<f4')
        return _content_key(ids,value.shape,value.tobytes())

    def ensure(self, ids: list[str], vectors: np.ndarray):
        if not self.available():
            raise AnnUnavailable("быстрый поиск недоступен в этой сборке")
        if len(ids) < MIN_ITEMS_FOR_ANN:
            raise AnnUnavailable(
                f"быстрый режим включается от {MIN_ITEMS_FOR_ANN} наблюдений")
        import faiss

        vectors=np.asarray(vectors,dtype=np.float32)
        if vectors.ndim!=2 or vectors.shape[0]!=len(ids) or vectors.shape[1]<1 or not np.isfinite(vectors).all() or len(set(ids))!=len(ids):
            raise ValueError('ANN requires unique IDs and finite gallery vectors')
        key = ('mutable',self._key_for(ids,vectors),self.neighbours,self.ef_construction)
        return self._ensure_verified(key,vectors,len(ids))

    def _ensure_verified(self,key,vectors,size):
        import faiss
        with self._lock:
            if key != self._key:
                index = faiss.IndexHNSWFlat(int(vectors.shape[1]), self.neighbours,
                                            faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = self.ef_construction
                started = time.perf_counter()
                index.add(np.ascontiguousarray(vectors, dtype=np.float32))
                self._built_seconds = time.perf_counter() - started
                self._index, self._key, self._size = index, key, size
            self._index.hnsw.efSearch = self.ef_search
            return self._index

    def ensure_published(self,snapshot):
        if not isinstance(snapshot,PublishedGallery):raise TypeError('Verified immutable PublishedGallery required')
        if not self.available():raise AnnUnavailable('быстрый поиск недоступен в этой сборке')
        if len(snapshot.ids)<MIN_ITEMS_FOR_ANN:raise AnnUnavailable(f'быстрый режим включается от {MIN_ITEMS_FOR_ANN} наблюдений')
        key=('published',snapshot.generation,snapshot.identity,snapshot.content_key,self.neighbours,self.ef_construction)
        return self._ensure_verified(key,snapshot.vectors,len(snapshot.ids))

    def shortlist_published(self,snapshot,query,size):
        index=self.ensure_published(snapshot)
        probe=np.ascontiguousarray(np.asarray(query,dtype=np.float32).reshape(1,-1))
        if probe.shape[1]!=snapshot.vectors.shape[1] or not np.isfinite(probe).all() or isinstance(size,bool) or int(size)<1:
            raise ValueError('Invalid ANN query or shortlist size')
        started=time.perf_counter()
        # Keep this index lease even if another thread publishes a new snapshot.
        with self._lock:_,found=index.search(probe,int(min(size,len(snapshot.ids))))
        picked=found[0]
        return picked[picked>=0].astype(np.int64),time.perf_counter()-started

    def shortlist(self, ids: list[str], vectors: np.ndarray, query: np.ndarray,
                  size: int) -> tuple[np.ndarray, float]:
        """Индексы кандидатов, отсортированные графом. Дальше их считают точно."""
        index = self.ensure(ids, vectors)
        probe = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if probe.shape[1]!=vectors.shape[1] or not np.isfinite(probe).all() or isinstance(size,bool) or int(size)<1:
            raise ValueError('Invalid ANN query or shortlist size')
        started = time.perf_counter()
        with self._lock:
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
