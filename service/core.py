"""HTTP-independent adapter for the supplied, unmodified ``vreid`` kernel.

The service has its own 1536-D / xywh / raw-cosine boundary. It deliberately
does not reinterpret the project's older Fingerprint and Candidate contracts.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import threading
from pathlib import Path
from types import MappingProxyType

import numpy as np


_IMPORT_LOCK = threading.RLock()
_KNOWN_DIMENSIONS = {
    # The received 19b checkpoint, not an architecture name guessed from a file.
    "608042f64c259b97590e57b81f6c03e7143fba43d9a9c320b4a2a346e9f5fd1c": 1536,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CoreAdapter:
    """Load a release lazily and submit one query to its original reranker.

    A process may use only one ``vreid`` source directory. A conflicting import
    raises an error instead of silently choosing another copy of the kernel.
    Encoding is serialized because the backbone and its device are shared.
    """

    def __init__(self, core_dir: Path, release_dir: Path, device: str = "cpu"):
        self.core_dir = Path(core_dir).resolve()
        self.release_dir = Path(release_dir).resolve()
        self.device = str(device)
        if not (self.core_dir / "vreid" / "rerank.py").is_file():
            raise ValueError(f"core_dir must contain vreid/rerank.py: {self.core_dir}")
        self._model_path = self.release_dir / "model.pt"
        self._recipe_path = self.release_dir / "recipe.json"
        if not self._model_path.is_file() or not self._recipe_path.is_file():
            raise ValueError("release_dir must contain model.pt and recipe.json")
        self.model_sha256 = _sha256(self._model_path)
        recipe_bytes = self._recipe_path.read_bytes()
        self.recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
        recipe = json.loads(recipe_bytes)
        self._validate_recipe(recipe)
        self._recipe = MappingProxyType(recipe)
        self.embedding_sha256 = self._embedding_identity(recipe)
        self._dimension = _KNOWN_DIMENSIONS.get(self.model_sha256)
        self._backbone = None
        self._lock = threading.RLock()

    # Поля рецепта, от которых зависит САМ ВЕКТОР. Всё остальное — порог отказа,
    # параметры ре-ранжирования, пояснения — на извлечение признаков не влияет.
    EMBEDDING_FIELDS = ("crop_pad", "fast_decode", "tta_flip")

    def _embedding_identity(self, recipe: dict) -> str:
        """Отпечаток того, чем посчитан вектор: веса плюс параметры извлечения.

        Именно он, а не хэш всего recipe.json, определяет, можно ли переиспользовать уже
        посчитанную галерею. Иначе правка порога отказа — числа, которое к векторам
        отношения не имеет, — заставляла бы пересчитывать миллион эмбеддингов.
        Изменение весов или любого из EMBEDDING_FIELDS отпечаток меняет, и галерея
        честно отвергается.
        """
        payload = {"model_sha256": self.model_sha256,
                   **{key: recipe.get(key) for key in self.EMBEDDING_FIELDS}}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_recipe(recipe: dict) -> None:
        if not isinstance(recipe, dict):
            raise ValueError("recipe.json must be an object")
        if recipe.get("dba", 0) != 0 or recipe.get("refuse_rate") is not None:
            raise ValueError("The independent-query service requires dba=0 and refuse_rate=null")
        if recipe.get("confidence", "top1") != "top1":
            raise ValueError("The service supports the release's top1 raw-cosine confidence")
        if recipe.get("mask_plate", False):
            raise ValueError("The supplied inference release requires mask_plate=false")
        if recipe.get("topk", 10) != 10 or recipe.get("candidates_topk", 1) != 1:
            raise ValueError("The service requires topk=10 and candidates_topk=1")
        if recipe.get("candidate_min_sim") is not None:
            raise ValueError("candidate_min_sim must be null for this release adapter")
        for key in ("threshold", "crop_pad"):
            value = recipe.get(key, 0.05 if key == "crop_pad" else None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"recipe.{key} must be finite")
        if recipe.get("crop_pad", 0.05) < 0:
            raise ValueError("recipe.crop_pad cannot be negative")
        for key in ("k1", "k2"):
            value = recipe.get(key, 10 if key == "k1" else 3)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"recipe.{key} must be a positive integer")
        lam = recipe.get("lam", 0.3)
        if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not 0 <= lam <= 1:
            raise ValueError("recipe.lam must be a number in [0, 1]")

    def _import_kernel(self, module: str):
        with _IMPORT_LOCK:
            expected = self.core_dir / "vreid"
            for name, loaded in tuple(sys.modules.items()):
                if name == "vreid" or name.startswith("vreid."):
                    source = getattr(loaded, "__file__", None)
                    if source is None or not Path(source).resolve().is_relative_to(expected):
                        raise RuntimeError(
                            f"Conflicting imported kernel {name!r} from {source!r}; "
                            f"expected {expected}. Start a fresh service process."
                        )
            sys.path.insert(0, str(self.core_dir))
            try:
                return importlib.import_module("vreid." + module)
            finally:
                sys.path.remove(str(self.core_dir))

    def _load_backbone(self):
        models = self._import_kernel("models")
        return models.get_backbone("ft:" + str(self._model_path), device=self.device)

    def _ensure_loaded(self):
        # RLock also permits dimension access during an already locked encode.
        with self._lock:
            if self._backbone is None:
                if _sha256(self._model_path) != self.model_sha256:
                    raise RuntimeError("model.pt changed since service initialization; restart the service")
                backbone = self._load_backbone()
                dimension = int(backbone.dim)
                if dimension < 1:
                    raise RuntimeError("Kernel returned an invalid embedding dimension")
                if self._dimension is not None and dimension != self._dimension:
                    raise RuntimeError("Loaded model dimension differs from the recorded release dimension")
                self._dimension = dimension
                self._backbone = backbone
            return self._backbone

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_loaded()
        return int(self._dimension)

    def metadata(self) -> dict:
        # Health/version calls do not load torch or the model. For a new unknown
        # checkpoint dimension is null until its first load; current 19b is known.
        return {
            "model_sha256": self.model_sha256,
            "recipe_sha256": self.recipe_sha256,
            "embedding_sha256": self.embedding_sha256,
            "dimension": self._dimension,
            "threshold": float(self._recipe["threshold"]),
            "device": self.device,
            "loaded": self._backbone is not None,
            "confidence_kind": "raw_cosine",
            "bbox_format": "xywh",
            "rerank": "k_reciprocal_single_query" if self._recipe.get("kr", False) else "cosine",
        }

    def _checked_bbox(self, image_path: Path, bbox_xywh) -> tuple:
        from PIL import Image

        bbox = np.asarray(bbox_xywh, dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= 0):
            raise ValueError("bbox_xywh must contain four finite numbers with positive width and height")
        with Image.open(Path(image_path)) as image:
            width, height = image.size
        x, y, w, h = bbox
        if x >= width or y >= height or x + w <= 0 or y + h <= 0:
            raise ValueError("bbox_xywh does not intersect the image")
        return tuple(bbox)

    def _crop(self, image_path: Path, bbox: tuple):
        """Тот же кроп, что и в боевом CLI: тот же pad, тот же быстрый декод."""
        backbone = self._ensure_loaded()
        extract = self._import_kernel("extract")
        return extract.open_crop(
            Path(image_path), bbox, pad=float(self._recipe.get("crop_pad", 0.05)),
            fast_decode=bool(self._recipe.get("fast_decode", False)),
            target=int(backbone.size), mask_frac=None,
        )

    def _embed_batch(self, batch) -> np.ndarray:
        backbone = self._ensure_loaded()
        vectors = np.asarray(backbone.embed_tensors(batch), dtype=np.float32)
        if self._recipe.get("tta_flip", False):
            import torch
            vectors = vectors + backbone.embed_tensors(torch.flip(batch, dims=[3]))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        if vectors.shape != (1, self.dimension):
            raise RuntimeError("Kernel returned an unexpected embedding shape")
        return self._vectors(vectors, "kernel output")[0].copy()

    def encode(self, image_path: Path, bbox_xywh: tuple[float, float, float, float]) -> np.ndarray:
        """Return the original crop/transform/forward result as float32 [D]."""
        bbox = self._checked_bbox(image_path, bbox_xywh)
        with self._lock:
            backbone = self._ensure_loaded()
            crop = self._crop(image_path, bbox)
            try:
                batch = backbone.transform(crop).unsqueeze(0)
            finally:
                crop.close()
            return self._embed_batch(batch)

    def explain(self, query_path: Path, query_bbox, gallery_path: Path, gallery_bbox,
                regions: int = 3) -> dict:
        """Точное разложение косинуса пары по патчам обеих картинок (ТЗ §10).

        Возвращает по карте на каждую сторону: PNG с наложением и список самых весомых
        клеток. Косинус здесь — тот же, что в выдаче поиска, а не пересчитанный иначе:
        разложение опирается на те же веса BNNeck и тот же autocast (см. service/explain.py).
        """
        from .explain import heatmap_png, patch_contributions, top_regions

        query_bbox = self._checked_bbox(query_path, query_bbox)
        gallery_bbox = self._checked_bbox(gallery_path, gallery_bbox)
        with self._lock:
            backbone = self._ensure_loaded()
            model = getattr(backbone, "m", None)
            if model is None:
                raise ValueError("объяснение определено только для дообученного релиза (ft:)")
            half = bool(getattr(backbone, "half", False))
            crops, batches, vectors = [], [], []
            try:
                for path, bbox in ((query_path, query_bbox), (gallery_path, gallery_bbox)):
                    crop = self._crop(path, bbox)
                    crops.append(crop)
                    batch = backbone.transform(crop).unsqueeze(0)
                    batches.append(batch)
                    vectors.append(self._embed_batch(batch))
                sides = {}
                for name, index, partner in (("query", 0, 1), ("candidate", 1, 0)):
                    result = patch_contributions(model, batches[index], vectors[partner], half=half)
                    sides[name] = {
                        "png": heatmap_png(crops[index], result["contributions"]),
                        "grid": result["grid"],
                        "constant": round(result["constant"], 6),
                        "patch_sum": round(float(result["contributions"].sum()), 6),
                        "reconstruction_error": result["reconstruction_error"],
                        "top_regions": top_regions(result["contributions"], regions),
                    }
            finally:
                for crop in crops:
                    crop.close()
        return {
            "cosine": float(vectors[0] @ vectors[1]),
            "threshold": float(self._recipe["threshold"]),
            "method": "exact additive decomposition of the cosine over patch tokens",
            "sides": sides,
        }

    def _vectors(self, value, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(f"{name} must have shape (N, {self.dimension})")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
        norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-4):
            raise ValueError(f"{name} must contain L2-normalized nonzero vectors")
        # Do not renormalize already normalized kernel output: retain its exact
        # raw cosine values and leave the original rerank normalization intact.
        return np.ascontiguousarray(array)

    def rank(self, query_vector, gallery_vectors, gallery_ids, query_id=None,
             heavy_query_vector=None, heavy_gallery_vectors=None) -> dict:
        """Rank against the entire gallery; a refusal still includes ten results.

        Когда рецепт включает каскад, точный порядок конкурсной сдачи получается только с
        векторами ре-ранкера: их задают heavy_query_vector и heavy_gallery_vectors. Без них
        возвращается порядок ПЕРВОЙ ступени, и это видно в ответе полем ``stage`` — молча
        отдавать другой порядок под видом конкурсного нельзя.
        """
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.dimension,):
            raise ValueError(f"query_vector must have shape ({self.dimension},)")
        query = self._vectors(query[None, :], "query_vector")
        gallery = self._vectors(gallery_vectors, "gallery_vectors")
        ids = list(gallery_ids)
        if len(ids) != len(gallery) or any(not isinstance(item, str) or not item.strip() for item in ids):
            raise ValueError("gallery_ids must contain one nonempty string per gallery vector")
        if len(set(ids)) != len(ids):
            raise ValueError("gallery_ids must be unique")
        if query_id is not None and (not isinstance(query_id, str) or not query_id.strip()):
            raise ValueError("query_id must be a nonempty string or null")
        rerank = self._import_kernel("rerank")
        if query_id is None:
            query_id = "__service_anonymous_query__"
            frame_ids = {item.split("#")[0] for item in ids}
            while query_id in frame_ids:
                query_id += "_"
        keys = [query_id] + ids
        block = rerank.frame_block_mask(keys, keys)
        np.fill_diagonal(block, True)
        valid = ~block[0, 1:]
        if int(valid.sum()) < 10:
            raise ValueError(
                f"At least 10 gallery records excluding the query's frame are required; got {int(valid.sum())}"
            )
        raw = (query @ gallery.T)[0]
        raw[~valid] = -np.inf
        if self._recipe.get("kr", False):
            distance = rerank.k_reciprocal(
                query, gallery, k1=int(self._recipe.get("k1", 10)),
                k2=int(self._recipe.get("k2", 3)),
                lam=float(self._recipe.get("lam", 0.3)),
                same_mask=block, shared=False, isolate_queries=True,
            )[0]
            scores = -distance
            scores[~valid] = -np.inf
        else:
            scores = raw
        stage = "shortlist"
        if self._recipe.get("cascade") and heavy_gallery_vectors is not None:
            from vreid.cascade import heavy_scores, rescore, shortlist
            heavy_q = np.asarray(heavy_query_vector, dtype=np.float32)
            heavy_g = np.asarray(heavy_gallery_vectors, dtype=np.float32)
            if heavy_q.ndim != 1 or heavy_g.ndim != 2 or heavy_g.shape[0] != len(ids)                     or heavy_g.shape[1] != heavy_q.shape[0]:
                raise ValueError("heavy vectors must be [D] and [len(gallery_ids), D]")
            if not (np.isfinite(heavy_q).all() and np.isfinite(heavy_g).all()):
                raise ValueError("heavy vectors must be finite")
            row = scores[None, :]
            blocked = ~np.isfinite(row)
            picked = shortlist(row, int(self._recipe.get("cascade_topk", 30)))
            scores = rescore(row, picked,
                             heavy_scores(heavy_q[None, :], heavy_g, picked, blocked),
                             float(self._recipe.get("cascade_alpha", 0.9)))[0]
            stage = "cascade"
        elif self._recipe.get("cascade"):
            stage = "shortlist_only_cascade_available"
        order = np.argsort(-scores, kind="stable")[:10]
        if not np.isfinite(scores[order]).all():
            raise RuntimeError("Kernel returned nonfinite ranking scores")
        ranking = [
            {"image_id": ids[j], "rank": position, "cosine": float(raw[j])}
            for position, j in enumerate(order, 1)
        ]
        threshold = float(self._recipe["threshold"])
        accepted = ranking[0]["cosine"] >= threshold
        return {
            "ranking": ranking, "accepted": bool(accepted),
            "candidate": dict(ranking[0]) if accepted else None,
            "threshold": threshold, "stage": stage,
        }
