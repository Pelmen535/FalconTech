"""Хранилище галереи: кадры, рамки и уже нормированные векторы в одной транзакции.

Два бэкенда с одним интерфейсом (ТЗ §6 — хранение эмбеддингов и метаданных в СУБД):

  GalleryStore          SQLite в файле. Ничего не требует снаружи, используется в тестах
                        и при запуске одним процессом.
  PostgresGalleryStore  PostgreSQL с расширением pgvector. Этот вариант поднимает
                        docker compose: база — отдельный контейнер, как и требует
                        микросервисное разделение.

Выбор делает `open_store` по строке подключения. Обе реализации проверяют, что
галерея принадлежит той же модели, тому же рецепту и той же размерности: индекс,
собранный другой моделью, молча переиспользовать нельзя.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import unicodedata
from typing import Iterator

import numpy as np

SCHEMA_VERSION = "1"


class DuplicateImageError(ValueError):
    """One or more image IDs already belong to this gallery."""


class GalleryIdentityError(ValueError):
    """Persisted gallery belongs to another model, recipe or dimension."""


def validate_image_id(value: str) -> str:
    """Идентификатор наблюдения. Ограничения одинаковы для обоих бэкендов, чтобы
    галерея, перенесённая между ними, оставалась той же галереей."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 128
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value)
    ):
        raise ValueError("image_id must be 1-128 characters without paths or control characters")
    return value


def prepare_item(item: dict, dimension: int) -> tuple:
    """Проверить запись и привести к виду, который кладётся в таблицу.

    Вектор обязан прийти уже L2-нормированным: нормализация здесь означала бы, что
    хранилище тихо меняет число, по которому потом считается порог отказа.
    """
    image_id = validate_image_id(item["image_id"])
    raw = item["image_bytes"]
    if not isinstance(raw, (bytes, bytearray, memoryview)) or not raw:
        raise ValueError("image_bytes must be nonempty bytes")
    try:
        bbox = np.asarray(item["bbox_xywh"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox_xywh must contain four finite numbers") from exc
    if (
        bbox.shape != (4,)
        or not np.isfinite(bbox).all()
        or np.any(bbox[:2] < 0)
        or np.any(bbox[2:] <= 0)
    ):
        raise ValueError("bbox_xywh must have finite x,y >= 0 and width,height > 0")
    try:
        original = np.asarray(item["embedding"])
        if original.dtype.kind not in "fiu":
            raise ValueError("embedding must contain real numbers")
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(original, dtype="<f4")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("embedding must contain real float32-compatible numbers") from exc
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f"embedding must be a finite vector of dimension {dimension}")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("embedding must already be L2-normalized (tolerance 1e-3)")
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return image_id, bytes(raw), json.dumps(bbox.tolist()), vector, created_at


def expected_identity(model_sha256: str, recipe_sha256: str, dimension: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "recipe_sha256": recipe_sha256,
        "dimension": str(dimension),
    }


def identity_mismatch(actual: dict, expected: dict) -> str:
    return ", ".join(key for key in sorted(set(expected) | set(actual))
                     if actual.get(key) != expected.get(key))


def check_arguments(model_sha256: str, recipe_sha256: str, dimension: int) -> None:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not isinstance(model_sha256, str) or not model_sha256:
        raise ValueError("model_sha256 must be a nonempty string")
    if not isinstance(recipe_sha256, str) or not recipe_sha256:
        raise ValueError("recipe_sha256 must be a nonempty string")


def check_paging(offset: int, limit: int) -> None:
    if (
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
        or isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("offset and limit must be nonnegative integers")


# --------------------------------------------------------------------------- SQLite
class GalleryStore:
    """Persistent, insertion-ordered gallery with one connection per operation."""

    backend = "sqlite"

    def __init__(
        self,
        data_dir: Path,
        model_sha256: str,
        recipe_sha256: str,
        dimension: int,
    ) -> None:
        check_arguments(model_sha256, recipe_sha256, dimension)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "gallery.sqlite3"
        self.dimension = dimension
        self.model_sha256 = model_sha256
        self.recipe_sha256 = recipe_sha256
        expected = expected_identity(model_sha256, recipe_sha256, dimension)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS gallery_metadata "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                actual = dict(conn.execute("SELECT key, value FROM gallery_metadata"))
                if actual and actual != expected:
                    raise GalleryIdentityError(
                        f"Gallery identity mismatch ({identity_mismatch(actual, expected)}); "
                        f"use a separate data directory"
                    )
                if not actual:
                    # Missing identity in a populated file cannot be repaired by guessing.
                    table = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gallery_items'"
                    ).fetchone()
                    if table and conn.execute("SELECT 1 FROM gallery_items LIMIT 1").fetchone():
                        raise GalleryIdentityError("Populated gallery has no model/recipe identity")
                    conn.executemany(
                        "INSERT INTO gallery_metadata (key, value) VALUES (?, ?)", expected.items()
                    )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS gallery_items ("
                    "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "image_id TEXT NOT NULL UNIQUE, image_bytes BLOB NOT NULL, "
                    "bbox_xywh TEXT NOT NULL, embedding BLOB NOT NULL, created_at TEXT NOT NULL)"
                )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _image_id(value: str) -> str:
        return validate_image_id(value)

    def _prepare(self, item: dict) -> tuple:
        image_id, raw, bbox, vector, created_at = prepare_item(item, self.dimension)
        return image_id, raw, bbox, vector.tobytes(), created_at

    @staticmethod
    def _metadata(row) -> dict:
        return {
            "image_id": row["image_id"],
            "bbox_xywh": json.loads(row["bbox_xywh"]),
            "created_at": row["created_at"],
        }

    def add(self, image_id: str, image_bytes: bytes, bbox_xywh: tuple, embedding: np.ndarray) -> dict:
        return self.add_many([{
            "image_id": image_id,
            "image_bytes": image_bytes,
            "bbox_xywh": bbox_xywh,
            "embedding": embedding,
        }])[0]

    def add_many(self, items: list[dict]) -> list[dict]:
        """Insert all items, or none if validation or any unique ID check fails."""
        prepared = [self._prepare(item) for item in items]
        ids = [row[0] for row in prepared]
        if len(ids) != len(set(ids)):
            raise DuplicateImageError("Duplicate image_id within batch")
        if not prepared:
            return []
        with self._connection() as conn:
            try:
                with conn:
                    conn.executemany(
                        "INSERT INTO gallery_items "
                        "(image_id,image_bytes,bbox_xywh,embedding,created_at) VALUES (?,?,?,?,?)",
                        prepared,
                    )
            except sqlite3.IntegrityError as exc:
                if "gallery_items.image_id" in str(exc):
                    raise DuplicateImageError("image_id already exists in gallery") from exc
                raise
        return [
            {"image_id": row[0], "bbox_xywh": json.loads(row[2]), "created_at": row[4]}
            for row in prepared
        ]

    def list_items(self, offset: int = 0, limit: int = 100) -> list[dict]:
        check_paging(offset, limit)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT image_id,bbox_xywh,created_at FROM gallery_items ORDER BY sequence LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._metadata(row) for row in rows]

    def count(self) -> int:
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM gallery_items").fetchone()[0])

    def get_item(self, image_id: str) -> dict:
        image_id = validate_image_id(image_id)
        with self._connection() as conn:
            row = conn.execute(
                'SELECT image_id,bbox_xywh,created_at FROM gallery_items WHERE image_id=?', (image_id,)
            ).fetchone()
        if row is None:
            raise KeyError(image_id)
        return self._metadata(row)

    def snapshot(self) -> tuple[list[str], np.ndarray]:
        """Return IDs and vectors from one consistent, insertion-ordered SELECT."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT image_id,embedding FROM gallery_items ORDER BY sequence"
            ).fetchall()
        if not rows:
            return [], np.empty((0, self.dimension), dtype=np.float32)
        vectors = np.stack([
            np.frombuffer(row["embedding"], dtype="<f4") for row in rows
        ]).astype(np.float32, copy=False)
        if vectors.shape != (len(rows), self.dimension) or not np.isfinite(vectors).all():
            raise ValueError("Persisted gallery contains malformed embeddings")
        return [row["image_id"] for row in rows], vectors

    def image(self, image_id: str) -> bytes:
        image_id = validate_image_id(image_id)
        with self._connection() as conn:
            row = conn.execute(
                "SELECT image_bytes FROM gallery_items WHERE image_id=?", (image_id,)
            ).fetchone()
        if row is None:
            raise KeyError(image_id)
        return bytes(row[0])

    def delete(self, image_id: str) -> bool:
        image_id = validate_image_id(image_id)
        with self._connection() as conn:
            with conn:
                cursor = conn.execute("DELETE FROM gallery_items WHERE image_id=?", (image_id,))
                return cursor.rowcount == 1


# --------------------------------------------------------------------- PostgreSQL
class PostgresGalleryStore:
    """Та же галерея в PostgreSQL с расширением pgvector.

    Вектор хранится типом `vector(D)`, а не как байты: тогда база умеет считать
    расстояния сама и для городских объёмов можно построить HNSW-индекс прямо в СУБД.
    Сервис при этом продолжает считать сходство ядром — чтобы ответ HTTP совпадал с
    конкурсным CLI бит в бит, а не «почти». `snapshot()` отдаёт те же векторы, что
    легли в базу, и дальше работает тот же код, что и на SQLite.
    """

    backend = "postgresql+pgvector"

    def __init__(self, dsn: str, model_sha256: str, recipe_sha256: str, dimension: int) -> None:
        check_arguments(model_sha256, recipe_sha256, dimension)
        import psycopg

        self.dsn = dsn
        self.dimension = dimension
        self.model_sha256 = model_sha256
        self.recipe_sha256 = recipe_sha256
        self._psycopg = psycopg
        expected = expected_identity(model_sha256, recipe_sha256, dimension)
        with self._connection() as conn:
            with conn.transaction():
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.execute("CREATE TABLE IF NOT EXISTS gallery_metadata "
                             "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS gallery_items ("
                    "sequence BIGSERIAL PRIMARY KEY, image_id TEXT NOT NULL UNIQUE, "
                    "image_bytes BYTEA NOT NULL, bbox_xywh JSONB NOT NULL, "
                    f"embedding vector({dimension}) NOT NULL, created_at TEXT NOT NULL)")
                actual = dict(conn.execute("SELECT key, value FROM gallery_metadata").fetchall())
                if actual and actual != expected:
                    raise GalleryIdentityError(
                        f"Gallery identity mismatch ({identity_mismatch(actual, expected)}); "
                        f"use a separate database")
                if not actual:
                    populated = conn.execute("SELECT 1 FROM gallery_items LIMIT 1").fetchone()
                    if populated:
                        raise GalleryIdentityError("Populated gallery has no model/recipe identity")
                    conn.execute("INSERT INTO gallery_metadata (key, value) "
                                 "SELECT * FROM unnest(%s::text[], %s::text[])",
                                 (list(expected), [expected[k] for k in expected]))

    @contextmanager
    def _connection(self):
        conn = self._psycopg.connect(self.dsn, autocommit=True, connect_timeout=15)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _vector_literal(vector: np.ndarray) -> str:
        # pgvector принимает текстовый литерал '[a,b,...]'. repr(float32) печатает
        # ровно столько цифр, сколько нужно для обратного разбора без потерь.
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    def _prepare(self, item: dict) -> tuple:
        image_id, raw, bbox, vector, created_at = prepare_item(item, self.dimension)
        return image_id, raw, bbox, self._vector_literal(vector), created_at

    def add(self, image_id: str, image_bytes: bytes, bbox_xywh: tuple, embedding: np.ndarray) -> dict:
        return self.add_many([{
            "image_id": image_id, "image_bytes": image_bytes,
            "bbox_xywh": bbox_xywh, "embedding": embedding,
        }])[0]

    def add_many(self, items: list[dict]) -> list[dict]:
        prepared = [self._prepare(item) for item in items]
        ids = [row[0] for row in prepared]
        if len(ids) != len(set(ids)):
            raise DuplicateImageError("Duplicate image_id within batch")
        if not prepared:
            return []
        with self._connection() as conn:
            try:
                with conn.transaction():
                    conn.cursor().executemany(
                        "INSERT INTO gallery_items "
                        "(image_id,image_bytes,bbox_xywh,embedding,created_at) "
                        "VALUES (%s,%s,%s::jsonb,%s::vector,%s)", prepared)
            except self._psycopg.errors.UniqueViolation as exc:
                raise DuplicateImageError("image_id already exists in gallery") from exc
        return [{"image_id": row[0], "bbox_xywh": json.loads(row[2]), "created_at": row[4]}
                for row in prepared]

    def list_items(self, offset: int = 0, limit: int = 100) -> list[dict]:
        check_paging(offset, limit)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT image_id,bbox_xywh,created_at FROM gallery_items "
                "ORDER BY sequence LIMIT %s OFFSET %s", (limit, offset)).fetchall()
        return [{"image_id": r[0], "bbox_xywh": r[1], "created_at": r[2]} for r in rows]

    def count(self) -> int:
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM gallery_items").fetchone()[0])

    def get_item(self, image_id: str) -> dict:
        image_id = validate_image_id(image_id)
        with self._connection() as conn:
            row = conn.execute("SELECT image_id,bbox_xywh,created_at FROM gallery_items "
                               "WHERE image_id=%s", (image_id,)).fetchone()
        if row is None:
            raise KeyError(image_id)
        return {"image_id": row[0], "bbox_xywh": row[1], "created_at": row[2]}

    def snapshot(self) -> tuple[list[str], np.ndarray]:
        with self._connection() as conn:
            rows = conn.execute("SELECT image_id, embedding::text FROM gallery_items "
                                "ORDER BY sequence").fetchall()
        if not rows:
            return [], np.empty((0, self.dimension), dtype=np.float32)
        # pgvector печатает вектор как '[1,2,3]' — это корректный JSON-массив.
        vectors = np.stack([np.asarray(json.loads(row[1]), dtype=np.float32) for row in rows])
        if vectors.shape != (len(rows), self.dimension) or not np.isfinite(vectors).all():
            raise ValueError("Persisted gallery contains malformed embeddings")
        return [row[0] for row in rows], vectors

    def image(self, image_id: str) -> bytes:
        image_id = validate_image_id(image_id)
        with self._connection() as conn:
            row = conn.execute("SELECT image_bytes FROM gallery_items WHERE image_id=%s",
                               (image_id,)).fetchone()
        if row is None:
            raise KeyError(image_id)
        return bytes(row[0])

    def delete(self, image_id: str) -> bool:
        image_id = validate_image_id(image_id)
        with self._connection() as conn:
            with conn.transaction():
                cursor = conn.execute("DELETE FROM gallery_items WHERE image_id=%s", (image_id,))
                return cursor.rowcount == 1


def open_store(settings, model_sha256: str, recipe_sha256: str, dimension: int):
    """Выбрать бэкенд по настройкам: строка подключения есть — PostgreSQL, нет — SQLite."""
    dsn = getattr(settings, "database_url", "") or ""
    if dsn:
        return PostgresGalleryStore(dsn, model_sha256, recipe_sha256, dimension)
    return GalleryStore(settings.data_dir, model_sha256, recipe_sha256, dimension)
