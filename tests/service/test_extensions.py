"""Проверки того, что добавлено поверх базового HTTP-контракта: объяснение, ANN,
демонстрационный набор, PostgreSQL-хранилище. Ни один тест не заявляет точность модели."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service.ann import AnnUnavailable, ShortlistIndex
from service.app import create_app
from service.config import Settings
from service.storage import GalleryIdentityError, PostgresGalleryStore

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------- объяснение
def linear_model(dim_half: int = 4, patches: int = 4):
    """Минимальный ViT-подобный объект: детерминированные токены и настоящий BNNeck.

    Проверять точность разложения на большой сети незачем — утверждение чисто
    алгебраическое: сумма вкладов патчей плюс постоянная часть равна косинусу.
    """
    import torch
    import torch.nn as nn

    class FakeBackbone(nn.Module):
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(11)
            self.tokens = torch.randn(1, patches + 1, dim_half, generator=generator)
            # Параметр нужен только чтобы у модуля было устройство.
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward_features(self, x):
            return self.tokens + 0.0 * x.sum()

    class FakeModel:
        is_vit = True

        def __init__(self):
            self.backbone = FakeBackbone()
            self.bnneck = nn.BatchNorm1d(2 * dim_half)
            generator = torch.Generator().manual_seed(12)
            with torch.no_grad():
                self.bnneck.weight.copy_(torch.rand(2 * dim_half, generator=generator) + 0.5)
                self.bnneck.bias.copy_(torch.randn(2 * dim_half, generator=generator) * 0.1)
                self.bnneck.running_mean.copy_(torch.randn(2 * dim_half, generator=generator) * 0.2)
                self.bnneck.running_var.copy_(torch.rand(2 * dim_half, generator=generator) + 0.5)
            self.bnneck.eval()
            self.dim = 2 * dim_half

        def features(self, x):
            tokens = self.backbone.forward_features(x)
            feature = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
            return feature, self.bnneck(feature)

    return FakeModel()


def test_patch_contributions_reconstruct_the_cosine_exactly():
    import torch
    from service.explain import patch_contributions

    model = linear_model()
    batch = torch.zeros(1, 3, 8, 8)
    partner = np.random.default_rng(3).normal(size=model.dim).astype(np.float32)
    partner /= np.linalg.norm(partner)

    result = patch_contributions(model, batch, partner)
    # Независимый расчёт косинуса через боевой путь модели.
    with torch.no_grad():
        _, bn = model.features(batch)
    own = (bn[0] / torch.linalg.vector_norm(bn[0])).numpy()
    assert result["cosine"] == pytest.approx(float(own @ partner), abs=1e-6)
    assert result["contributions"].sum() + result["constant"] == pytest.approx(
        result["cosine"], abs=1e-6)
    assert result["reconstruction_error"] < 1e-6


def test_explain_rejects_a_backbone_without_bnneck():
    import torch
    from service.explain import ExplainUnavailable, patch_contributions

    class NotAVit:
        is_vit = False
        dim = 8
    with pytest.raises(ExplainUnavailable):
        patch_contributions(NotAVit(), torch.zeros(1, 3, 8, 8), np.zeros(8, dtype=np.float32))


def test_heatmap_is_a_png_of_the_crop_size():
    from service.explain import heatmap_png

    crop = Image.new("RGB", (40, 24), color=(10, 20, 30))
    png = heatmap_png(crop, np.arange(9, dtype=np.float32).reshape(3, 3))
    with Image.open(io.BytesIO(png)) as rendered:
        assert rendered.format == "PNG" and rendered.size == (40, 24)


def test_top_regions_are_centred_and_inside_the_crop():
    from service.explain import top_regions

    values = np.zeros((4, 4), dtype=np.float32)
    values[2, 3] = 5.0
    regions = top_regions(values, count=2)
    assert regions[0]["box_fraction"][:2] == [0.75, 0.5]
    assert all(0 <= r["box_fraction"][0] <= 1 and 0 <= r["box_fraction"][1] <= 1 for r in regions)
    # Центрирование: у лучшей клетки вклад строго больше, у прочих — отрицательный.
    assert regions[0]["contribution"] > 0 > regions[1]["contribution"]


# --------------------------------------------------------------------------- ANN
def test_small_gallery_never_uses_ann():
    index = ShortlistIndex()
    vectors = np.eye(16, 8, dtype=np.float32)
    with pytest.raises(AnnUnavailable):
        index.shortlist([f"g{i}" for i in range(16)], vectors, vectors[0], size=10)


@pytest.mark.skipif(not ShortlistIndex.available(), reason="faiss не установлен")
def test_shortlist_finds_the_planted_neighbour_and_caches_the_index():
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(600, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"g{i}" for i in range(len(vectors))]
    index = ShortlistIndex()

    query = vectors[123] + 0.01 * rng.normal(size=32).astype(np.float32)
    query /= np.linalg.norm(query)
    picked, seconds = index.shortlist(ids, vectors, query, size=50)
    assert 123 in picked.tolist() and len(picked) == 50 and seconds >= 0

    built = index.stats()["build_seconds"]
    index.shortlist(ids, vectors, query, size=50)
    assert index.stats()["build_seconds"] == built, "индекс пересобрался без изменения галереи"

    # Удаление записи обязано инвалидировать кеш: иначе шортлист вернёт исчезнувший id.
    index.shortlist(ids[:-1], vectors[:-1], query, size=50)
    assert index.stats()["built_for_items"] == len(ids) - 1


# ------------------------------------------------------------------ HTTP-контракт
class TinyCore:
    """Ядро-заглушка: считает ранжирование, но не умеет объяснять."""

    dimension = 4
    model_sha256 = "c" * 64
    recipe_sha256 = "d" * 64

    def metadata(self):
        return dict(loaded=True, dimension=self.dimension, device="cpu", threshold=0.5,
                    model_sha256=self.model_sha256, recipe_sha256=self.recipe_sha256,
                    rerank="cosine", confidence_kind="raw_cosine", bbox_format="xywh")

    def encode(self, path, bbox):
        with Image.open(path) as image:
            image.load()
            marker = image.getpixel((0, 0))[0]
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[marker % self.dimension] = 1.0
        return vector

    def rank(self, query, gallery, ids, query_id=None):
        scores = np.asarray(gallery, dtype=np.float32) @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:10]
        ranking = [{"image_id": ids[j], "rank": position, "cosine": float(scores[j])}
                   for position, j in enumerate(order, 1)]
        accepted = ranking[0]["cosine"] >= 0.5
        return {"ranking": ranking, "accepted": accepted,
                "candidate": dict(ranking[0]) if accepted else None, "threshold": 0.5}


def frame(marker: int) -> bytes:
    image = Image.new("RGB", (32, 32), color=(marker, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def demo_bundle(folder: Path, count: int = 12) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "query.json").write_text(json.dumps(
        {"image": "query.jpg", "image_id": "demo-query", "bbox": [0, 0, 20, 20]}), encoding="utf-8")
    (folder / "query.jpg").write_bytes(frame(1))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        rows = ["image_id,x,y,w,h"]
        for i in range(count):
            rows.append(f"demo-{i},0,0,20,20")
            archive.writestr(f"images/demo-{i}.jpg", frame(i))
        archive.writestr("test_gallery.csv", "\n".join(rows))
    (folder / "gallery.zip").write_bytes(buffer.getvalue())


@pytest.fixture
def client(tmp_path):
    demo_bundle(tmp_path / "demo")
    settings = Settings(core_dir=tmp_path / "core", release_dir=tmp_path / "release",
                        data_dir=tmp_path / "data", device="cpu",
                        demo_dir=tmp_path / "demo")
    return TestClient(create_app(settings, TinyCore()))


def test_health_reports_the_storage_backend(client):
    assert client.get("/health").json()["storage"] == "sqlite"


def test_openapi_is_served_and_documents_the_public_paths(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"]
    for path in ("/v1/search", "/v1/explain", "/v1/gallery", "/v1/index/stats"):
        assert path in schema["paths"], path
    # Swagger UI обязан работать без интернета: страница ссылается на локальные файлы.
    page = client.get("/docs")
    assert page.status_code == 200 and "/static/vendor/swagger-ui-bundle.js" in page.text
    assert client.get("/static/vendor/swagger-ui-bundle.js").status_code == 200


def test_static_serving_refuses_paths_outside_the_folder(client):
    assert client.get("/static/../config.py").status_code in (307, 404)
    assert client.get("/static/does-not-exist.js").status_code == 404


def test_demo_endpoints_load_a_working_gallery(client):
    info = client.get("/v1/demo").json()
    assert info["available"] and info["bbox"] == [0, 0, 20, 20]
    assert client.get("/v1/demo/query.jpg").headers["content-type"] == "image/jpeg"

    created = client.post("/v1/demo/gallery")
    assert created.status_code == 201 and created.json()["added"] == 12
    # Повторный вызов на непустой галерее ничего не трогает.
    assert client.post("/v1/demo/gallery").status_code == 409
    assert client.get("/health").json()["gallery_count"] == 12


def test_search_falls_back_to_exact_when_the_gallery_is_too_small_for_ann(client):
    client.post("/v1/demo/gallery")
    with open_demo_query(client) as (files, data):
        answer = client.post("/v1/search?mode=ann", data=data, files=files).json()
    assert answer["search_mode"] == "exact"
    assert any("ANN" in note for note in answer["notes"])
    assert len(answer["ranking"]) == 10


def test_search_rejects_an_unknown_mode(client):
    client.post("/v1/demo/gallery")
    with open_demo_query(client) as (files, data):
        assert client.post("/v1/search?mode=magic", data=data, files=files).status_code == 422


def test_index_stats_says_when_no_benchmark_was_run(client):
    stats = client.get("/v1/index/stats").json()
    assert stats["active_mode"] == "exact" and stats["benchmark"] is None
    assert stats["ann"]["min_items_for_ann"] >= 2


def test_explain_reports_not_implemented_for_a_core_without_it(client):
    client.post("/v1/demo/gallery")
    answer = client.post("/v1/explain", data={"bbox": "[0,0,20,20]", "gallery_id": "demo-0"},
                         files={"image": ("q.jpg", frame(1), "image/jpeg")})
    assert answer.status_code == 501


class open_demo_query:
    """Маленький помощник: запрос из демонстрационного кадра."""

    def __init__(self, client):
        self.client = client

    def __enter__(self):
        return ({"image": ("q.jpg", frame(3), "image/jpeg")}, {"bbox": "[0,0,20,20]"})

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------------- PostgreSQL
DSN = os.environ.get("VREID_TEST_DATABASE_URL", "")


@pytest.mark.skipif(not DSN, reason="VREID_TEST_DATABASE_URL не задан: нет живой базы")
def test_postgres_store_round_trip_and_identity_guard():
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS gallery_items")
        conn.execute("DROP TABLE IF EXISTS gallery_metadata")

    store = PostgresGalleryStore(DSN, "e" * 64, "f" * 64, 3)
    vector = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    store.add("car-a", b"bytes-a", (1, 2, 30, 40), vector)
    store.add("car-b", b"bytes-b", (3, 4, 50, 60), vector[::-1].copy())

    ids, vectors = store.snapshot()
    assert ids == ["car-a", "car-b"] and vectors.shape == (2, 3)
    assert np.allclose(vectors[0], vector, atol=0), "вектор изменился при записи в базу"
    assert store.count() == 2
    assert store.get_item("car-b")["bbox_xywh"] == [3, 4, 50, 60]
    assert store.image("car-a") == b"bytes-a"
    assert store.list_items(offset=1, limit=1)[0]["image_id"] == "car-b"
    assert store.delete("car-a") and not store.delete("car-a")

    with pytest.raises(GalleryIdentityError):
        PostgresGalleryStore(DSN, "0" * 64, "f" * 64, 3)


# --------------------------------------------- привязка галереи к отпечатку извлечения
def adapter_with(tmp_path, recipe_changes: dict):
    """Адаптер поверх настоящего ядра с синтетическим чекпойнтом: веса не грузятся."""
    from service.core import CoreAdapter

    release = tmp_path / ("release-" + str(abs(hash(tuple(sorted(recipe_changes.items()))))))
    release.mkdir(parents=True, exist_ok=True)
    (release / "model.pt").write_bytes(b"synthetic checkpoint; never loaded")
    base = dict(threshold=0.55, confidence="top1", dba=0, refuse_rate=None, kr=True, k1=6, k2=2,
                crop_pad=0.05, mask_plate=False, topk=10, candidates_topk=1,
                candidate_min_sim=None, fast_decode=True, tta_flip=False)
    (release / "recipe.json").write_text(json.dumps(base | recipe_changes), encoding="utf-8")
    return CoreAdapter(ROOT, release)


def test_threshold_change_keeps_the_gallery_but_crop_change_invalidates_it(tmp_path):
    """Порог отказа к векторам отношения не имеет; параметры кропа — имеют.

    Если привязывать галерею к хэшу всего recipe.json, правка одного числа порога
    обесценивала бы уже посчитанные эмбеддинги. Здесь проверяется, что этого не происходит
    и что защита при этом не ослабла.
    """
    base = adapter_with(tmp_path, {})
    other_threshold = adapter_with(tmp_path, {"threshold": 0.42})
    other_rerank = adapter_with(tmp_path, {"k1": 20})
    other_crop = adapter_with(tmp_path, {"crop_pad": 0.2})
    other_decode = adapter_with(tmp_path, {"fast_decode": False})

    assert base.recipe_sha256 != other_threshold.recipe_sha256, "тест должен менять файл"
    assert base.embedding_sha256 == other_threshold.embedding_sha256
    assert base.embedding_sha256 == other_rerank.embedding_sha256
    assert base.embedding_sha256 != other_crop.embedding_sha256
    assert base.embedding_sha256 != other_decode.embedding_sha256
    assert base.metadata()["embedding_sha256"] == base.embedding_sha256
