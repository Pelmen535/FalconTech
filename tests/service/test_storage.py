from concurrent.futures import ThreadPoolExecutor
import sqlite3

import numpy as np
import pytest

from service.storage import DuplicateImageError, GalleryIdentityError, GalleryStore


def make_store(path, **overrides):
    args = dict(model_sha256="a" * 64, recipe_sha256="b" * 64, dimension=3)
    args.update(overrides)
    return GalleryStore(path, **args)


def item(image_id="car"):
    return dict(
        image_id=image_id, image_bytes=b"image bytes", bbox_xywh=(0, 2, 30, 40),
        embedding=np.array([0.6, 0.8, 0], dtype=np.float32),
    )


def test_persistence_and_insertion_order(tmp_path):
    store = make_store(tmp_path)
    metadata = store.add_many([item("z"), item("a")])
    reloaded = make_store(tmp_path)
    assert reloaded.count() == 2
    assert reloaded.list_items() == metadata
    assert metadata[0]["created_at"].endswith("Z")
    assert reloaded.list_items(offset=1, limit=1) == [metadata[1]]
    assert reloaded.image("z") == b"image bytes"
    ids, vectors = reloaded.snapshot()
    assert ids == ["z", "a"]
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, [item()["embedding"], item()["embedding"]])


@pytest.mark.parametrize("override", [
    {"model_sha256": "c" * 64}, {"recipe_sha256": "c" * 64}, {"dimension": 4},
])
def test_identity_mismatch_refuses_existing_gallery(tmp_path, override):
    original = make_store(tmp_path)
    original.add(**item())
    with pytest.raises(GalleryIdentityError, match="identity mismatch"):
        make_store(tmp_path, **override)
    assert make_store(tmp_path).image("car") == b"image bytes"


def test_missing_identity_cannot_adopt_populated_gallery(tmp_path):
    store = make_store(tmp_path)
    store.add(**item())
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM gallery_metadata")
    with pytest.raises(GalleryIdentityError, match="no model/recipe identity"):
        make_store(tmp_path)


def test_duplicate_batch_rolls_back_preceding_insert(tmp_path):
    store = make_store(tmp_path)
    store.add(**item("existing"))
    with pytest.raises(DuplicateImageError):
        store.add_many([item("new"), item("existing")])
    assert [x["image_id"] for x in store.list_items()] == ["existing"]
    with pytest.raises(DuplicateImageError):
        store.add_many([item("new"), item("new")])
    assert store.count() == 1
    with pytest.raises(KeyError):
        store.image("new")


def test_invalid_batch_never_partially_inserts(tmp_path):
    store = make_store(tmp_path)
    bad = item("bad")
    bad["embedding"] = np.zeros(3)
    with pytest.raises(ValueError):
        store.add_many([item("valid"), bad])
    assert store.count() == 0
    assert store.add_many([]) == []


def test_delete_and_empty_snapshot(tmp_path):
    store = make_store(tmp_path)
    store.add(**item())
    assert store.delete("car") is True
    assert store.delete("car") is False
    with pytest.raises(KeyError):
        store.image("car")
    ids, matrix = store.snapshot()
    assert ids == []
    assert matrix.shape == (0, 3)
    assert matrix.dtype == np.float32


@pytest.mark.parametrize("vector", [
    np.zeros(3), np.ones(3), [1, 0], [[1, 0, 0]], [np.nan, 0, 0],
    [np.inf, 0, 0], [1 + 1j, 0, 0], ["1", "0", "0"], [True, False, False],
])
def test_malformed_embeddings_rejected(tmp_path, vector):
    store = make_store(tmp_path)
    value = item()
    value["embedding"] = vector
    with pytest.raises(ValueError, match="embedding"):
        store.add(**value)
    assert store.count() == 0


@pytest.mark.parametrize("image_id", ["", " ", "../x", "a/b", "a\\b", "..", "x\n", "x\x00", "x" * 129])
def test_invalid_ids_rejected(tmp_path, image_id):
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="image_id"):
        store.add(**item(image_id))


@pytest.mark.parametrize("bbox", [(0, 0, 0, 1), (-1, 0, 1, 1), (0, 0, np.inf, 1), (1, 2, 3)])
def test_invalid_bbox_rejected(tmp_path, bbox):
    store = make_store(tmp_path)
    value = item()
    value["bbox_xywh"] = bbox
    with pytest.raises(ValueError, match="bbox_xywh"):
        store.add(**value)


def test_concurrent_operations_use_independent_connections(tmp_path):
    store = make_store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: store.add(**item(str(index))), range(16)))
    assert store.count() == 16
    ids, vectors = store.snapshot()
    assert set(ids) == {str(index) for index in range(16)}
    assert vectors.shape == (16, 3)


def test_near_unit_vectors_preserved_without_silent_normalization(tmp_path):
    store = make_store(tmp_path)
    value = item()
    value["embedding"] = np.array([1.0005, 0, 0], dtype=np.float32)
    store.add(**value)
    np.testing.assert_array_equal(store.snapshot()[1][0], value["embedding"])
