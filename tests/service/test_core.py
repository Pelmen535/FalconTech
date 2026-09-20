"""Adapter tests against the received 19b kernel; no training or accuracy claims."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from service.core import CoreAdapter


ROOT = Path(__file__).resolve().parents[2]
# Ядро — сам репозиторий: vreid/ лежит рядом с service/. Переменная оставлена,
# чтобы можно было проверить адаптер против другой копии ядра.
CORE = Path(os.environ.get("VREID_CORE_DIR") or ROOT).expanduser().resolve()
PACKAGED_INPUTS = Path(__file__).resolve().parent / "fixtures/inputs_csv"
INPUTS = Path(os.environ.get("VREID_TEST_INPUTS_DIR") or PACKAGED_INPUTS).expanduser().resolve()
# Выход боевого CLI, против которого сверяется адаптер сервиса. По умолчанию — актуальная
# сдача: тогда тест доказывает, что HTTP-путь выдаёт ровно тот же топ-10 и то же решение
# об отказе, что и конкурсный прогон, а не «похожий».
EXAMPLE_OUTPUT = Path(os.environ.get("VREID_EXAMPLE_OUTPUT") or ROOT / "submission").expanduser().resolve()
RECIPE = dict(threshold=0.55, confidence="top1", dba=0, refuse_rate=None,
              kr=True, k1=6, k2=2, crop_pad=0.05, mask_plate=False,
              topk=10, candidates_topk=1, candidate_min_sim=None,
              fast_decode=True, tta_flip=False)


@pytest.fixture
def make_adapter(tmp_path):
    def make(**changes):
        if not (CORE / "vreid/rerank.py").is_file():
            pytest.skip("Received original 19b kernel is not present")
        release = tmp_path / "release"
        release.mkdir(exist_ok=True)
        (release / "model.pt").write_bytes(b"synthetic checkpoint; never loaded")
        (release / "recipe.json").write_text(json.dumps(RECIPE | changes), encoding="utf-8")
        adapter = CoreAdapter(CORE, release)
        adapter._dimension = 8
        return adapter
    return make


def normalized(seed=27, count=24):
    vectors = np.random.default_rng(seed).normal(size=(count + 1, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors[0], vectors[1:], [f"gallery-{i}" for i in range(count)]


def test_original_reranker_receives_one_query_and_entire_gallery(make_adapter, monkeypatch):
    adapter = make_adapter(threshold=-1.0)
    query, gallery, ids = normalized(count=27)
    kernel = adapter._import_kernel("rerank")
    original = kernel.k_reciprocal
    calls = []

    def spy(q, g, **kwargs):
        calls.append((q.copy(), g.copy(), kwargs))
        return original(q, g, **kwargs)

    monkeypatch.setattr(kernel, "k_reciprocal", spy)
    result = adapter.rank(query, gallery, ids, query_id="query")
    assert len(calls) == 1
    q, g, settings = calls[0]
    np.testing.assert_array_equal(q, query[None])
    np.testing.assert_array_equal(g, gallery)
    assert settings["isolate_queries"] is True and settings["shared"] is False
    assert settings["same_mask"].shape == (28, 28)
    assert settings["k1"] == 6 and settings["k2"] == 2
    assert len(result["ranking"]) == 10
    assert len({row["image_id"] for row in result["ranking"]}) == 10
    assert result["accepted"] is True
    assert result["candidate"] == result["ranking"][0]
    assert Path(kernel.__file__).resolve().is_relative_to(CORE.resolve())


def test_refusal_preserves_ten_ranked_results(make_adapter):
    query, gallery, ids = normalized()
    accepted = make_adapter(threshold=-1.0).rank(query, gallery, ids)
    refused = make_adapter(threshold=1.1).rank(query, gallery, ids)
    assert refused["ranking"] == accepted["ranking"]
    assert refused["accepted"] is False
    assert refused["candidate"] is None
    assert [row["rank"] for row in refused["ranking"]] == list(range(1, 11))


def test_refusal_uses_selected_candidates_raw_cosine_not_maximum(make_adapter, monkeypatch):
    adapter = make_adapter(threshold=0.55)
    query = np.eye(8, dtype=np.float32)[0]
    gallery = np.tile(np.eye(8, dtype=np.float32)[1], (12, 1))
    gallery[0] = query  # raw maximum 1.0, deliberately not rerank top-1
    gallery[1] = [0.4, np.sqrt(0.84), 0, 0, 0, 0, 0, 0]
    kernel = adapter._import_kernel("rerank")

    def selected_low_cosine(q, g, **kwargs):
        distances = np.arange(len(g), dtype=np.float32)[None] + 1
        distances[0, 1] = 0
        return distances

    monkeypatch.setattr(kernel, "k_reciprocal", selected_low_cosine)
    result = adapter.rank(query, gallery, [str(i) for i in range(len(gallery))])
    assert result["ranking"][0]["image_id"] == "1"
    assert result["ranking"][0]["cosine"] == pytest.approx(0.4)
    assert result["accepted"] is False and result["candidate"] is None


def test_cosine_mode_and_threshold_boundary(make_adapter):
    adapter = make_adapter(kr=False, threshold=1.0)
    ids = [f"gallery-{i}" for i in range(12)]
    # Use an exactly represented unit vector to test the inclusive boundary.
    query = np.eye(8, dtype=np.float32)[0]
    gallery = np.tile(np.eye(8, dtype=np.float32)[1], (12, 1))
    gallery[7] = query
    result = adapter.rank(query, gallery, ids[:12])
    assert result["candidate"]["image_id"] == ids[7]
    assert result["candidate"]["cosine"] == 1.0
    assert result["accepted"] is True


def test_excludes_all_records_of_query_frame_before_counting_ten(make_adapter):
    adapter = make_adapter(kr=False, threshold=-1)
    query, gallery, ids = normalized(count=12)
    ids[:2] = ["frame#0", "frame#1"]
    gallery[:2] = query
    result = adapter.rank(query, gallery, ids, query_id="frame#query")
    assert {row["image_id"] for row in result["ranking"]} == set(ids[2:])
    with pytest.raises(ValueError, match="At least 10.*got 9"):
        adapter.rank(query, gallery[:-1], ids[:-1], query_id="frame#query")


def test_anonymous_query_id_does_not_hide_a_gallery_record(make_adapter):
    adapter = make_adapter(kr=False)
    query, gallery, ids = normalized(count=10)
    ids[0] = "__service_anonymous_query__"
    result = adapter.rank(query, gallery, ids)
    assert {row["image_id"] for row in result["ranking"]} == set(ids)


@pytest.mark.parametrize("damage, message", [
    ("nan", "NaN or infinity"), ("zero", "L2-normalized"),
    ("duplicate", "unique"), ("empty", "nonempty"),
    ("dimension", "shape"), ("few", "At least 10"),
])
def test_invalid_gallery_is_rejected(make_adapter, damage, message):
    adapter = make_adapter()
    query, gallery, ids = normalized(count=12)
    if damage == "nan":
        gallery[0, 0] = np.nan
    elif damage == "zero":
        gallery[0] = 0
    elif damage == "duplicate":
        ids[1] = ids[0]
    elif damage == "empty":
        ids[0] = " "
    elif damage == "dimension":
        gallery = gallery[:, :-1]
    elif damage == "few":
        gallery, ids = gallery[:9], ids[:9]
    with pytest.raises(ValueError, match=message):
        adapter.rank(query, gallery, ids)


@pytest.mark.parametrize("changes", [dict(dba=1), dict(refuse_rate=0.2),
    dict(confidence="margin"), dict(mask_plate=True), dict(candidates_topk=10),
    dict(threshold=float("nan")), dict(k1=0)])
def test_unsupported_recipe_fails_before_model_load(make_adapter, changes):
    with pytest.raises(ValueError):
        make_adapter(**changes)


def test_unknown_checkpoint_health_does_not_guess_dimension_or_load(make_adapter, monkeypatch):
    adapter = make_adapter()
    adapter._dimension = None  # restore constructor state for the unknown fixture hash

    def forbidden_load():
        pytest.fail("Health/metadata must not load an unknown checkpoint")

    monkeypatch.setattr(adapter, "_load_backbone", forbidden_load)
    metadata = adapter.metadata()
    assert metadata["dimension"] is None
    assert metadata["loaded"] is False
    assert metadata["confidence_kind"] == "raw_cosine"
    assert metadata["bbox_format"] == "xywh"


def test_model_change_after_initialization_is_detected(make_adapter, monkeypatch):
    adapter = make_adapter()
    adapter._model_path.write_bytes(b"replaced checkpoint")
    monkeypatch.setattr(adapter, "_load_backbone", lambda: pytest.fail("Changed model was loaded"))
    with pytest.raises(RuntimeError, match="changed since"):
        adapter._ensure_loaded()


def test_loaded_dimension_mismatch_is_rejected(make_adapter, monkeypatch):
    adapter = make_adapter()
    monkeypatch.setattr(adapter, "_load_backbone", lambda: SimpleNamespace(dim=9))
    with pytest.raises(RuntimeError, match="dimension differs"):
        adapter._ensure_loaded()


def test_conflicting_kernel_import_is_rejected(make_adapter, monkeypatch, tmp_path):
    adapter = make_adapter()
    foreign = ModuleType("vreid.foreign")
    foreign.__file__ = str(tmp_path / "another-kernel/vreid/foreign.py")
    monkeypatch.setitem(sys.modules, "vreid.foreign", foreign)
    with pytest.raises(RuntimeError, match="Conflicting imported kernel"):
        adapter._import_kernel("rerank")


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_representative_cached_queries_match_received_output_on_full_750_gallery():
    """Checks adapter parity, not accuracy: labels and fresh image encoding are absent."""
    required = [CORE / "release/model.pt", EXAMPLE_OUTPUT / "embeddings.npy",
                EXAMPLE_OUTPUT / "submission.csv", EXAMPLE_OUTPUT / "candidates.csv",
                INPUTS / "test_gallery.csv", INPUTS / "test_query.csv"]
    if not all(path.is_file() for path in required):
        pytest.skip("Release checkpoint and a competition run in ./submission are required")
    adapter = CoreAdapter(CORE, CORE / "release")
    assert adapter.metadata()["dimension"] == 1536
    query_ids = [row["image_id"] for row in _csv(INPUTS / "test_query.csv")]
    gallery_ids = [row["image_id"] for row in _csv(INPUTS / "test_gallery.csv")]
    # Длины берутся из фактических CSV: числа 1110/750 относятся к опубликованному
    # набору и вшивать их нельзя (ответ 27).
    assert len(gallery_ids) >= 10 and len(query_ids) >= 1
    embeddings = np.load(EXAMPLE_OUTPUT / "embeddings.npy", allow_pickle=False)
    assert embeddings.shape == (len(query_ids) + len(gallery_ids), 1536)
    rankings = {row["query_id"]: [row[f"gallery_id_{i}"] for i in range(1, 11)]
                for row in _csv(EXAMPLE_OUTPUT / "submission.csv")}
    candidates = {row["query_id"]: row for row in _csv(EXAMPLE_OUTPUT / "candidates.csv")}
    # Cover both refusal outcomes and separated positions in the original CSV.
    selected = {0, len(query_ids) // 3, len(query_ids) // 2, len(query_ids) - 1}
    selected.update(next(i for i, key in enumerate(query_ids) if (key in candidates) == accepted)
                    for accepted in (True, False))
    saw = set()
    for index in sorted(selected):
        key = query_ids[index]
        result = adapter.rank(embeddings[index], embeddings[len(query_ids):], gallery_ids, key)
        assert [row["image_id"] for row in result["ranking"]] == rankings[key]
        assert result["accepted"] == (key in candidates)
        saw.add(result["accepted"])
        if result["accepted"]:
            assert result["candidate"]["image_id"] == candidates[key]["gallery_id"]
            # The received writer serializes confidence with five decimals.
            assert result["candidate"]["cosine"] == pytest.approx(
                float(candidates[key]["confidence"]), abs=5.5e-6)
        else:
            assert result["candidate"] is None
    assert saw == {True, False}
    assert adapter.metadata()["loaded"] is False  # no duplicate model inference
