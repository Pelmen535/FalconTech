"""HTTP behavior tests with a tiny deterministic encoder, never accuracy claims."""
from dataclasses import replace
from io import BytesIO, StringIO
import csv
import zipfile

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from service.app import create_app
from service.config import Settings


class FakeCore:
    dimension = 8
    model_sha256 = "a" * 64
    recipe_sha256 = "b" * 64

    def __init__(self, accepted=True, fail_on_encode=None):
        self.accepted = accepted
        self.encode_calls = []
        self.rank_calls = []
        self.fail_on_encode = fail_on_encode

    def metadata(self):
        return dict(loaded=bool(self.encode_calls), dimension=self.dimension,
                    device="cpu", model_sha256=self.model_sha256,
                    recipe_sha256=self.recipe_sha256, threshold=0.55)

    def encode(self, path, bbox):
        with Image.open(path) as image:
            image.load()
            marker = image.getpixel((0, 0))[0]
        self.encode_calls.append((marker, tuple(bbox)))
        if self.fail_on_encode == len(self.encode_calls):
            raise ValueError("Test encoder rejected this input")
        result = np.zeros(self.dimension, dtype=np.float32)
        result[marker % self.dimension] = 1.0
        return result

    def rank(self, query_vector, gallery_vectors, gallery_ids, query_id=None):
        self.rank_calls.append((query_vector.copy(), gallery_vectors.copy(), list(gallery_ids), query_id))
        ranking = [dict(image_id=key, rank=index + 1, cosine=0.8 - index / 10)
                   for index, key in enumerate(gallery_ids[:10])]
        return dict(ranking=ranking, accepted=self.accepted,
                    candidate=ranking[0].copy() if self.accepted else None, threshold=0.55)


def png(marker=1):
    stream = BytesIO()
    Image.new("RGB", (32, 24), (marker, 30, 40)).save(stream, format="PNG")
    return stream.getvalue()


def settings_at(tmp_path, **overrides):
    settings = Settings(core_dir=tmp_path / "core", release_dir=tmp_path / "release",
                        data_dir=tmp_path / "data")
    return replace(settings, **overrides)


@pytest.fixture
def service(tmp_path):
    core = FakeCore()
    settings = settings_at(tmp_path)
    with TestClient(create_app(settings, core)) as client:
        yield client, core, settings


def add(client, key="car-0", marker=1, bbox="[1,2,20,10]", **kwargs):
    return client.post("/v1/gallery/items", data={"image_id": key, "bbox": bbox},
                       files={"image": ("vehicle.png", png(marker), "image/png")}, **kwargs)


def search(client, **fields):
    return client.post("/v1/search", data={"bbox": "[1,2,20,10]", **fields},
                       files={"image": ("query.png", png(7), "image/png")})


def archive(keys, *, broken_image=None, bad_bbox=None, extra_member=None):
    manifest = StringIO(newline="")
    writer = csv.DictWriter(manifest, fieldnames=["image_id", "x", "y", "w", "h"])
    writer.writeheader()
    for key in keys:
        writer.writerow(dict(image_id=key, x=1, y=2, w=40 if key == bad_bbox else 20, h=10))
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as package:
        package.writestr("test_gallery.csv", manifest.getvalue())
        for index, key in enumerate(dict.fromkeys(keys)):
            package.writestr("images/" + key + ".png", b"broken" if key == broken_image else png(index))
        if extra_member:
            package.writestr(extra_member, b"untrusted extra file")
    return stream.getvalue()


def import_zip(client, data):
    return client.post("/v1/gallery/import", files={"archive": ("gallery.zip", data, "application/zip")})


def test_thumbnail_shows_annotated_vehicle_and_keeps_original(service):
    client, _, _ = service
    row = add(client, bbox='[1,2,20,10]').json()
    thumb = client.get(row['thumbnail_url'])
    assert thumb.status_code == 200
    with Image.open(BytesIO(thumb.content)) as image:
        assert image.size == (20, 10)
    assert client.get(row['image_url']).content == png(1)


def test_add_read_image_and_delete(service):
    client, core, _ = service
    assert client.get("/health").json()["gallery_count"] == 0
    response = add(client, key="машина 1")
    assert response.status_code == 201, response.text
    item = response.json()
    assert item["image_id"] == "машина 1"
    assert item["bbox_xywh"] == [1, 2, 20, 10]
    assert item["created_at"].endswith("Z")
    assert core.encode_calls == [(1, (1, 2, 20, 10))]
    assert client.get("/v1/gallery").json() == {"count": 1, "items": [item]}
    image = client.get(item["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content == png(1)
    assert client.delete("/v1/gallery/items/машина 1").status_code == 204
    assert client.get(item["image_url"]).status_code == 404
    assert client.delete("/v1/gallery/items/машина 1").status_code == 404
    assert client.get("/v1/gallery").json() == {"count": 0, "items": []}


@pytest.mark.parametrize("accepted", [True, False])
def test_search_returns_ten_and_preserves_core_refusal(service, accepted):
    client, core, _ = service
    core.accepted = accepted
    keys = [f"car-{index}" for index in range(12)]
    assert import_zip(client, archive(keys)).status_code == 201
    response = search(client, query_id="query-7")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["accepted"] is accepted
    assert len(result["ranking"]) == 10
    assert [item["image_id"] for item in result["ranking"]] == keys[:10]
    assert [item["rank"] for item in result["ranking"]] == list(range(1, 11))
    assert result["ranking"][-1]["cosine"] == pytest.approx(-0.1)
    assert result["candidate"] == (result["ranking"][0] if accepted else None)
    assert result["confidence_scale"] == "raw_cosine"
    assert result["gallery_count"] == 12
    assert result["model_sha256"] == core.model_sha256
    assert result["recipe_sha256"] == core.recipe_sha256
    assert result["threshold"] == 0.55
    assert client.get(result["ranking"][0]["image_url"]).status_code == 200
    query_vector, gallery_vectors, passed_ids, query_id = core.rank_calls[-1]
    assert passed_ids == keys  # Whole gallery, not an early top-10 shortlist.
    assert gallery_vectors.shape == (12, 8)
    np.testing.assert_array_equal(gallery_vectors, np.eye(8, dtype=np.float32)[np.arange(12) % 8])
    np.testing.assert_array_equal(query_vector, np.eye(8, dtype=np.float32)[7])
    assert query_id == "query-7"


def test_search_requires_ten_before_running_encoder(service):
    client, core, _ = service
    assert search(client).status_code == 409
    assert core.encode_calls == []
    assert import_zip(client, archive([f"car-{i}" for i in range(9)])).status_code == 201
    count = len(core.encode_calls)
    assert search(client).status_code == 409
    assert len(core.encode_calls) == count
    assert core.rank_calls == []


@pytest.mark.parametrize("bbox", ["not json", "[1,2,3]", "[true,0,2,2]", "[-1,0,3,3]",
                                   "[0,0,0,5]", "[0,0,NaN,5]", "[0,0,33,24]", "null"])
def test_bad_bbox_rejected_without_mutation(service, bbox):
    client, core, _ = service
    response = add(client, bbox=bbox)
    assert response.status_code == 422, response.text
    assert client.get("/v1/gallery").json()["count"] == 0
    assert core.encode_calls == []


def test_unreadable_image_rejected_without_mutation(service):
    client, core, _ = service
    response = client.post("/v1/gallery/items", data={"bbox": "[0,0,1,1]"},
                           files={"image": ("fake.png", b"not an image", "image/png")})
    assert response.status_code == 422
    assert client.get("/v1/gallery").json()["count"] == 0
    assert core.encode_calls == []


def test_duplicate_single_item_does_not_replace_original(service):
    client, core, _ = service
    original = add(client).json()
    assert add(client, marker=7).status_code == 409
    assert len(core.encode_calls) == 1
    assert client.get(original["image_url"]).content == png(1)


def test_zip_import_has_no_partial_writes_on_existing_id(service):
    client, core, _ = service
    original = add(client, key="existing").json()
    response = import_zip(client, archive(["new-first", "existing"]))
    assert response.status_code == 409
    assert client.get("/v1/gallery").json() == {"count": 1, "items": [original]}
    assert len(core.encode_calls) == 1


@pytest.mark.parametrize("problem", ["duplicate", "bad_image", "bad_bbox"])
def test_zip_validation_is_atomic(service, problem):
    client, core, _ = service
    raw = archive(["first", "first"] if problem == "duplicate" else ["first", "second"],
                  broken_image="second" if problem == "bad_image" else None,
                  bad_bbox="second" if problem == "bad_bbox" else None)
    assert import_zip(client, raw).status_code == 422
    assert client.get("/v1/gallery").json()["count"] == 0
    assert core.encode_calls == []


def test_failed_encoding_rolls_back_import_and_releases_busy_gate(tmp_path):
    core = FakeCore(fail_on_encode=2)
    with TestClient(create_app(settings_at(tmp_path), core)) as client:
        assert import_zip(client, archive(["first", "second"])).status_code == 422
        assert client.get("/v1/gallery").json()["count"] == 0
        assert client.get("/health").json()["busy"] is False
        assert add(client).status_code == 201


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:/escape.txt", "..\\escape.txt"])
def test_unsafe_zip_path_rejected_without_import(service, member):
    client, core, settings = service
    response = import_zip(client, archive(["car"], extra_member=member))
    assert response.status_code == 422, response.text
    assert client.get("/v1/gallery").json()["count"] == 0
    assert core.encode_calls == []
    assert not (settings.data_dir.parent / "escape.txt").exists()


def test_api_key_protects_reads_writes_and_images_but_not_health(tmp_path):
    core = FakeCore()
    with TestClient(create_app(settings_at(tmp_path, api_key="secret-test-key"), core)) as client:
        assert client.get("/health").status_code == 200
        for path in ("/v1/model", "/v1/gallery", "/v1/gallery/items/car-0/image"):
            assert client.get(path).status_code == 401
            assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
        assert add(client).status_code == 401
        assert client.delete("/v1/gallery/items/car-0").status_code == 401
        assert import_zip(client, archive(["car"])).status_code == 401
        assert search(client).status_code == 401
        assert core.encode_calls == []
        client.headers["X-API-Key"] = "secret-test-key"
        assert add(client).status_code == 201
        assert client.get("/v1/gallery/items/car-0/image").content == png(1)
        metadata = client.get("/v1/model").json()
        assert metadata["confidence_scale"] == "raw_cosine"
        assert metadata["bbox_format"] == "xywh"
        assert metadata["dimension"] == 8


def test_restart_preserves_images_boxes_and_vectors_without_reencoding(tmp_path):
    settings = settings_at(tmp_path)
    keys = [f"car-{index}" for index in range(10)]
    with TestClient(create_app(settings, FakeCore())) as client:
        assert import_zip(client, archive(keys)).json() == {"added": 10, "count": 10}
        original = client.get("/v1/gallery").json()
    fresh_core = FakeCore()
    with TestClient(create_app(settings, fresh_core)) as client:
        assert client.get("/health").json()["gallery_count"] == 10
        assert client.get("/v1/gallery").json() == original
        assert client.get(original["items"][7]["image_url"]).content == png(7)
        assert fresh_core.encode_calls == []
        assert search(client).status_code == 200
        assert len(fresh_core.encode_calls) == 1  # Query only; persisted gallery survives.
        np.testing.assert_array_equal(fresh_core.rank_calls[0][1],
                                      np.eye(8, dtype=np.float32)[np.arange(10) % 8])


def test_gallery_limit_and_pagination_are_enforced(tmp_path):
    with TestClient(create_app(settings_at(tmp_path, max_gallery_items=3), FakeCore())) as client:
        assert import_zip(client, archive(["a", "b", "c"])).status_code == 201
        assert add(client, key="d").status_code == 409
        page = client.get("/v1/gallery", params={"offset": 1, "limit": 1}).json()
        assert page["count"] == 3
        assert [item["image_id"] for item in page["items"]] == ["b"]
        assert client.get("/v1/gallery", params={"offset": -1}).status_code == 422
        assert client.get("/v1/gallery", params={"limit": 501}).status_code == 422
