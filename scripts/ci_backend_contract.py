"""Run the real model through the HTTP app and compare it with the offline CLI.

This is a release integration check on synthetic images, not an accuracy test.
The app uses an isolated SQLite gallery; no server or external database is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.app import create_app
from service.config import Settings


def _csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.reader(stream))


def check(data: Path, cli_out: Path, release: Path, work: Path) -> None:
    settings = Settings(core_dir=ROOT, release_dir=release, data_dir=work,
                        device="cpu", demo_dir=work)
    queries = _csv(data / "test_query.csv")[1:]
    gallery = _csv(data / "test_gallery.csv")[1:]
    expected_ranks = {row[0]: row[1:] for row in _csv(cli_out / "submission.csv")}
    expected_candidates = {row[0]: row[1] for row in _csv(cli_out / "candidates.csv")[1:]}
    info = json.loads((cli_out / "run_info.json").read_text(encoding="utf-8"))

    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["gallery_count"] == 0
        assert health.json()["model_sha256"] == info["model_sha256"]
        assert health.json()["recipe_sha256"] == info["recipe_sha256"]
        schema = client.get("/openapi.json")
        assert schema.status_code == 200 and "/v1/search" in schema.json()["paths"]

        for image_id, x, y, w, h in gallery:
            image = data / "images" / f"{image_id}.jpg"
            response = client.post(
                "/v1/gallery/items",
                data={"bbox": json.dumps([int(x), int(y), int(w), int(h)]),
                      "image_id": image_id},
                files={"image": (image.name, image.read_bytes(), "image/jpeg")},
            )
            assert response.status_code == 201, response.text
        assert client.get("/health").json()["gallery_count"] == len(gallery)

        for query_id, x, y, w, h in queries:
            image = data / "images" / f"{query_id}.jpg"
            response = client.post(
                "/v1/search",
                data={"bbox": json.dumps([int(x), int(y), int(w), int(h)]),
                      "query_id": query_id},
                files={"image": (image.name, image.read_bytes(), "image/jpeg")},
            )
            assert response.status_code == 200, response.text
            result = response.json()
            ranked_ids = [item["image_id"] for item in result["ranking"]]
            assert ranked_ids == expected_ranks[query_id]
            assert result["confidence_scale"] == "raw_cosine"
            assert result["accepted"] == (query_id in expected_candidates)
            assert result["candidate"] == (
                result["ranking"][0] if result["accepted"] else None
            )
            if result["accepted"]:
                assert result["candidate"]["image_id"] == expected_candidates[query_id]
            thumb = client.get(result["ranking"][0]["thumbnail_url"])
            assert thumb.status_code == 200 and thumb.headers["content-type"].startswith("image/")

    # A new process would reopen this database without re-encoding the gallery.
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get("/health").json()["gallery_count"] == len(gallery)
    print("[smoke] real-model API, CLI ranking/refusal parity and persistence passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cli-out", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    check(args.data, args.cli_out, args.release, args.work)


if __name__ == "__main__":
    main()
