"""Small deterministic input and output-contract check for the release image.

The pictures are synthetic and carry no accuracy claim. Run inside the built
image, so ``create`` and ``check`` use the same dependencies as inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


QUERY_IDS = ("smoke_q0", "smoke_q1")
GALLERY_IDS = tuple(f"smoke_g{i:02d}" for i in range(10))


def create(data: Path) -> None:
    images = data / "images"
    images.mkdir(parents=True, exist_ok=True)
    for position, image_id in enumerate((*QUERY_IDS, *GALLERY_IDS)):
        image = Image.new("RGB", (160, 120), (35 + position * 7, 55, 70))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 35, 138, 91), fill=(90 + position * 10, 110, 130))
        draw.ellipse((38, 81, 58, 101), fill=(18, 18, 18))
        draw.ellipse((105, 81, 125, 101), fill=(18, 18, 18))
        image.save(images / f"{image_id}.jpg", format="JPEG", quality=90)

    for filename, ids in (("test_query.csv", QUERY_IDS), ("test_gallery.csv", GALLERY_IDS)):
        with (data / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image_id", "x", "y", "w", "h"))
            writer.writerows((image_id, 15, 30, 130, 80) for image_id in ids)
    print(f"[smoke] created {len(QUERY_IDS)} queries and {len(GALLERY_IDS)} gallery images")


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.reader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(data: Path, out: Path, release: Path) -> None:
    for name in ("embeddings.npy", "submission.csv", "candidates.csv", "run_info.json"):
        assert (out / name).is_file(), f"missing {name}"

    query_ids = [row[0] for row in _rows(data / "test_query.csv")[1:]]
    gallery_ids = [row[0] for row in _rows(data / "test_gallery.csv")[1:]]
    assert query_ids == list(QUERY_IDS) and gallery_ids == list(GALLERY_IDS)

    embeddings = np.load(out / "embeddings.npy", allow_pickle=False)
    assert embeddings.ndim == 2 and embeddings.shape[0] == len(query_ids) + len(gallery_ids)
    assert embeddings.shape[1] > 0 and embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)

    ranking = _rows(out / "submission.csv")
    assert len(ranking) == len(query_ids)
    for expected_query, row in zip(query_ids, ranking):
        assert row[0] == expected_query and len(row) == 11
        assert len(set(row[1:])) == 10 and set(row[1:]) == set(gallery_ids)

    candidates = _rows(out / "candidates.csv")
    assert candidates and candidates[0] == ["query_id", "gallery_id", "confidence"]
    candidate_rows = {}
    for row in candidates[1:]:
        assert len(row) == 3 and row[0] in query_ids and row[1] in gallery_ids
        assert row[0] not in candidate_rows, "only one top-confidence row per query"
        candidate_rows[row[0]] = row
        confidence = float(row[2])
        assert np.isfinite(confidence) and -1 <= confidence <= 1
        selected = next(r for r in ranking if r[0] == row[0])[1]
        assert row[1] == selected, "candidate must be the ranked top-1"

    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["n_query"] == len(query_ids) and info["n_gallery"] == len(gallery_ids)
    assert info["dim"] == embeddings.shape[1]
    assert info["refused"] == len(query_ids) - len(candidate_rows)
    assert info["model_sha256"] == _sha256(release / "model.pt")
    assert info["recipe_sha256"] == _sha256(release / "recipe.json")
    assert info["input_csv_sha256"] == {
        name: _sha256(data / name) for name in ("test_query.csv", "test_gallery.csv")
    }
    threshold = float(info["threshold_used"])
    for i, query_id in enumerate(query_ids):
        top_gallery_id = ranking[i][1]
        j = gallery_ids.index(top_gallery_id)
        cosine = float(embeddings[i] @ embeddings[len(query_ids) + j])
        expected_accept = cosine >= threshold
        assert (query_id in candidate_rows) == expected_accept
        if expected_accept:
            assert abs(float(candidate_rows[query_id][2]) - cosine) <= 5e-5
    print("[smoke] real-model offline inference and submission contract passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "check"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--release", type=Path, default=Path("/app/release"))
    args = parser.parse_args()
    if args.action == "create":
        create(args.data)
    else:
        if args.out is None:
            parser.error("check requires --out")
        check(args.data, args.out, args.release)


if __name__ == "__main__":
    main()
