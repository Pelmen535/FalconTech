import numpy as np
import pytest

from vreid.compress import Codec, evaluate_variants


def _synthetic(n_ids=80, cams=3, D=256, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_ids, D))
    emb, vids, cams_ = [], [], []
    for v in range(n_ids):
        for c in range(cams):
            e = centers[v] + noise * rng.normal(size=D)
            emb.append(e / np.linalg.norm(e)); vids.append(v); cams_.append(c)
    emb = np.array(emb, dtype=np.float32)
    vids, cams_ = np.array(vids), np.array(cams_)
    q = cams_ == 0
    return emb[q], emb[~q], vids[q], cams_[q], vids[~q], cams_[~q]


def test_bytes_per_vector():
    assert Codec("f32").bytes_per_vector(768) == 3072
    assert Codec("f16").bytes_per_vector(768) == 1536
    assert Codec("pca128").bytes_per_vector(768) == 256
    assert Codec("pca128_i8").bytes_per_vector(768) == 128
    assert Codec("pca256_bin").bytes_per_vector(768) == 32
    assert Codec("pq16").bytes_per_vector(768) == 16


def test_roundtrip_shapes():
    q, g, *_ = _synthetic()
    for name in ["f16", "pca64", "pcaw64", "pca64_i8", "pca64_bin"]:
        c = Codec(name).fit(g)
        codes = c.encode(q)
        assert codes.shape[0] == q.shape[0]
        sims = c.similarity(codes, c.encode(g))
        assert sims.shape == (q.shape[0], g.shape[0])
        assert np.isfinite(sims).all()


def test_int8_close_to_float():
    q, g, *_ = _synthetic()
    a = Codec("pca64").fit(g)
    b = Codec("pca64_i8").fit(g)
    sa = a.similarity(a.encode(q), a.encode(g))
    sb = b.similarity(b.encode(q), b.encode(g))
    assert np.abs(sa - sb).max() < 0.05


def test_quality_curve_monotone_enough():
    q, g, qv, qc, gv, gc = _synthetic()
    rows = {r["variant"]: r for r in evaluate_variants(q, g, qv, qc, gv, gc,
                                                        variants=["f32", "pca64", "pca64_i8", "pca64_bin"])}
    assert rows["f32"]["rank1"] > 0.9
    assert rows["pca64_i8"]["rank1"] >= rows["f32"]["rank1"] - 0.1
    assert rows["pca64_bin"]["bytes"] == 8
