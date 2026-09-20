import numpy as np

from vreid.metrics import evaluate
from vreid.rerank import aqe, dba, frame_block_mask, k_reciprocal
from vreid.local_match import fuse, match_pair, local_scores


def _data(n_ids=40, views=3, D=32, noise=0.9, seed=0):
    """Шумные эмбеддинги: у каждой машины views наблюдений; шум большой, чтобы было что улучшать."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n_ids, D))
    emb, vids, cams = [], [], []
    for v in range(n_ids):
        for k in range(views):
            e = c[v] + noise * rng.normal(size=D)
            emb.append(e / np.linalg.norm(e)); vids.append(v); cams.append(k)
    emb, vids, cams = np.array(emb, np.float32), np.array(vids), np.array(cams)
    q = cams == 0
    return emb[q], emb[~q], vids[q], cams[q], vids[~q], cams[~q]


def test_k_reciprocal_improves_or_holds_map():
    q, g, qv, qc, gv, gc = _data()
    base = evaluate(q @ g.T, qv, qc, gv, gc, cross_camera_only=True)["mAP"]
    d = k_reciprocal(q, g, k1=8, k2=3, lam=0.3)
    rr = evaluate(-d, qv, qc, gv, gc, cross_camera_only=True)["mAP"]
    assert d.shape == (len(q), len(g))
    assert rr >= base - 0.02


def test_dba_refined_vectors_help():
    q, g, qv, qc, gv, gc = _data()
    base = evaluate(q @ g.T, qv, qc, gv, gc, cross_camera_only=True)["mAP"]
    allv = np.concatenate([q, g])
    ref = dba(allv, k=2)
    q2, g2 = ref[: len(q)], ref[len(q):]
    after = evaluate(q2 @ g2.T, qv, qc, gv, gc, cross_camera_only=True)["mAP"]
    assert np.allclose(np.linalg.norm(ref, axis=1), 1.0, atol=1e-5)
    assert after > base


def test_aqe_shape_and_norm():
    q, g, *_ = _data()
    q2 = aqe(q, g, k=2)
    assert q2.shape == q.shape and np.allclose(np.linalg.norm(q2, axis=1), 1.0, atol=1e-5)


def test_frame_block_mask():
    m = frame_block_mask(["a#0", "b#1"], ["a#3", "c#0", "b#0"])
    assert m.tolist() == [[True, False, False], [False, False, True]]


def test_match_pair_detects_shift():
    rng = np.random.default_rng(0)
    h = w = 8
    fa = rng.normal(size=(h * w, 16)); fa /= np.linalg.norm(fa, axis=1, keepdims=True)
    # fb = fa, сдвинутый на (dx=1, dy=0) по сетке; за краем — случайные патчи
    fb = rng.normal(size=(h * w, 16)); fb /= np.linalg.norm(fb, axis=1, keepdims=True)
    for y in range(h):
        for x in range(w - 1):
            fb[y * w + x + 1] = fa[y * w + x]
    r = match_pair(fa, fb, (h, w), min_sim=0.9, tol=0.5)
    assert r["n_inliers"] >= h * (w - 1) - 2
    assert abs(r["shift"][0] - 1) < 1e-6 and abs(r["shift"][1]) < 1e-6
    # несвязанные признаки — почти без совпадений
    fc = rng.normal(size=(h * w, 16)); fc /= np.linalg.norm(fc, axis=1, keepdims=True)
    assert match_pair(fa, fc, (h, w), min_sim=0.9)["n_inliers"] <= 3


def test_fuse_reorders_topk():
    sims = np.array([[0.9, 0.8, 0.1]])
    order = np.argsort(-sims, axis=1)
    local = np.array([[0.0, 10.0]])       # у второго кандидата 10 совпадений из T=20
    new = fuse(sims, order, local, beta=1.0, T=20)
    assert new[0, 1] > new[0, 0] and new[0, 2] == 0.1
    ls = local_scores(np.zeros((1, 4, 3)), np.zeros((3, 4, 3)), order, (2, 2), topk=2, verbose=False)
    assert ls.shape == (1, 2)
