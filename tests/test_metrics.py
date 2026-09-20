import numpy as np

from vreid.metrics import evaluate
from vreid.bits import bits_from_count, calibrate_tau, bits_joint
from vreid.index import GalleryIndex


def test_perfect_ranking():
    # 3 запроса, галерея: для каждого id по 2 кадра с других камер
    g_vids = np.array([1, 1, 2, 2, 3, 3])
    g_cams = np.array([2, 3, 2, 3, 2, 3])
    q_vids = np.array([1, 2, 3])
    q_cams = np.array([1, 1, 1])
    sims = np.zeros((3, 6))
    for i, v in enumerate(q_vids):
        sims[i, g_vids == v] = 1.0
    m = evaluate(sims, q_vids, q_cams, g_vids, g_cams)
    assert m["rank1"] == 1.0 and m["mAP"] == 1.0 and m["n_query_valid"] == 3


def test_ap_hand_computed():
    # один запрос, галерея из 4: совпадения на позициях 1 и 3 (ранжируем по sims)
    g_vids = np.array([7, 9, 7, 9])
    g_cams = np.array([2, 2, 2, 2])
    sims = np.array([[0.9, 0.8, 0.7, 0.6]])
    m = evaluate(sims, np.array([7]), np.array([1]), g_vids, g_cams, ranks=(1, 2))
    # AP = mean(1/1, 2/3) = 0.8333
    assert abs(m["mAP"] - (1.0 + 2 / 3) / 2) < 1e-9
    assert m["rank1"] == 1.0 and m["rank2"] == 1.0


def test_same_camera_excluded():
    # единственное совпадение — с той же камеры → запрос невалиден, а второй запрос валиден
    g_vids = np.array([1, 2])
    g_cams = np.array([1, 5])
    sims = np.array([[1.0, 0.0], [0.0, 1.0]])
    m = evaluate(sims, np.array([1, 2]), np.array([1, 1]), g_vids, g_cams)
    assert m["n_query_valid"] == 1 and m["rank1"] == 1.0


def test_cross_camera_only_harder():
    # совпадение с той же камеры (другой id не мешает) — в cross-cam режиме выкидывается
    g_vids = np.array([1, 1, 2])
    g_cams = np.array([1, 2, 1])
    sims = np.array([[0.95, 0.5, 0.9]])   # тот же id с той же камеры ранжирован первым
    std = evaluate(sims, np.array([1]), np.array([1]), g_vids, g_cams)
    cc = evaluate(sims, np.array([1]), np.array([1]), g_vids, g_cams, cross_camera_only=True)
    # standard: выкинут только (id1,cam1); остаются [id2:0.9, id1:0.5] → rank1=0, AP=1/2
    assert std["rank1"] == 0.0 and abs(std["mAP"] - 0.5) < 1e-9
    # cross-cam: выкинуты все кадры с камеры 1; остаётся [id1:0.5] → rank1=1
    assert cc["rank1"] == 1.0 and cc["mAP"] == 1.0


def test_bits_formula():
    assert bits_from_count(1, 1024) == 10.0
    assert bits_from_count(1024, 1024) == 0.0
    assert bits_from_count(0, 1024) == 10.0  # ноль кандидатов не даёт бесконечность


def test_calibrate_and_joint():
    rng = np.random.default_rng(0)
    # 50 машин × 3 камеры, эмбеддинги = центр машины + шум
    centers = rng.normal(size=(50, 16))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    g_emb, g_vids, g_cams = [], [], []
    for v in range(50):
        for c in range(3):
            e = centers[v] + 0.15 * rng.normal(size=16)
            g_emb.append(e / np.linalg.norm(e)); g_vids.append(v); g_cams.append(c)
    g_emb, g_vids, g_cams = np.array(g_emb, dtype=np.float32), np.array(g_vids), np.array(g_cams)
    q = centers[:10] + 0.15 * rng.normal(size=(10, 16))
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    index = GalleryIndex(g_emb, g_vids, g_cams)
    sims = index.all_sims(q)
    cal = calibrate_tau(sims, np.arange(10), np.full(10, 99), g_vids, g_cams, recall=0.9)
    assert 0 < cal["tau"] < 1
    # два наблюдения одной машины: совместные биты ≥ битов одного наблюдения
    single = bits_joint(index, q[:1], cal["tau"])
    joint = bits_joint(index, np.stack([q[0], q[0] + 0.05 * rng.normal(size=16).astype(np.float32)]), cal["tau"])
    assert joint["bits"] >= single["bits"] - 1e-9
    assert joint["n_cand"] <= single["n_cand"]
