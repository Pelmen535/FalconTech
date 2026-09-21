import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vreid.hackathon_data import make_query_gallery, read_annotations, split_open_set
from vreid.datasets import Record
from vreid.refusal import evaluate_refusal, minp, pr_auc, refusal_curve, choose_threshold


def _recs(n_ids=20, cams=3):
    out = []
    for v in range(n_ids):
        for c in range(cams):
            out.append(Record(f"/img/{v}_{c}.jpg", v, c + 1, (0, 0, 10, 10), f"{v}_{c}#0"))
    return out


def test_open_set_split_disjoint():
    fit, val = split_open_set(_recs(), val_frac=0.25, seed=1)
    assert {r.vid for r in fit}.isdisjoint({r.vid for r in val})
    assert len(fit) + len(val) == 60
    assert len({r.vid for r in val}) == 5


def test_query_gallery_distractors_and_cross_cam():
    _, val = split_open_set(_recs(n_ids=40), val_frac=0.5, seed=0)
    q, g, has = make_query_gallery(val, distractor_frac=0.2, queries_per_id=1, seed=0)
    g_ids = {r.vid for r in g.records}
    for r, h in zip(q.records, has):
        assert (r.vid in g_ids) == h
        if h:  # у запроса с парой есть кадр той же машины с другой камеры
            assert any(o.vid == r.vid and o.cam != r.cam for o in g.records)
    assert (~has).sum() == 4


def test_refusal_curve_perfect_separation():
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    correct = np.array([True, True, True, False, False])
    has = np.array([True, True, True, False, False])
    best = choose_threshold(refusal_curve(conf, correct, has))
    assert best["f1"] == 1.0 and best["tnr"] == 1.0
    assert pr_auc(conf, has & correct) == 1.0


def test_minp_hand():
    # запрос 0: совпадения на позициях 1 и 3 из 4 → last=3, m=2 → INP = 1 - (3-2)/3 = 2/3
    sims = np.array([[0.9, 0.8, 0.7, 0.6]])
    g_vids = np.array([1, 2, 1, 3])
    order = np.argsort(-sims, axis=1)
    assert abs(minp(order, sims, np.array([1]), g_vids, np.array([True])) - 2 / 3) < 1e-9


def test_evaluate_refusal_runs():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(30, 8)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    g = np.concatenate([q[:20] + 0.1 * rng.normal(size=(20, 8)), rng.normal(size=(40, 8))])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    q_vids = np.arange(30); g_vids = np.concatenate([np.arange(20), 1000 + np.arange(40)])
    has = q_vids < 20
    res = evaluate_refusal(q @ g.T, q_vids, np.zeros(30, int), g_vids, np.ones(60, int), has, tau=0.9)
    assert res["best"]["f1"] > 0.9 and res["n_no_match"] == 10
    assert 0 <= res["mINP"] <= 1


def test_read_annotations_bbox_formats(tmp_path):
    (tmp_path / "images").mkdir()
    p = tmp_path / "a.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "x", "y", "w", "h", "vehicle_id"])
        w.writerow(["img1", 10, 20, 30, 40, "A"]); w.writerow(["img1", 5, 5, 6, 6, "B"]); w.writerow(["img2", 0, 0, 1, 1, "A"])
    recs = read_annotations(p, tmp_path / "images")
    assert recs[0].bbox == (10, 20, 30, 40) and recs[0].vid == recs[2].vid != recs[1].vid
    assert recs[0].key == "img1#0" and recs[1].key == "img1#1"


@pytest.mark.slow
def test_end_to_end_toy_hack(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    d = tmp_path / "toy_hack"
    # на Windows консоль дочернего процесса — cp1251; скрипты печатают юникод, поэтому включаем UTF-8
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    run = lambda *args: subprocess.run([sys.executable, *args], check=True, cwd=root, env=env)
    run(str(root / "scripts/make_toy_hackathon.py"), "--root", str(d), "--ids", "40", "--cams", "4")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"name: t\ndataset: {{root: '{d.as_posix()}', val_frac: 0.3, distractor_frac: 0.2, pseudo_camera: cluster}}\n"
                   f"runs_dir: '{(tmp_path / 'runs').as_posix()}'\nresults_dir: '{(tmp_path / 'results').as_posix()}'\n", encoding="utf-8")
    run("-m", "vreid.hack_cli", "val", "--config", str(cfg), "--backbone", "dummy", "--workers", "0")
    run("-m", "vreid.hack_cli", "submit", "--config", str(cfg), "--backbone", "dummy", "--workers", "0", "--out", str(tmp_path / "sub"))
    emb = np.load(tmp_path / "sub" / "embeddings.npy")
    n_test = sum(1 for _ in open(d / "test.csv")) - 1
    assert emb.shape[0] == n_test and emb.dtype == np.float32
    # Официальный формат submission.csv — БЕЗ строки заголовка: так написано в шапке
    # organizer/evaluate.py и так устроен organizer/example_submission.zip. Строка на
    # каждый запрос, в строке query_id плюс ровно десять идентификаторов галереи.
    rows = list(csv.reader(open(tmp_path / "sub" / "submission.csv", encoding="utf-8")))
    assert rows[0][0] != "query_id", "в submission.csv не должно быть строки заголовка"
    assert len(rows) == n_test and all(len(r) == 11 for r in rows)
