# -*- coding: utf-8 -*-
"""Регрессия на дефекты R01/R02 из внешнего аудита 19.09.

R01: нестабильная сортировка соседей в k-reciprocal — при точно равных расстояниях
     порядок решал introsort, а он зависит от раскладки матрицы, то есть от числа
     остальных запросов. На реальных данных ничьих нет, поэтому прежние проверки
     на 666 запросах это не ловили.
R02: гейт памяти в predict.py выключал ре-ранжирование целиком, а порог считался
     от (Q+G) — значит чужие запросы могли переключить алгоритм для нашего.
"""
import numpy as np

from vreid.rerank import k_reciprocal, k_reciprocal_chunked

K1, K2, LAM = 6, 2, 0.3


def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _tie_case(seed=0, G=20, D=16):
    rng = np.random.default_rng(seed)
    g = _norm(rng.normal(size=(G, D)).astype(np.float32))
    for i in (1, 2, 3, 10):
        g[i] = g[0]                       # точные дубликаты в галерее
    q = _norm((g[0] + 0.01 * rng.normal(size=(1, D))).astype(np.float32))
    other = _norm(rng.normal(size=(15, D)).astype(np.float32))
    return q, other, g


def test_ties_do_not_depend_on_other_queries():
    q, other, g = _tie_case()
    alone = k_reciprocal(q, g, K1, K2, LAM)
    crowd = k_reciprocal(np.concatenate([q, other]), g, K1, K2, LAM)
    assert np.abs(alone[0] - crowd[0]).max() < 1e-6


def test_permuting_other_queries_changes_nothing():
    q, other, g = _tie_case()
    rng = np.random.default_rng(1)
    a = k_reciprocal(np.concatenate([q, other]), g, K1, K2, LAM)
    b = k_reciprocal(np.concatenate([q, other[rng.permutation(len(other))]]), g, K1, K2, LAM)
    assert np.abs(a[0] - b[0]).max() < 1e-6


def test_chunking_is_exact():
    q, other, g = _tie_case()
    Q = np.concatenate([q, other])
    full = k_reciprocal(Q, g, K1, K2, LAM)
    for c in (1, 3, 7, 16):
        assert np.abs(full - k_reciprocal_chunked(Q, g, K1, K2, LAM, chunk=c)).max() < 1e-6


def test_k1_clamp_uses_gallery_not_total():
    """k1 упирался в N-1 = Q+G-1: потолок ехал вместе с числом запросов."""
    rng = np.random.default_rng(2)
    g = _norm(rng.normal(size=(4, 8)).astype(np.float32))       # галерея меньше k1
    q = _norm(rng.normal(size=(1, 8)).astype(np.float32))
    other = _norm(rng.normal(size=(30, 8)).astype(np.float32))
    a = k_reciprocal(q, g, 10, 3, LAM)
    b = k_reciprocal(np.concatenate([q, other]), g, 10, 3, LAM)
    assert np.abs(a[0] - b[0]).max() < 1e-6


def test_nan_confidence_is_refused(tmp_path):
    """NaN < порог — ложь по IEEE, поэтому раньше сбой проходил за уверенный ответ."""
    from vreid.submit import write_candidates
    sims = np.array([[0.9, 0.1], [np.nan, np.nan]])
    st = write_candidates(["q1", "q2"], ["g0", "g1"], sims, np.array([0.9, np.nan]), 0.5,
                          tmp_path / "c.csv", conf_sims=sims, exclude_self=False)
    assert st["n_refused"] == 1 and st["n_nonfinite"] == 1
    body = (tmp_path / "c.csv").read_text(encoding="utf-8")
    assert "nan" not in body.lower()
