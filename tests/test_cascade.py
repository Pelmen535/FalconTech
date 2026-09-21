"""Свойства каскада, на которых держится его допустимость и воспроизводимость.

Главное, что здесь проверяется, — не «стало лучше», а «решение по запросу зависит только
от него самого» (ответ 38) и «alpha=0 не меняет ничего». Первое делает каскад законным,
второе делает измерение прироста осмысленным: если alpha=0 уже двигает порядок, значит
сравнивать «с каскадом» и «без каскада» нечестно.
"""
import numpy as np
import pytest

from vreid.cascade import heavy_scores, rescore, shortlist


def _ranking(seed=0, n_query=12, n_gallery=40):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_query, n_gallery)).astype(np.float32)
    heavy_q = rng.normal(size=(n_query, 16)).astype(np.float32)
    heavy_g = rng.normal(size=(n_gallery, 16)).astype(np.float32)
    return base, heavy_q, heavy_g


def test_alpha_zero_keeps_order():
    """alpha=0 обязан воспроизводить базовый порядок — иначе прирост каскада не измерить."""
    base, hq, hg = _ranking()
    order = shortlist(base, 10)
    heavy = heavy_scores(hq, hg, order)
    out = rescore(base, order, heavy, 0.0)
    assert np.array_equal(np.argsort(-out, axis=1, kind="stable"),
                          np.argsort(-base, axis=1, kind="stable"))


def test_shortlist_stays_on_top():
    """Как бы тяжёлая модель ни переставила кандидатов, шортлист занимает верх ранжирования
    целиком: каскад уточняет порядок внутри топ-K, а не выкидывает кандидатов в хвост."""
    base, hq, hg = _ranking()
    order = shortlist(base, 7)
    heavy = heavy_scores(hq, hg, order)
    out = rescore(base, order, heavy, 1.0)
    top = np.argsort(-out, axis=1, kind="stable")[:, :7]
    for i in range(base.shape[0]):
        assert set(top[i].tolist()) == set(order[i].tolist())


def test_query_independence():
    """Строка результата не меняется от того, какие ещё запросы пришли в пачке.
    Это ровно то, что запрещает ответ 38, и проверять это надо кодом, а не обещанием."""
    base, hq, hg = _ranking(seed=3, n_query=9)
    order = shortlist(base, 6)
    heavy = heavy_scores(hq, hg, order)
    full = rescore(base, order, heavy, 0.6)
    for subset in ([0], [2, 5], [1, 3, 8]):
        part_order = shortlist(base[subset], 6)
        part = rescore(base[subset], part_order, heavy_scores(hq[subset], hg, part_order), 0.6)
        assert np.allclose(part, full[subset], equal_nan=True)


def test_blocked_candidates_never_enter():
    """Запрещённые пары (-inf, кадр той же камеры) не попадают ни в шортлист, ни наверх."""
    base, hq, hg = _ranking(seed=1)
    base[:, :5] = -np.inf
    order = shortlist(base, 8)
    assert not np.isin(order, np.arange(5)).any()
    out = rescore(base, order, heavy_scores(hq, hg, order), 0.9)
    assert np.isneginf(out[:, :5]).all()


def test_short_gallery_marks_missing_with_minus_one():
    """Если валидных кандидатов меньше K, лишние позиции помечаются -1, а не случайным
    индексом: иначе в выдачу попал бы запрещённый кадр."""
    base = np.full((3, 6), -np.inf, dtype=np.float32)
    base[:, :2] = np.array([[0.9, 0.1], [0.5, 0.4], [0.2, 0.7]], dtype=np.float32)
    order = shortlist(base, 5)
    assert (order[:, 2:] == -1).all()
    assert (order[:, :2] >= 0).all()
    heavy = heavy_scores(*_ranking(seed=2, n_query=3, n_gallery=6)[1:], order)
    assert np.isneginf(heavy[:, 2:]).all()
    out = rescore(base, order, heavy, 0.8)
    assert np.isneginf(out[:, 2:]).all() or np.isfinite(out[np.arange(3), order[:, 0]]).all()


def test_alpha_out_of_range_rejected():
    base, hq, hg = _ranking()
    order = shortlist(base, 5)
    heavy = heavy_scores(hq, hg, order)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            rescore(base, order, heavy, bad)


def test_rescore_does_not_mutate_input():
    """Вызывающая сторона сравнивает «до» и «после» — входная матрица должна пережить вызов."""
    base, hq, hg = _ranking()
    before = base.copy()
    order = shortlist(base, 9)
    rescore(base, order, heavy_scores(hq, hg, order), 0.5)
    assert np.array_equal(base, before)
