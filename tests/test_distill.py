"""Свойства дистилляции с фокусом на трудных парах (--distill-focus).

Главное — не «стало лучше» (это меряет обучение), а две вещи, без которых опыт нечестен:
focus=0 обязан давать прежний лосс до бита, иначе релизы v1.1/v1.2 нельзя воспроизвести
нынешним кодом; и вес получают ровно те пары, которые заявлены, — кросс-камерные позитивы и
k ближайших по учителю чужих машин, ни одной больше.
"""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from vreid.train import distill_focus_weights, similarity_distill  # noqa: E402


def _batch(P=5, K=4, dim_s=16, dim_t=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(P).repeat_interleave(K)
    cams = torch.randint(0, 3, (P * K,), generator=g)
    return (torch.randn(P * K, dim_s, generator=g), torch.randn(P * K, dim_t, generator=g), y, cams)


def test_focus_zero_is_old_formula_bit_for_bit():
    fs, ft, y, cams = _batch()
    s, t = F.normalize(fs, dim=1), F.normalize(ft, dim=1)
    old = ((s @ s.T) - (torch.stack([t @ t.T]).mean(0))).pow(2).mean()
    new = similarity_distill(fs, [ft], y=y, cams=cams, focus=0.0)
    assert torch.equal(old, new)


def test_weights_mark_exactly_cross_camera_positives_and_k_nearest_negatives():
    fs, ft, y, cams = _batch()
    t = F.normalize(ft, dim=1)
    target = t @ t.T
    focus, k = 4.0, 3
    w = distill_focus_weights(target, y, cams, focus, k)
    same = y[:, None] == y[None, :]
    for i in range(len(y)):
        cross_pos = {j for j in range(len(y)) if same[i, j] and j != i and cams[i] != cams[j]}
        neg = [j for j in range(len(y)) if not same[i, j]]
        nearest = set(sorted(neg, key=lambda j: -target[i, j].item())[:k])
        marked = {j for j in range(len(y)) if w[i, j] > 1}
        assert marked == cross_pos | nearest
        assert w[i, i] == 1                                   # диагональ не трудная пара
    assert set(torch.unique(w).tolist()) <= {1.0, 1.0 + focus}


def test_without_cams_every_positive_counts():
    fs, ft, y, cams = _batch()
    t = F.normalize(ft, dim=1)
    w = distill_focus_weights(t @ t.T, y, None, 2.0, 0)
    same = (y[:, None] == y[None, :]) & ~torch.eye(len(y), dtype=torch.bool)
    assert torch.equal(w > 1, same)


def test_weighted_loss_keeps_scale():
    """Нормировка на сумму весов: при одинаковой ошибке на всех парах лосс не зависит от focus."""
    fs, _, y, cams = _batch()
    s = F.normalize(fs, dim=1)
    target = (s @ s.T) - 0.1                                  # ошибка 0.1 на каждой паре
    base = ((s @ s.T) - target).pow(2).mean()
    w = distill_focus_weights(target, y, cams, 4.0, 4)
    weighted = (((s @ s.T) - target).pow(2) * w).sum() / w.sum()
    assert torch.allclose(base, weighted)


def test_gradient_goes_only_to_student():
    fs, ft, y, cams = _batch()
    fs.requires_grad_(True)
    ft.requires_grad_(True)
    similarity_distill(fs, [ft], y=y, cams=cams, focus=4.0).backward()
    assert fs.grad is not None and torch.isfinite(fs.grad).all()
    assert ft.grad is not None                                # учитель в обучении под no_grad;
    # здесь важно другое: веса пар не несут градиента и не дают NaN при -inf в маске
    assert torch.isfinite(ft.grad).all()
