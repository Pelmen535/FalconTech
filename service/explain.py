"""Интерпретируемость: какие участки кадра дали косинус между запросом и кандидатом.

ТЗ §10 просит визуализацию областей, «наиболее сильно повлиявших на решение модели».
Обычный ответ — Grad-CAM: градиент скалярной цели по карте активаций. Здесь он не нужен,
потому что у нашей архитектуры вклад патча в косинус выписывается ТОЧНО, без градиентов
и без приближений.

Почему так. Эмбеддинг собирается как

    f  = concat( CLS , mean_p token_p )            (768 + 768 = 1536)
    b  = BNNeck(f) = a * f + c,   a = gamma/sqrt(var+eps),  c = beta - a*mu
    z  = b / ||b||

BNNeck в eval — это поэлементная аффинная функция, а усреднение по патчам линейно.
Значит для косинуса z_q · z_g можно раскрыть половину, отвечающую за патчи:

    z_q · z_g = ( Σ_p w(p) + const ) ,
    w(p) = (1/||b_q||) * (1/P) * Σ_j a_[768+j] * token_p,j * z_g,[768+j]

w(p) — это ровно столько косинуса, сколько добавил патч p, в тех же единицах.
const — вклад CLS-половины и свободных членов BNNeck, он от патчей не зависит.
Сумма частей равна целому: проверяется в тестах (test_explain.py) и в самом ответе API
полем `reconstruction_error`.

Карта считается симметрично для запроса и для кандидата: «куда смотрел запрос» и
«чем ответил кандидат» — разные вопросы, и оператору полезны оба.
"""
from __future__ import annotations

import numpy as np


class ExplainUnavailable(RuntimeError):
    """Бэкбон не ViT с BNNeck — точное разложение неприменимо."""


def _grid(n_patches: int) -> tuple[int, int]:
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise ExplainUnavailable("число патчей не квадрат — сетку не восстановить")
    return side, side


def patch_contributions(model, batch, partner_unit: np.ndarray, half: bool = False) -> dict:
    """Вклад каждого патча одного изображения в косинус с уже нормированным вектором партнёра.

    model  — vreid.train.ReIDModel (backbone + bnneck)
    batch  — тензор [1, 3, H, W], уже прошедший препроцессинг бэкбона
    partner_unit — L2-нормированный эмбеддинг второй картинки, np.float32 [D]
    half   — считать бэкбон под autocast, как в боевом пути: иначе косинус в объяснении
             разойдётся с косинусом в выдаче поиска в четвёртом знаке

    Возвращает карту вкладов [gh, gw], постоянную часть и сам косинус.
    """
    import torch

    backbone = getattr(model, "backbone", None)
    bnneck = getattr(model, "bnneck", None)
    if backbone is None or bnneck is None or not getattr(model, "is_vit", False):
        raise ExplainUnavailable("точное разложение определено только для ViT + BNNeck")

    device = next(backbone.parameters()).device
    partner = torch.as_tensor(np.asarray(partner_unit, dtype=np.float32), device=device)
    if partner.ndim != 1 or partner.shape[0] != int(model.dim):
        raise ValueError("partner_unit должен быть вектором размерности модели")

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=bool(half)):
        tokens = backbone.forward_features(batch.to(device))
    tokens = tokens.float()                                            # [1, 1+P, C]
    with torch.no_grad():
        n_prefix = int(getattr(backbone, "num_prefix_tokens", 1))
        cls = tokens[:, 0]                                             # [1, C]
        patch_tokens = tokens[:, n_prefix:]                            # [1, P, C]
        n_patches = patch_tokens.shape[1]
        feature = torch.cat([cls, patch_tokens.mean(dim=1)], dim=1)    # [1, 2C]

        # BNNeck в eval: b = a*f + c. Берём его параметры напрямую, а не через forward,
        # чтобы разложение опиралось на ту же арифметику, что и боевой путь.
        eps = float(getattr(bnneck, "eps", 1e-5))
        var = bnneck.running_var.float()
        scale = (bnneck.weight.float() if bnneck.weight is not None
                 else torch.ones_like(var)) / torch.sqrt(var + eps)
        shift = (bnneck.bias.float() if bnneck.bias is not None
                 else torch.zeros_like(var)) - scale * bnneck.running_mean.float()
        b = scale * feature[0] + shift
        norm = torch.linalg.vector_norm(b)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise ExplainUnavailable("нулевой или нечисловой эмбеддинг")
        own_unit = b / norm
        cosine = float(torch.dot(own_unit, partner))

        channels = cls.shape[1]
        if scale.shape[0] != 2 * channels:
            raise ExplainUnavailable("BNNeck не соответствует схеме concat(CLS, mean patch)")
        patch_scale = scale[channels:] * partner[channels:]            # [C]
        # w(p) = (1/||b||) * (1/P) * Σ_j a_j * token_p,j * z_partner_j
        weights = (patch_tokens[0] @ patch_scale) / (norm * n_patches)  # [P]
        # Всё, что не зависит от патчей: CLS-половина и свободные члены обеих половин.
        constant = cosine - float(weights.sum())

    grid_h, grid_w = _grid(n_patches)
    contributions = weights.detach().cpu().numpy().astype(np.float32).reshape(grid_h, grid_w)
    return {
        "contributions": contributions,
        "constant": float(constant),
        "cosine": cosine,
        "grid": [grid_h, grid_w],
        "reconstruction_error": float(abs(contributions.sum() + constant - cosine)),
    }


def heatmap_png(crop, contributions: np.ndarray, alpha: float = 0.6,
                center: bool = True, percentile: float = 98.0) -> bytes:
    """Наложить карту вкладов на кроп и вернуть PNG.

    center=True показывает не сам вклад, а отклонение от среднего по патчам. Так надо:
    у похожей пары почти все патчи вносят положительный вклад, и картинка «всё зелёное»
    не говорит ничего. Оператору нужен ответ «чем именно этот кроп отличается от ровного
    фона согласия» — то есть где сходство выше и ниже типичного. Сами значения в JSON
    остаются сырыми и в сумме дают косинус.

    Шкала симметричная и обрезана по перцентилю: одна выбивающаяся клетка не должна
    схлопывать всю остальную картинку в серое.
    """
    from io import BytesIO
    from PIL import Image

    base = crop.convert("RGB")
    values = np.asarray(contributions, dtype=np.float32)
    if center:
        values = values - float(values.mean())
    limit = float(np.percentile(np.abs(values), percentile)) if values.size else 0.0
    scaled = np.clip(values / limit, -1, 1) if limit > 0 else np.zeros_like(values)

    positive = np.clip(scaled, 0, 1)
    negative = np.clip(-scaled, 0, 1)
    # Зелёный — вклад выше типичного, красный — ниже, серый — как у всех.
    red = 0.5 + 0.5 * negative - 0.25 * positive
    green = 0.5 + 0.5 * positive - 0.25 * negative
    blue = 0.5 - 0.35 * (positive + negative)
    colors = np.clip(np.stack([red, green, blue], axis=-1), 0, 1) * 255

    overlay = Image.fromarray(colors.astype(np.uint8)).resize(base.size, Image.BICUBIC)
    # Непрозрачность пропорциональна модулю отклонения: нейтральные зоны не закрашиваются.
    strength = Image.fromarray((np.abs(scaled) * (alpha * 255)).astype(np.uint8)
                               ).resize(base.size, Image.BICUBIC)
    blended = Image.composite(overlay, base, strength)
    buffer = BytesIO()
    blended.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def top_regions(contributions: np.ndarray, count: int = 3) -> list[dict]:
    """Клетки с наибольшим отклонением вверх — чтобы ответ был не только картинкой.

    Считается по тому же центрированному значению, что и картинка, иначе список
    расходился бы с тем, что видно на экране."""
    values = np.asarray(contributions, dtype=np.float32)
    values = values - float(values.mean())
    grid_h, grid_w = values.shape
    order = np.argsort(-values, axis=None)[:max(0, int(count))]
    regions = []
    for flat in order:
        row, column = divmod(int(flat), grid_w)
        regions.append({
            "contribution": float(values[row, column]),
            "box_fraction": [round(column / grid_w, 4), round(row / grid_h, 4),
                             round(1 / grid_w, 4), round(1 / grid_h, 4)],
        })
    return regions
