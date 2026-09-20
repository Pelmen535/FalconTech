"""Поиск зоны анонимизации номера на кропе ТС.

ЗАЧЕМ. Кейс — реидентификация БЕЗ распознавания номера. Проверять это фиксированным
прямоугольником «низ по центру» нельзя: при косом ракурсе он мимо пластины, зато закрывает
бампер и низ решётки, и падение качества уже не о номере. Нужно закрывать ровно номер.

ЧТО В ДАННЫХ. Номера уже анонимизированы организаторами — пикселизацией. Замер на реальном
кропе (1278x1080): строка пикселей поперёк номера идёт ступенями «23 23 ... 23 | 65 65 ... 65»
с шагом ~74 px, а связные области постоянной яркости там — идеальные прямоугольники 72x15 px
с заполнением bbox 1.00, выстроенные решёткой (шаг 74 по x, 17 по y). На номер шириной 320 px
приходится 4-5 ячеек, то есть от самих символов не осталось ничего.

КАК ИЩЕМ. Плоские связные компоненты, похожие на прямоугольник, у которых есть соседи ТАКОГО
ЖЕ размера. Гладкая краска тоже плоская, но даёт одну большую кляксу неправильной формы без
повторяющихся соседей; текстура и решётка радиатора плоскими компонентами не являются вовсе.
Проверено глазами на выборках: находится на 92.8% кропов (500 шт.), медианный бокс — низ по
центру, 17% ширины и 6% высоты кропа, что и есть геометрия пластины.

Детектор нужен для диагностики и для обучения, а не для боевого инференса: 50 мс на кроп при
бюджете ~33 мс на ТС. Поэтому боксы считаются один раз и кладутся в кеш (build_cache).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import cv2
    _CV2_ERR = None
except Exception as e:              # opencv не нужен для боевого инференса, но нужен здесь
    cv2, _CV2_ERR = None, e


def require_cv2() -> None:
    """Падать громко. Тихий возврат пустого списка один раз уже стоил ночи: кеш собрался
    из 11416 пустых записей, абляция показала «маска ничего не меняет», и это выглядело как
    результат, а было отсутствием opencv."""
    if cv2 is None:
        raise SystemExit(
            f"нужен opencv, а импорт не удался: {_CV2_ERR}\n"
            f"поставь: pip install opencv-python-headless")

FLAT_T = 4.0        # перепад яркости внутри ячейки, уровней
FILL_MIN = 0.92     # компонента должна заполнять свой bbox — ячейка это прямоугольник
SIZE_TOL = 0.28     # насколько соседняя ячейка может отличаться размером
GAP_TOL = 0.8       # зазор между соседями, в долях размера ячейки
MIN_CELLS = 3       # меньше — не решётка, а случайное совпадение
PAD = 0.06          # запас вокруг найденной зоны, в долях её размера

# Запасная полоса: где номер может быть, если детектор ничего не нашёл. Нужна только для
# режима «закрыть гарантированно», и доля таких кропов измеряется отдельно.
FALLBACK_BAND = (0.22, 0.62, 0.78, 0.93)


def _cells(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    H, W = gray.shape
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    k = np.ones((3, 3), np.uint8)
    flat = ((cv2.dilate(g, k) - cv2.erode(g, k)) < FLAT_T).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(flat, 4)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 3 or h < 3 or area < 14:
            continue
        if w > 0.45 * W or h > 0.45 * H:            # пол-кропа одной ячейкой не бывает
            continue
        if area / float(w * h) < FILL_MIN:
            continue
        out.append((int(x), int(y), int(w), int(h)))
    return out


def _group(cells: list[tuple[int, int, int, int]]) -> list[list[int]]:
    """Объединение ячеек одинакового размера, стоящих рядом (система непересекающихся множеств)."""
    n = len(cells)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        xi, yi, wi, hi = cells[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = cells[j]
            if abs(wi - wj) > SIZE_TOL * max(wi, wj) or abs(hi - hj) > SIZE_TOL * max(hi, hj):
                continue
            gap_x = max(0, max(xi, xj) - min(xi + wi, xj + wj))
            gap_y = max(0, max(yi, yj) - min(yi + hi, yj + hj))
            if gap_x <= GAP_TOL * max(wi, wj) and gap_y <= GAP_TOL * max(hi, hj):
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= MIN_CELLS]


def detect(crop_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Зоны анонимизации на кропе -> [(x, y, w, h)] в пикселях кропа. Пустой список — не нашли."""
    require_cv2()
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
    H, W = gray.shape
    cells = _cells(gray)
    if len(cells) < MIN_CELLS:
        return []
    out = []
    for grp in _group(cells):
        x0 = min(cells[i][0] for i in grp); y0 = min(cells[i][1] for i in grp)
        x1 = max(cells[i][0] + cells[i][2] for i in grp)
        y1 = max(cells[i][1] + cells[i][3] for i in grp)
        px, py = int(PAD * (x1 - x0)) + 1, int(PAD * (y1 - y0)) + 1
        out.append((max(0, x0 - px), max(0, y0 - py),
                    min(W, x1 + px) - max(0, x0 - px), min(H, y1 + py) - max(0, y0 - py)))
    return out


def fallback_box(W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = FALLBACK_BAND
    return int(x0 * W), int(y0 * H), int((x1 - x0) * W), int((y1 - y0) * H)


def control_boxes(boxes, W=1.0, H=1.0, mode: str = "up", seed: int = 0):
    """Контрольные прямоугольники ТОЙ ЖЕ площади и формы, но там, где номера заведомо нет.

    Без них падение качества от маски не интерпретируется: часть его — просто потеря пикселей.
      up   — тот же бокс, поднятый на треть высоты кропа (стекло, крыша, верх кузова);
      rand — случайное место вне найденной зоны (в среднем по датасету это и есть «цена»
             потери w*h пикселей вообще, безотносительно к тому, что на них было).
    """
    rng = np.random.default_rng(seed)
    out = []
    for (x, y, w, h) in boxes:
        if mode == "up":
            out.append([x, max(0.0, y - 0.35 * H), w, h])
        elif mode == "rand":
            nx, ny = x, y
            for _ in range(20):
                nx = float(rng.uniform(0, max(1e-6, W - w)))
                ny = float(rng.uniform(0, max(1e-6, H - h)))
                if abs(nx - x) > w or abs(ny - y) > h:      # не накрываем сам номер
                    break
            out.append([nx, ny, w, h])
        else:
            raise ValueError(f"неизвестный контроль: {mode}")
    return out


def apply_mask_frac(im, fracs, fill=(127, 127, 127)):
    """Закрасить на кропе (PIL.Image) прямоугольники, заданные в долях его размера."""
    if not fracs:
        return im
    from PIL import ImageDraw
    im = im.convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for (fx, fy, fw, fh) in fracs:
        x0, y0 = int(round(fx * W)), int(round(fy * H))
        x1, y1 = int(round((fx + fw) * W)), int(round((fy + fh) * H))
        if x1 > x0 and y1 > y0:
            d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)
    return im


# --- кеш боксов: детекция 50 мс/кроп, поэтому считаем один раз ------------------------------
# В кеше боксы хранятся В ДОЛЯХ кропа, а не в пикселях: тогда они не зависят ни от разрешения
# кадра, ни от ускоренного декодирования (оно уменьшает кроп в 2/4/8 раз).

MODES = ("det", "auto", "band", "up", "rand")


def cache_key(path: str, bbox) -> str:
    stem = Path(path).stem
    if bbox is None:
        return stem
    x, y, w, h = (int(round(float(v))) for v in bbox)
    return f"{stem}:{x},{y},{w},{h}"


def load_cache(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("boxes", d)


def save_cache(boxes: dict, path: str | Path, pad: float) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    found = sum(1 for v in boxes.values() if v)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"pad": pad, "n": len(boxes), "found": found, "boxes": boxes}, f)


def build_cache(records, pad: float = 0.05, out: str | Path | None = None,
                workers: int = 0, verbose: bool = True) -> dict:
    """Прогон детектора по всем записям -> {ключ: [[fx, fy, fw, fh], ...]}."""
    from PIL import Image
    require_cv2()
    boxes: dict[str, list] = {}
    for i, r in enumerate(records):
        try:
            with Image.open(r.path) as src:
                im = src.convert("RGB")
        except OSError:
            boxes[cache_key(r.path, r.bbox)] = []
            continue
        if r.bbox is not None:
            x, y, w, h = r.bbox
            p = pad * max(w, h)
            x0, y0 = max(0, int(round(x - p))), max(0, int(round(y - p)))
            x1, y1 = min(im.width, int(round(x + w + p))), min(im.height, int(round(y + h + p)))
            if x1 <= x0 or y1 <= y0:
                x0, y0, x1, y1 = 0, 0, im.width, im.height
            im = im.crop((x0, y0, x1, y1))
        a = np.asarray(im)
        H, W = a.shape[:2]
        boxes[cache_key(r.path, r.bbox)] = [
            [round(bx / W, 5), round(by / H, 5), round(bw / W, 5), round(bh / H, 5)]
            for (bx, by, bw, bh) in detect(a)]
        if verbose and (i + 1) % 500 == 0:
            f = sum(1 for v in boxes.values() if v)
            print(f"[plate] {i + 1}/{len(records)}, зона найдена на {f / (i + 1) * 100:.1f}%")
    if out is not None:
        save_cache(boxes, out, pad)
        f = sum(1 for v in boxes.values() if v)
        print(f"[plate] {len(boxes)} записей, зона найдена на {f / max(len(boxes), 1) * 100:.1f}% → {out}")
    return boxes


def boxes_frac(cache: dict, path: str, bbox, mode: str, seed: int = 0):
    """Доли кропа (fx, fy, fw, fh), которые надо закрасить, по режиму:

      det  — только найденное (где не нашли — не закрываем ничего, это «честная» диагностика);
      auto — найденное, а не нашли — запасная полоса: номер закрыт гарантированно;
      band — всегда запасная полоса (грубый прямоугольник, каким была первая версия);
      up / rand — КОНТРОЛЬ: тот же бокс той же площади, но там, где номера заведомо нет.
    """
    if mode not in MODES:
        raise ValueError(f"неизвестный режим маски: {mode}, доступны {MODES}")
    fb = [FALLBACK_BAND[0], FALLBACK_BAND[1],
          FALLBACK_BAND[2] - FALLBACK_BAND[0], FALLBACK_BAND[3] - FALLBACK_BAND[1]]
    if mode == "band":
        return [fb]
    found = [list(b) for b in cache.get(cache_key(path, bbox), [])]
    if mode == "det":
        return found
    if mode == "auto":
        return found or [fb]
    # Контроль парный к «det»: закрываем ТЕ ЖЕ кропы и ТУ ЖЕ площадь, только в другом месте.
    # Где зона не найдена — не закрываем ничего, иначе сравнение перестаёт быть парным.
    if not found:
        return []
    return control_boxes(found, 1.0, 1.0, mode,
                         seed=int.from_bytes(cache_key(path, bbox).encode()[-6:], "little") % 2 ** 31)
