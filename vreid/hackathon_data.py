"""Формат данных организаторов (ТЗ §5):

    images/                      полные кадры, случайные имена
    train.csv                    image_id, x, y, w, h, vehicle_id
    test.csv                     image_id, x, y, w, h          (vehicle_id нет)

Один кадр может содержать несколько bbox. Камера/время/гео не выдаются.

Что делает модуль:
  read_annotations     — CSV → список Record (имена колонок настраиваются в конфиге)
  split_open_set       — train → train-fit / val по vehicle_id без пересечения
  make_query_gallery   — val → query / gallery + дистракторы (запросы без пары в галерее)
  assign_pseudo_cameras — псевдокамера: кадр = камера; кадры группируются по разрешению и фону
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets import Record, Split

DEFAULT_COLS = {"image_id": "image_id", "x": "x", "y": "y", "w": "w", "h": "h", "vehicle_id": "vehicle_id",
                "camera_id": "camera_id"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


_img_index: dict[Path, dict[str, str]] = {}


def _build_index(images_dir: Path) -> dict[str, str]:
    """Рекурсивный указатель «имя файла → путь», строится один раз на папку.

    Страховка на закрытый прогон: в опубликованном наборе картинки лежат плоско в images/,
    но если на стенде они окажутся в подпапках (images/cam01/...) или с другим
    регистром расширения, прямой путь не сойдётся и весь прогон упадёт на первом же кадре.
    Обход папки стоит долю секунды и случается только после первого промаха."""
    idx: dict[str, str] = {}
    for f in images_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        for k in (f.name, f.stem, f.name.lower(), f.stem.lower()):
            idx.setdefault(k, str(f))
    return idx


def _resolve_image(images_dir: Path, image_id: str) -> str:
    p = images_dir / image_id
    if p.exists():
        return str(p)
    for ext in IMG_EXT:
        q = images_dir / f"{image_id}{ext}"
        if q.exists():
            return str(q)
    if images_dir not in _img_index:
        _img_index[images_dir] = _build_index(images_dir) if images_dir.is_dir() else {}
        if _img_index[images_dir]:
            print(f"[data] прямой путь к {image_id} не нашёлся — построил указатель "
                  f"{images_dir} ({len(_img_index[images_dir])} ключей, включая подпапки)")
    idx = _img_index[images_dir]
    name = Path(image_id).name
    for k in (image_id, name, Path(name).stem, name.lower(), Path(name).stem.lower()):
        if k in idx:
            return idx[k]
    return str(p)  # пусть упадёт при открытии с понятным путём


def read_annotations(csv_path: str | Path, images_dir: str | Path, cols: dict | None = None,
                     bbox_format: str = "xywh", strict: bool = False) -> list[Record]:
    """CSV → Record'ы. key = image_id, если в файле один bbox на кадр (как у организаторов),
    иначе '<image_id>#<номер строки>'. Этим ключом пишем submission/candidates и держим порядок
    embeddings.npy. camera_id берётся из CSV, если колонка есть (иначе cam=0 → псевдокамера)."""
    cols = {**DEFAULT_COLS, **(cols or {})}
    images_dir = Path(images_dir)
    if bbox_format not in ('xywh','xyxy'):
        raise ValueError('bbox_format must be xywh or xyxy')
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = [str(r[cols["image_id"]]) for r in rows]
    unique = len(set(ids)) == len(ids)
    if strict and (not ids or not unique or any(not x.strip() or x!=x.strip() for x in ids)):
        raise ValueError('Competition input requires nonempty unique image_id; IDs will not be rewritten')
    has_cam = bool(rows) and cols["camera_id"] in rows[0]
    out = []
    for i, row in enumerate(rows):
        image_id = ids[i]
        x, y = float(row[cols["x"]]), float(row[cols["y"]])
        w, h = float(row[cols["w"]]), float(row[cols["h"]])
        if bbox_format == "xyxy":
            w, h = w - x, h - y
        if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
            # Жёстко — только в конкурсном режиме. На обучении падение из-за одной строки
            # стоит часы прогона, а в сдаче молчаливый кроп всего кадра — это уже другой объект.
            if strict:
                raise ValueError(f"битый bbox у {image_id}: нужны конечные положительные x, y, w, h")
            print(f"[data] ВНИМАНИЕ: битый bbox у {image_id} ({x}, {y}, {w}, {h})")
        vid_raw = row.get(cols["vehicle_id"], "")
        vid = _vid_to_int(vid_raw) if vid_raw not in ("", None) else -1
        cam = _to_int_cam(row[cols["camera_id"]]) if has_cam else 0
        out.append(Record(path=_resolve_image(images_dir, image_id), vid=vid, cam=cam,
                          bbox=(x, y, w, h), key=image_id if unique else f"{image_id}#{i}"))
    return out


_cam_map: dict[str, int] = {}


def _to_int_cam(s) -> int:
    s = str(s).strip()
    try:
        return int(float(s)) + 1  # +1: cam=0 зарезервирован за «камера неизвестна»
    except ValueError:
        if s not in _cam_map:
            _cam_map[s] = len(_cam_map) + 1
        return _cam_map[s]


_vid_map: dict[str, int] = {}


def _vid_to_int(s: str) -> int:
    s = str(s).strip()
    try:
        return int(float(s))
    except ValueError:
        if s not in _vid_map:
            _vid_map[s] = len(_vid_map) + 1
        return _vid_map[s]


# ---------------------------------------------------------------------------
def split_open_set(records: list[Record], val_frac: float = 0.2, seed: int = 0) -> tuple[list[Record], list[Record]]:
    """Разбиение по vehicle_id: множества id в train-fit и val не пересекаются."""
    ids = sorted({r.vid for r in records if r.vid >= 0})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = set(ids[:n_val])
    fit = [r for r in records if r.vid not in val_ids]
    val = [r for r in records if r.vid in val_ids]
    return fit, val


def make_query_gallery(val: list[Record], distractor_frac: float = 0.2, queries_per_id: int = 1,
                       seed: int = 0, cross_camera_queries: bool = True) -> tuple[Split, Split, np.ndarray]:
    """val → (query, gallery, has_match[Q]).

    Для каждого id: queries_per_id кадров в запрос, остальные в галерею.
    Дистракторы (distractor_frac от id): один кадр в запрос, в галерею — ничего → has_match=False.
    cross_camera_queries: в запрос выбираем кадр так, чтобы в галерее у этого id остался
    хотя бы один кадр с ДРУГОЙ псевдокамеры (иначе запрос по кросс-камерному протоколу невалиден).
    """
    rng = random.Random(seed)
    by_id: dict[int, list[Record]] = defaultdict(list)
    for r in val:
        by_id[r.vid].append(r)
    ids = sorted(by_id)
    rng.shuffle(ids)
    n_dis = int(round(len(ids) * distractor_frac))
    distractors = set(ids[:n_dis])

    q, g, has = [], [], []
    for vid in ids:
        recs = by_id[vid][:]
        rng.shuffle(recs)
        if vid in distractors:
            q.append(recs[0]); has.append(False)
            continue
        if len(recs) < 2:
            g.extend(recs)  # одиночный кадр — только в галерею
            continue
        cams = {r.cam for r in recs}
        chosen = []
        if cross_camera_queries and len(cams) >= 2:
            for r in recs:
                others = [o for o in recs if o is not r and o.cam != r.cam]
                if others:
                    chosen.append(r)
                if len(chosen) >= queries_per_id:
                    break
        if not chosen:
            chosen = recs[:queries_per_id]
        rest = [r for r in recs if r not in chosen]
        if not rest:
            chosen, rest = chosen[:-1], [chosen[-1]]
        q.extend(chosen); has.extend([True] * len(chosen)); g.extend(rest)
    return Split("query", q), Split("gallery", g), np.array(has, dtype=bool)


def make_query_gallery_track(val: list[Record], distractor_frac: float = 0.1,
                             seed: int = 0) -> tuple[Split, Split, np.ndarray]:
    """Протокол «как тест организаторов» (восстановлен по test_query/test_gallery):
    галерея — ровно ОДИН кадр на трек (vehicle_id × camera), запрос — все остальные кадры.
    Значит у почти каждого запроса в галерее есть кадр той же камеры (почти дубликат, sim≈0.9),
    а кросс-камерные совпадения — по одному кадру на другую камеру.
    Дистракторы (distractor_frac от id): машина целиком убрана из галереи, все её кадры — запросы
    без пары (has_match=False). Камеры в запросе НЕ маскируются при оценке отказа (на тесте их нет),
    но кросс-камерный mAP по-прежнему считается с маской по настоящим камерам."""
    rng = random.Random(seed)
    by_id: dict[int, list[Record]] = defaultdict(list)
    for r in val:
        by_id[r.vid].append(r)
    ids = sorted(by_id)
    rng.shuffle(ids)
    n_dis = int(round(len(ids) * distractor_frac))
    distractors = set(ids[:n_dis])

    q, g, has = [], [], []
    for vid in ids:
        recs = by_id[vid][:]
        rng.shuffle(recs)
        if vid in distractors:
            q.extend(recs); has.extend([False] * len(recs))
            continue
        by_cam: dict[int, list[Record]] = defaultdict(list)
        for r in recs:
            by_cam[r.cam].append(r)
        for cam in sorted(by_cam):
            track = by_cam[cam]
            g.append(track[0])
            q.extend(track[1:]); has.extend([True] * (len(track) - 1))
    return Split("query", q), Split("gallery", g), np.array(has, dtype=bool)


# ---------------------------------------------------------------------------
def frame_background_feature(path: str, bboxes: list[tuple], size=(32, 24)) -> np.ndarray:
    """Кадр без машин: bbox'ы заливаются средним цветом, кадр сжимается до size, серый, нормируется."""
    with Image.open(path) as im:
        _size_cache.setdefault(path, im.size)
        im.draft("RGB", (size[0] * 8, size[1] * 8))  # быстрый декод JPEG в уменьшенном виде
        im = im.convert("RGB")
        W, H = im.size
        arr = np.asarray(im).astype(np.float32)
        mean = arr.reshape(-1, 3).mean(axis=0)
        # bbox дан в координатах исходного кадра; draft мог уменьшить масштаб
        oW, oH = _orig_size(path)
        sx, sy = W / max(1, oW), H / max(1, oH)
        for (x, y, w, h) in bboxes:
            x0, y0 = int(x * sx), int(y * sy)
            x1, y1 = int((x + w) * sx) + 1, int((y + h) * sy) + 1
            arr[max(0, y0):y1, max(0, x0):x1] = mean
        small = Image.fromarray(arr.astype(np.uint8)).convert("L").resize(size)
        v = np.asarray(small, dtype=np.float32).ravel()
        v = (v - v.mean()) / (v.std() + 1e-6)
        return v


_size_cache: dict[str, tuple] = {}


def _orig_size(path: str) -> tuple:
    if path not in _size_cache:
        with Image.open(path) as im:
            _size_cache[path] = im.size
    return _size_cache[path]


def assign_pseudo_cameras(records: list[Record], distance_threshold: float = 0.35,
                          verbose: bool = True) -> list[Record]:
    """Псевдокамера для данных без camera_id.

    1) все bbox одного кадра — одна камера (это точно);
    2) кадры группируются по разрешению, внутри группы — агломеративная кластеризация
       признака фона (косинусная дистанция ≤ distance_threshold → одна камера).
    Порог подбирается глазами: посмотри results/pseudo_cameras_*.txt и примеры кадров одного кластера.
    """
    from sklearn.cluster import AgglomerativeClustering

    by_frame: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_frame[r.path].append(r)
    frames = sorted(by_frame)
    feats, sizes = [], []
    for p in frames:
        sizes.append(_orig_size(p))
        feats.append(frame_background_feature(p, [r.bbox for r in by_frame[p] if r.bbox]))
    feats = np.stack(feats)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8

    cam_of_frame: dict[str, int] = {}
    next_cam = 1
    for size in sorted(set(sizes)):
        idx = [i for i, s in enumerate(sizes) if s == size]
        if len(idx) == 1:
            cam_of_frame[frames[idx[0]]] = next_cam; next_cam += 1
            continue
        sub = feats[idx]
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                     metric="cosine", linkage="average").fit(sub)
        for i, lab in zip(idx, cl.labels_):
            cam_of_frame[frames[i]] = next_cam + int(lab)
        next_cam += int(cl.labels_.max()) + 1

    out = [Record(r.path, r.vid, cam_of_frame[r.path], r.bbox, r.key) for r in records]
    if verbose:
        n_cams = len(set(cam_of_frame.values()))
        print(f"[pseudo-cam] кадров {len(frames)}, разрешений {len(set(sizes))}, псевдокамер {n_cams}")
    return out


def frames_only_cameras(records: list[Record]) -> list[Record]:
    """Самый строгий вариант: каждый кадр — своя камера (без кластеризации)."""
    ids: dict[str, int] = {}
    out = []
    for r in records:
        if r.path not in ids:
            ids[r.path] = len(ids) + 1
        out.append(Record(r.path, r.vid, ids[r.path], r.bbox, r.key))
    return out


# ---------------------------------------------------------------------------
def build_local_validation(cfg: dict) -> dict:
    """Из train.csv организаторов → {'fit': Split, 'query': Split, 'gallery': Split, 'has_match': np.ndarray}."""
    root = Path(cfg["root"])
    recs = read_annotations(root / cfg.get("train_csv", "train.csv"), root / cfg.get("images_dir", "images"),
                            cfg.get("cols"), cfg.get("bbox_format", "xywh"))
    print(f"[hackathon] train.csv: {len(recs)} bbox, {len({r.vid for r in recs})} vehicle_id, "
          f"{len({r.path for r in recs})} кадров")
    mode = cfg.get("pseudo_camera", "auto")
    have_real = any(r.cam != 0 for r in recs)
    if have_real and mode in ("auto", "real"):
        print(f"[hackathon] камеры из CSV: {len({r.cam for r in recs})} шт. — псевдокамера не нужна")
    elif mode in ("cluster", "auto"):
        recs = assign_pseudo_cameras(recs, cfg.get("pseudo_camera_threshold", 0.35))
    elif mode == "frame":
        recs = frames_only_cameras(recs)
    fit, val = split_open_set(recs, cfg.get("val_frac", 0.2), cfg.get("seed", 0))
    protocol = cfg.get("protocol", "id")
    if protocol == "track":
        q, g, has = make_query_gallery_track(val, cfg.get("distractor_frac", 0.1), cfg.get("seed", 0))
    else:
        q, g, has = make_query_gallery(val, cfg.get("distractor_frac", 0.2), cfg.get("queries_per_id", 1), cfg.get("seed", 0))
    print(f"[hackathon] протокол «{protocol}», fit: {len(fit)} bbox / {len({r.vid for r in fit})} id; "
          f"val query: {len(q)} (без пары: {int((~has).sum())}), gallery: {len(g)}")
    return {"fit": Split("fit", fit), "query": q, "gallery": g, "has_match": has, "protocol": protocol,
            # уверенность для отказа: на тесте камер нет → при протоколе track не маскируем свою камеру
            "refusal_mask_cam": protocol != "track"}


def load_test(cfg: dict) -> dict[str, Split]:
    """Тест организаторов: либо пара test_query.csv / test_gallery.csv → {'query', 'gallery'},
    либо один test.csv → {'test'} (каждая запись — запрос ко всем остальным)."""
    root = Path(cfg["root"])
    images = root / cfg.get("images_dir", "images")
    args = (cfg.get("cols"), cfg.get("bbox_format", "xywh"))
    qcsv, gcsv = cfg.get("test_query_csv"), cfg.get("test_gallery_csv")
    if qcsv and gcsv and (root / qcsv).exists() and (root / gcsv).exists():
        q = Split("query", read_annotations(root / qcsv, images, *args))
        g = Split("gallery", read_annotations(root / gcsv, images, *args))
        print(f"[hackathon] test: query {len(q)}, gallery {len(g)}")
        return {"query": q, "gallery": g}
    recs = read_annotations(root / cfg.get("test_csv", "test.csv"), images, *args)
    return {"test": Split("test", recs)}
