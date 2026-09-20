"""Извлечение эмбеддингов для сплита → .npz (emb, vids, cams, paths)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .datasets import Split
from .models import Backbone


_bad_bbox_seen: set = set()


def _bad_bbox(bbox, size) -> None:
    """Один раз на каждый случай, но не больше десяти — чтобы не затопить лог."""
    key = (tuple(float(v) for v in bbox), size)
    if len(_bad_bbox_seen) < 10 and key not in _bad_bbox_seen:
        _bad_bbox_seen.add(key)
        print(f"[extract] ВНИМАНИЕ: bbox {bbox} вне кадра {size} — прижал к границе. "
              f"Этот эмбеддинг считается не по тому объекту, проверь входной CSV")
    elif len(_bad_bbox_seen) == 10:
        _bad_bbox_seen.add("...")
        print("[extract] битых bbox больше десяти — дальше не повторяюсь")


def crop_bbox(im: Image.Image, bbox, pad: float = 0.08, mask_frac=None) -> Image.Image:
    """Кроп по (x, y, w, h) с паддингом pad·max(w,h) с каждой стороны, в границах кадра.
    mask_frac — прямоугольники в ДОЛЯХ кропа, которые надо закрасить (см. vreid/plate.py)."""
    x, y, w, h = bbox
    if not np.isfinite([x,y,w,h,pad]).all() or w<=0 or h<=0 or pad<0:
        raise ValueError('Invalid bbox or padding')
    p = pad * max(w, h)
    W, H = im.size
    x0, y0 = max(0, int(round(x - p))), max(0, int(round(y - p)))
    x1, y1 = min(W, int(round(x + w + p))), min(H, int(round(y + h + p)))
    if x1 <= x0 or y1 <= y0:
        # ББox сам по себе корректен (конечный, w>0, h>0), но целиком вне кадра.
        # Было у нас: брать весь кадр — это тихая подмена объекта.
        # Было у напарника: падать — а это ноль за весь прогон из-за одной строки,
        # при том что строка embedding обязательна для каждого image_id (ответ 49).
        # Стало: прижимаем к ближайшему куску кадра и громко сообщаем. Результат плохой,
        # но локальный: теряется один запрос, а не вся сдача.
        x0 = min(max(0, x0), W - 1); y0 = min(max(0, y0), H - 1)
        x1 = max(x0 + 1, min(W, x1)); y1 = max(y0 + 1, min(H, y1))
        _bad_bbox(bbox, (W, H))
    crop = im.crop((x0, y0, x1, y1))
    if mask_frac:
        from .plate import apply_mask_frac
        crop = apply_mask_frac(crop, mask_frac)
    return crop


def open_crop(path, bbox, pad: float = 0.08, fast_decode: bool = False, target: int = 336,
              mask_frac=None) -> Image.Image:
    """Открыть кадр и вырезать bbox. При fast_decode просим у libjpeg сразу уменьшенный масштаб
    (PIL draft умеет 1/2, 1/4, 1/8) — но только если кроп после уменьшения всё ещё не мельче
    входа модели, иначе теряем детали. На кадрах 1920x1080 это заметно дешевле полного декода."""
    with Image.open(path) as src:            # закрываем дескриптор сразу: воркеров много
        ow, oh = src.size
        if bbox is not None and (len(bbox)!=4 or not np.isfinite(bbox).all() or min(bbox[2:])<=0):
            raise ValueError('Invalid bbox before decoding')
        if fast_decode and bbox is not None:
            need = target / max(float(bbox[2]), float(bbox[3]))   # во сколько уменьшать можно
            if need < 1.0:
                src.draft("RGB", (max(1, int(ow * need)), max(1, int(oh * need))))
        im = src.convert("RGB")
    if bbox is None:
        if mask_frac:
            from .plate import apply_mask_frac
            im = apply_mask_frac(im, mask_frac)
        return im
    sx, sy = im.width / ow, im.height / oh
    b = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)
    return crop_bbox(im, b, pad, mask_frac)


class _DS:
    def __init__(self, split: Split, backbone: Backbone, pad: float = 0.08, mask=None,
                 fast_decode: bool = False):
        self.split, self.bb, self.pad, self.mask = split, backbone, pad, mask
        self.fast, self.size = fast_decode, int(getattr(backbone, "size", 336))

    def __len__(self):
        return len(self.split)

    def _frac(self, r):
        if not self.mask:
            return None
        from .plate import boxes_frac
        return boxes_frac(self.mask["cache"], r.path, r.bbox, self.mask["mode"])

    def __getitem__(self, i):
        r = self.split.records[i]
        return self.bb.transform(open_crop(r.path, r.bbox, self.pad, self.fast, self.size,
                                           self._frac(r)))


def extract_split(split: Split, backbone: Backbone, batch_size: int = 64,
                  num_workers: int = 4, out: str | Path | None = None,
                  pad: float = 0.08, mask=None, tta_flip: bool = False,
                  fast_decode: bool = False) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    loader = DataLoader(_DS(split, backbone, pad, mask, fast_decode), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    import os, time
    throttle = float(os.environ.get("VREID_THROTTLE", "0") or 0)
    chunks = []
    desc = f"extract[{split.name}]" + ("+flip" if tta_flip else "")
    t_gpu, t_wall = 0.0, time.perf_counter()
    for batch in tqdm(loader, desc=desc, unit="batch"):
        t0 = time.perf_counter()
        e = backbone.embed_tensors(batch)
        if tta_flip and batch.ndim == 4:   # тест-тайм аугментация: среднее вектора кропа и его зеркала
            e = e + backbone.embed_tensors(torch.flip(batch, dims=[3]))
            e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        chunks.append(e)
        t_gpu += time.perf_counter() - t0
        if throttle > 0:
            time.sleep(throttle * (time.perf_counter() - t0))
    emb = np.concatenate(chunks, axis=0).astype(np.float32)
    t_all = time.perf_counter() - t_wall
    n = max(len(emb), 1)
    print(f"[extract] {split.name}: {t_all / n * 1000:.1f} мс/кроп всего = "
          f"{t_gpu / n * 1000:.1f} сеть + {(t_all - t_gpu) / n * 1000:.1f} чтение кадров с диска")
    data = dict(emb=emb, vids=split.vids, cams=split.cams, paths=np.array(split.paths),
                keys=np.array(split.keys))
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **data)
        print(f"[extract] {split.name}: {emb.shape} → {out}")
    return data


def load_npz(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
