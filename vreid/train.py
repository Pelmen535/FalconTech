"""Дообучение backbone под данные организаторов (metric learning по vehicle_id).

  python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b --epochs 20 --out weights/dinov2_b_ft

Что внутри:
  * данные: fit-часть локальной open-set валидации (см. hackathon_data.build_local_validation);
    val-часть — для ранней остановки по кросс-камерному mAP (по псевдокамере);
  * кропы режутся один раз в кэш с запасом 30% (crops_cache/), дальше — случайный джиттер bbox
    внутри запаса: это дёшево и даёт устойчивость к неточным bbox;
  * аугментации под камеры: сдвиг/масштаб bbox, цвет, grayscale (ночь), понижение разрешения + JPEG
    (плохие камеры), random erasing (перекрытия), флип; маска зоны номера — как на инференсе;
  * модель: timm backbone → cls+mean → BNNeck; лоссы: ArcFace (по id) + triplet batch-hard;
    PK-сэмплер (P машин × K кадров в батче);
  * веса сохраняются в <out>/best.pt и грузятся через get_backbone("ft:<out>/best.pt").
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

from .datasets import Record, Split
from .extract import crop_bbox
from .hackathon_data import build_local_validation
from .metrics import evaluate, format_metrics


# --------------------------------------------------------------------------- кэш кропов
def prepare_crops(records: list[Record], cache_dir: Path, pad: float = 0.30, quality: int = 95) -> list[Record]:
    """Вырезает bbox с запасом pad и сохраняет как JPEG. Возвращает записи, где path — кроп,
    а bbox — положение исходного bbox внутри кропа (чтобы джиттер знал, где машина)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    by_frame: dict[str, list[tuple[int, Record]]] = defaultdict(list)
    for i, r in enumerate(records):
        by_frame[r.path].append((i, r))
    todo = {i: None for i in range(len(records))}
    for path, items in by_frame.items():
        need = [(i, r) for i, r in items if not (cache_dir / f"{_safe(r.key)}.jpg").exists()]
        if need:
            with Image.open(path) as im:
                im = im.convert("RGB")
                W, H = im.size
                for i, r in need:
                    x, y, w, h = r.bbox
                    p = pad * max(w, h)
                    x0, y0 = max(0, int(x - p)), max(0, int(y - p))
                    x1, y1 = min(W, int(x + w + p)), min(H, int(y + h + p))
                    im.crop((x0, y0, x1, y1)).save(cache_dir / f"{_safe(r.key)}.jpg", quality=quality)
        for i, r in items:
            x, y, w, h = r.bbox
            p = pad * max(w, h)
            x0, y0 = max(0, int(x - p)), max(0, int(y - p))
            todo[i] = Record(str(cache_dir / f"{_safe(r.key)}.jpg"), r.vid, r.cam, (x - x0, y - y0, w, h), r.key)
    return [todo[i] for i in range(len(records))]


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


# --------------------------------------------------------------------------- аугментации
class TrainTransform:
    def __init__(self, img_size: int, mean, std, mask_p: float = 0.0, strong: bool = True):
        import torchvision.transforms as T
        self.size = img_size
        # mask_p — вероятность закрасить зону анонимизации номера (боксы приходят из кеша).
        # Это аугментация «номера нет», а не фиксированный прямоугольник: она учит модель
        # не опираться на эту зону, при этом на инференсе ничего закрывать не нужно.
        self.mask_p = float(mask_p)
        self.strong = strong
        self.color = T.ColorJitter(0.3, 0.3, 0.3, 0.05)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        self.erase = T.RandomErasing(p=0.4, scale=(0.02, 0.2), value="random")

    def __call__(self, im: Image.Image, bbox, mask_frac=None) -> "torch.Tensor":
        x, y, w, h = bbox
        # джиттер bbox: масштаб 0.9–1.2, сдвиг ±10%
        s = random.uniform(0.9, 1.2)
        dx, dy = random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h
        cx, cy = x + w / 2 + dx, y + h / 2 + dy
        nw, nh = w * s, h * s
        crop = im.crop((int(cx - nw / 2), int(cy - nh / 2), int(cx + nw / 2), int(cy + nh / 2)))
        if mask_frac and self.mask_p and random.random() < self.mask_p:
            from .plate import apply_mask_frac
            crop = apply_mask_frac(crop, mask_frac)
        if random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        if self.strong:
            crop = self.color(crop)
            if random.random() < 0.1:
                crop = crop.convert("L").convert("RGB")
            if random.random() < 0.3:  # плохая камера: даунскейл + jpeg
                f = random.uniform(0.3, 0.7)
                small = crop.resize((max(8, int(crop.width * f)), max(8, int(crop.height * f))), Image.BILINEAR)
                buf = io.BytesIO(); small.save(buf, "JPEG", quality=random.randint(30, 70)); buf.seek(0)
                crop = Image.open(buf).convert("RGB")
            if random.random() < 0.15:
                crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        crop = crop.resize((self.size, self.size), Image.BICUBIC)
        t = self.norm(crop)
        return self.erase(t) if self.strong else t


class EvalTransform:
    def __init__(self, img_size: int, mean, std, pad: float, mask_frac=None):
        import torchvision.transforms as T
        self.size, self.pad, self.mask_frac = img_size, pad, mask_frac
        self.norm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __call__(self, im: Image.Image, bbox, mask_frac=None):
        crop = crop_bbox(im, bbox, self.pad, mask_frac) if bbox is not None else im
        return self.norm(crop.resize((self.size, self.size), Image.BICUBIC))


class CropDataset:
    def __init__(self, records: list[Record], tf, label_map: dict | None = None, mask=None):
        self.records, self.tf, self.label_map = records, tf, label_map
        self.mask = mask          # {"cache": ..., "mode": ...} из vreid/plate.py или None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        mf = None
        if self.mask:
            from .plate import boxes_frac
            mf = boxes_frac(self.mask["cache"], r.path, r.bbox, self.mask["mode"])
        with Image.open(r.path) as im:
            x = self.tf(im.convert("RGB"), r.bbox, mf)
        y = self.label_map[r.vid] if self.label_map else -1
        return x, y, r.cam


class PKSampler:
    """Батч = P машин × K кадров каждой. Машины с < K кадрами добираются повтором."""

    def __init__(self, labels: list[int], P: int, K: int, seed: int = 0):
        self.by_label = defaultdict(list)
        for i, l in enumerate(labels):
            self.by_label[l].append(i)
        self.P, self.K = P, K
        self.rng = random.Random(seed)
        self.n_batches = max(1, len(labels) // (P * K))

    def __iter__(self):
        labels = list(self.by_label)
        for _ in range(self.n_batches):
            chosen = self.rng.sample(labels, min(self.P, len(labels)))
            batch = []
            for l in chosen:
                idx = self.by_label[l]
                batch += self.rng.sample(idx, self.K) if len(idx) >= self.K else [self.rng.choice(idx) for _ in range(self.K)]
            yield batch

    def __len__(self):
        return self.n_batches


class CamAwarePKSampler(PKSampler):
    """То же P×K, но K кадров машины берутся с РАЗНЫХ камер по кругу: сначала по одному кадру
    с каждой камеры, потом добор. Тогда «трудный позитив» в батче почти всегда кросс-камерный —
    ровно то, что считает жюри. Машины с одной камерой ведут себя как в обычном сэмплере."""

    def __init__(self, labels: list[int], cams: list[int], P: int, K: int, seed: int = 0):
        super().__init__(labels, P, K, seed)
        self.by_label_cam: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, (l, c) in enumerate(zip(labels, cams)):
            self.by_label_cam[l][c].append(i)

    def _pick(self, l: int) -> list[int]:
        groups = [list(v) for v in self.by_label_cam[l].values()]
        for g in groups:
            self.rng.shuffle(g)
        self.rng.shuffle(groups)
        out, r = [], 0
        while len(out) < self.K and any(groups):
            g = groups[r % len(groups)]
            if g:
                out.append(g.pop())
            r += 1
            if r > 10 * self.K:
                break
        while len(out) < self.K:  # добор повтором, если кадров меньше K
            out.append(self.rng.choice(self.by_label[l]))
        return out

    def __iter__(self):
        labels = list(self.by_label)
        for _ in range(self.n_batches):
            chosen = self.rng.sample(labels, min(self.P, len(labels)))
            batch = []
            for l in chosen:
                batch += self._pick(l)
            yield batch


# --------------------------------------------------------------------------- модель и лоссы
def build_model(model_name: str, img_size: int, pretrained: bool = True):
    import timm
    import torch.nn as nn
    kwargs = {"pretrained": pretrained, "num_classes": 0}
    if "patch14" in model_name or "dinov2" in model_name or "vit" in model_name:
        kwargs["img_size"] = img_size
        kwargs["dynamic_img_size"] = True
    backbone = timm.create_model(model_name, **kwargs)
    cfg = timm.data.resolve_data_config({}, model=backbone)
    return backbone, cfg["mean"], cfg["std"]


class ReIDModel:
    """Обёртка: backbone → cls+mean (ViT) или pooled (CNN) → BNNeck. Сохраняется/грузится целиком."""

    def __init__(self, model_name: str, img_size: int, pretrained: bool = True):
        import torch.nn as nn
        self.model_name, self.img_size = model_name, img_size
        self.backbone, self.mean, self.std = build_model(model_name, img_size, pretrained)
        self.is_vit = hasattr(self.backbone, "forward_features") and hasattr(self.backbone, "cls_token")
        d = self.backbone.num_features * (2 if self.is_vit else 1)
        self.bnneck = nn.BatchNorm1d(d)
        self.bnneck.bias.requires_grad_(False)
        self.dim = d

    def features(self, x):
        import torch
        if self.is_vit:
            tok = self.backbone.forward_features(x)
            n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            f = torch.cat([tok[:, 0], tok[:, n_prefix:].mean(dim=1)], dim=1)
        else:
            f = self.backbone(x)
        return f, self.bnneck(f)

    def modules(self):
        return [self.backbone, self.bnneck]

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

    def train(self):
        for m in self.modules():
            m.train()

    def eval(self):
        for m in self.modules():
            m.eval()

    def state(self):
        return {"model_name": self.model_name, "img_size": self.img_size, "dim": self.dim,
                "mean": list(self.mean), "std": list(self.std),
                "backbone": self.backbone.state_dict(), "bnneck": self.bnneck.state_dict()}

    @staticmethod
    def load(path: str, device: str):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ReIDModel(ck["model_name"], ck["img_size"], pretrained=False)
        m.backbone.load_state_dict(ck["backbone"])
        m.bnneck.load_state_dict(ck["bnneck"])
        return m.to(device)


class ArcFace:
    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.3, device="cpu",
                 label_smoothing: float = 0.0):
        import torch
        import torch.nn as nn
        self.W = nn.Parameter(torch.empty(n_classes, dim, device=device))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m, self.ls = s, m, label_smoothing

    def __call__(self, f, y):
        import torch
        import torch.nn.functional as F
        cos = F.linear(F.normalize(f), F.normalize(self.W)).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, y, label_smoothing=self.ls)


def similarity_distill(f_student, f_teacher):
    """Дистилляция геометрии, а не самих векторов (Tung & Mori, 2019).

    Учитель и ученик могут иметь разную размерность (у нас 2048 против 1536), поэтому сравниваем
    не эмбеддинги, а матрицы попарных косинусов внутри батча: B×B у обоих, размерность не мешает.
    Ученик перенимает ровно то, что нужно для поиска — кто на кого похож и насколько, — а не
    случайную систему координат учителя."""
    import torch.nn.functional as F
    s = F.normalize(f_student, dim=1)
    t = F.normalize(f_teacher, dim=1)
    return ((s @ s.T) - (t @ t.T)).pow(2).mean()


def triplet_batch_hard(f, y, margin: float = 0.3, cams=None):
    """Batch-hard triplet. Если переданы cams — самый трудный позитив ищется только среди кадров
    той же машины с ДРУГОЙ камеры (как в протоколе жюри); если таких в батче нет — среди любых."""
    import torch
    import torch.nn.functional as F
    f = F.normalize(f)
    d = torch.cdist(f, f)
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos_mask = same & ~eye
    if cams is not None:
        cross = pos_mask & (cams[:, None] != cams[None, :])
        has_cross = cross.any(dim=1, keepdim=True)
        pos_mask = torch.where(has_cross, cross, pos_mask)
    pos = d.masked_fill(~pos_mask, -1).max(dim=1).values
    neg = d.masked_fill(same, 9.0).min(dim=1).values
    return F.relu(pos - neg + margin).mean()


# --------------------------------------------------------------------------- автоподбор батча
def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ in ("OutOfMemoryError",)


def find_batch_P(model, arc, img_size: int, K: int, device: str, p_max: int = 24, p_min: int = 2,
                 headroom: float = 0.80, teacher=None) -> int:
    """Подбор числа машин в батче под память GPU без лобовых OOM:
    1) замер пика памяти на маленьком батче (P=2), 2) линейная экстраполяция на headroom·VRAM,
    3) проверка одним реальным шагом; если всё же не влезло — шаг вниз на 20% и повтор.
    На CPU возвращает 4."""
    import torch
    if device != "cuda":
        return 4

    def try_step(P):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(P * K, 3, img_size, img_size, device=device)
        y = torch.arange(P, device=device).repeat_interleave(K) % arc.W.shape[0]
        with torch.autocast(device_type="cuda"):
            f, fb = model.features(x)
        loss = arc(fb.float(), y) + triplet_batch_hard(f.float(), y)
        if teacher is not None:                      # учитель тоже занимает память, хоть и без градиентов
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                teacher.features(x)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        for m in model.modules():
            m.zero_grad(set_to_none=True)
        arc.W.grad = None
        del x, y, f, fb, loss
        torch.cuda.empty_cache()
        return peak

    total = torch.cuda.get_device_properties(0).total_memory
    budget = headroom * total
    model.train()
    base = torch.cuda.memory_allocated()
    try:
        peak_small = try_step(p_min)
    except Exception as e:
        if not _is_oom(e):
            raise
        print(f"[train] авто-батч: даже P={p_min} не влезает — уменьшай img-size или включай --freeze-blocks")
        return p_min
    per_car = max((peak_small - base) / p_min, 1.0)
    P = int(max(p_min, min(p_max, (budget - base) // per_car)))
    print(f"[train] авто-батч: оценка по P={p_min}: {per_car / 2**30:.2f} ГБ на машину (×K={K}), "
          f"база {base / 2**30:.1f} ГБ, бюджет {budget / 2**30:.1f} ГБ → пробую P={P}")
    while P >= p_min:
        try:
            peak = try_step(P)
            if peak <= budget or P == p_min:
                print(f"[train] авто-батч: P={P} (×K={K} = {P * K} кропов), пик {peak / 2**30:.1f} ГБ из {total / 2**30:.1f}")
                return P
            print(f"[train] авто-батч: P={P} влезает, но пик {peak / 2**30:.1f} ГБ выше бюджета — уменьшаю")
        except Exception as e:
            if not _is_oom(e):
                raise
            for m in model.modules():
                m.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"[train] авто-батч: P={P} — OOM, уменьшаю")
        P = max(p_min, int(P * 0.8)) if P > p_min else p_min - 1
    return p_min


# --------------------------------------------------------------------------- валидация
def embed_records(model: ReIDModel, records: list[Record], tf, device, batch_size=64, workers=4):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(CropDataset(records, tf), batch_size=batch_size, shuffle=False, num_workers=workers)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
        for x, _, _ in dl:
            _, fb = model.features(x.to(device))
            out.append(torch.nn.functional.normalize(fb.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def validate(model, q: Split, g: Split, has_match, tf, device, batch_size, workers) -> dict:
    qe, ge = embed_records(model, q.records, tf, device, batch_size, workers), embed_records(model, g.records, tf, device, batch_size, workers)
    sims = qe @ ge.T
    m = has_match
    # Правило жюри: убираем только пары «тот же vehicle_id И та же camera_id», и считаем mAP@10.
    # Раньше здесь стоял строгий cross-camera режим без cutoff — то есть лучшая эпоха выбиралась
    # по метрике, которая завышает результат и не совпадает с оцениваемой.
    return evaluate(sims[m], q.vids[m], q.cams[m], g.vids, g.cams, cutoff=10)


# --------------------------------------------------------------------------- обучение
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="dinov2_b", help="dinov2_s|dinov2_b|dinov2_l|clip_b|resnet50|<timm name>")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--P", default="auto", help="машин в батче; auto — подобрать максимум, влезающий в память GPU")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5, help="lr backbone")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1, help="эпох разогрева")
    ap.add_argument("--freeze-blocks", type=int, default=0, help="заморозить первые N блоков ViT")
    ap.add_argument("--arc-m", type=float, default=0.3)
    ap.add_argument("--arc-ls", type=float, default=0.0,
                    help="label smoothing в ArcFace; 0.1 против переобучения (loss уходит в 0.1 к 12-й эпохе)")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="переопределить dataset.val_frac; 0 — обучение на ВСЕХ id без валидации (финальная модель)")
    ap.add_argument("--mask-p", type=float, default=0.0,
                    help="вероятность закрасить зону анонимизации номера при обучении (0 — не закрывать). "
                         "Учит не опираться на эту зону, при этом на инференсе закрывать ничего не нужно")
    ap.add_argument("--plate-cache", default="results/plate_boxes.json",
                    help="кеш боксов зоны анонимизации (см. scripts/build_plate_cache.py)")
    ap.add_argument("--tri-w", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--no-select", action="store_true",
                    help="ровно --epochs эпох, без выбора лучшей и без ранней остановки "
                         "(для валидационного двойника релиза, который обучен с --val-frac 0)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size-eval", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pretrained", default="true")
    ap.add_argument("--no-strong-aug", action="store_true")
    ap.add_argument("--distill", default=None,
                    help="путь к весам учителя (weights/.../best.pt): переносим геометрию сходств")
    ap.add_argument("--distill-w", type=float, default=1.0, help="вес слагаемого дистилляции")
    ap.add_argument("--cam-aware", action="store_true",
                    help="K кадров машины брать с разных камер (кросс-камерные позитивы в батче)")
    ap.add_argument("--cross-cam-triplet", action="store_true",
                    help="в triplet трудный позитив искать только с другой камеры")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--throttle", type=float, default=None,
                    help="доля паузы после каждого шага (0.15 = GPU занят ~85%%), чтобы компьютер не лагал; "
                         "по умолчанию из переменной окружения VREID_THROTTLE")
    a = ap.parse_args(argv)
    import os
    throttle = a.throttle if a.throttle is not None else float(os.environ.get("VREID_THROTTLE", "0") or 0)
    if throttle > 0:
        print(f"[train] throttle={throttle:.2f}: после каждого шага пауза {throttle * 100:.0f}% от времени шага")

    import torch
    from torch.utils.data import DataLoader
    from .models import _REGISTRY
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(a.out or f"weights/{cfg['name']}_{a.backbone}_{a.img_size}")
    out.mkdir(parents=True, exist_ok=True)

    if a.val_frac is not None:
        cfg["dataset"] = {**cfg["dataset"], "val_frac": a.val_frac}
    full_fit = float(cfg["dataset"].get("val_frac", 0.2)) <= 0
    if full_fit:
        from .hackathon_data import read_annotations
        ds_cfg = cfg["dataset"]; root = Path(ds_cfg["root"])
        recs = read_annotations(root / ds_cfg.get("train_csv", "train.csv"),
                                root / ds_cfg.get("images_dir", "images"),
                                ds_cfg.get("cols"), ds_cfg.get("bbox_format", "xywh"))
        d = {"fit": Split("fit", recs), "query": None, "gallery": None, "has_match": None}
        print(f"[train] ФИНАЛЬНЫЙ режим: все {len({r.vid for r in recs})} id в обучении, валидации нет — "
              f"сохраняется последняя эпоха, число эпох задай по лучшему прогону с валидацией")
    else:
        d = build_local_validation(cfg["dataset"])
    crop_cfg = cfg.get("crop", {})
    cache = Path(cfg.get("runs_dir", "runs")) / cfg["name"] / "crops_cache"
    t0 = time.time()
    fit = prepare_crops(d["fit"].records, cache / "fit")
    print(f"[train] кропы fit: {len(fit)} ({time.time() - t0:.0f} c)")
    ids = sorted({r.vid for r in fit})
    label_map = {v: i for i, v in enumerate(ids)}

    model_name = _REGISTRY.get(a.backbone, {}).get("model_name", a.backbone)
    model = ReIDModel(model_name, a.img_size, pretrained=(a.pretrained.lower() == "true")).to(device)
    if a.freeze_blocks and hasattr(model.backbone, "blocks"):
        for p in model.backbone.patch_embed.parameters():
            p.requires_grad_(False)
        for blk in model.backbone.blocks[: a.freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad_(False)
    arc = ArcFace(model.dim, len(ids), m=a.arc_m, device=device, label_smoothing=a.arc_ls)
    print(f"[train] {model_name}, D={model.dim}, id={len(ids)}, device={device}")

    teacher = None
    if a.distill:
        teacher = ReIDModel.load(a.distill, device)
        for sub in teacher.modules():
            sub.eval()
            for prm in sub.parameters():
                prm.requires_grad_(False)
        if teacher.img_size != a.img_size:
            raise SystemExit(f"учитель обучен на входе {teacher.img_size}, ученик на {a.img_size} — "
                             f"нужен одинаковый вход, иначе кропы не совпадут")
        print(f"[train] дистилляция от {a.distill}: D учителя {teacher.dim}, вес {a.distill_w}")

    if str(a.P).lower() == "auto":
        a.P = find_batch_P(model, arc, a.img_size, a.K, device, teacher=teacher)
    else:
        a.P = int(a.P)

    tf_train = TrainTransform(a.img_size, model.mean, model.std, float(a.mask_p), strong=not a.no_strong_aug)
    tf_eval = EvalTransform(a.img_size, model.mean, model.std, crop_cfg.get("pad", 0.08))
    mask = None
    if a.mask_p > 0:
        from .plate import load_cache
        mask = {"cache": load_cache(a.plate_cache), "mode": "det"}
        print(f"[train] аугментация «номер закрыт» с p={a.mask_p}, боксов в кеше {len(mask['cache'])}")
    ds = CropDataset(fit, tf_train, label_map, mask=mask)
    if a.cam_aware:
        sampler = CamAwarePKSampler([label_map[r.vid] for r in fit], [r.cam for r in fit], a.P, a.K, a.seed)
        print("[train] сэмплер: camera-aware (кадры одной машины — с разных камер)")
    else:
        sampler = PKSampler([label_map[r.vid] for r in fit], a.P, a.K, a.seed)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=(device == "cuda"))

    params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": a.lr},
              {"params": list(model.bnneck.parameters()) + [arc.W], "lr": a.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=a.wd)
    steps_total = a.epochs * len(sampler)
    steps_warm = a.warmup * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps_warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - steps_warm) / max(1, steps_total - steps_warm)))))
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    if full_fit:
        base = {"mAP": -1.0}
        log = []
    else:
        base = validate(model, d["query"], d["gallery"], d["has_match"], tf_eval, device, a.batch_size_eval, a.workers)
        print(format_metrics(base, "[val] до обучения"))
        log = [{"epoch": 0, **base}]
    best, best_ep, bad = base["mAP"], -1, 0
    torch.save(model.state(), out / "best.pt")
    step = 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y, cam in dl:
            t_step = time.perf_counter()
            x, y = x.to(device, non_blocking=True), y.to(device)
            cam = cam.to(device) if a.cross_cam_triplet else None
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
                f, fb = model.features(x)
            f, fb = f.float(), fb.float()
            loss = arc(fb, y) + a.tri_w * triplet_batch_hard(f, y, cams=cam)
            if teacher is not None:
                with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                                     enabled=(device == "cuda")):
                    _, tb = teacher.features(x)
                loss = loss + a.distill_w * similarity_distill(fb, tb.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 5.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            losses.append(loss.item())
            if throttle > 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                time.sleep(throttle * (time.perf_counter() - t_step))
        if full_fit:   # валидации нет: держим последнюю эпоху
            m = {"mAP": float(ep)}
            print(f"[ep {ep:02d}] loss={np.mean(losses):.3f} ({time.time() - t0:.0f} c)")
        else:
            m = validate(model, d["query"], d["gallery"], d["has_match"], tf_eval, device, a.batch_size_eval, a.workers)
            print(f"[ep {ep:02d}] loss={np.mean(losses):.3f} {format_metrics(m, '')} ({time.time() - t0:.0f} c)")
        log.append({"epoch": ep, "loss": float(np.mean(losses)), **m})
        if a.no_select:
            # Фиксированное число эпох, без отбора лучшей и без ранней остановки.
            # Нужно, чтобы валидационный двойник строился ТЕМ ЖЕ способом, что и релиз
            # (`--val-frac 0`, 18 эпох, никакого выбора эпохи). Иначе мы сравниваем модель,
            # отобранную по валидации, с моделью, которая отбора не видела, и разница между ними
            # — не свойство рецепта, а сам факт отбора.
            best, best_ep = m["mAP"], ep
            torch.save(model.state(), out / "best.pt")
        elif m["mAP"] > best:
            best, best_ep, bad = m["mAP"], ep, 0
            torch.save(model.state(), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[train] ранняя остановка: {a.patience} эпох без улучшения")
                break
        with open(out / "log.json", "w", encoding="utf-8") as f:
            json.dump({"args": vars(a), "best_mAP": best, "best_epoch": best_ep, "log": log}, f, ensure_ascii=False, indent=2)
    if full_fit:
        print(f"[train] финальная модель: {a.epochs} эпох на всех id → {out / 'best.pt'} (валидации не было)")
    elif a.no_select:
        print(f"[train] без отбора: сохранена эпоха {best_ep} из {a.epochs}, mAP на ней "
              f"{best * 100:.1f}% → {out / 'best.pt'} (метрика печаталась, но на выбор не влияла)")
    else:
        print(f"[train] лучший mAP={best * 100:.1f}% (эпоха {best_ep}) → {out / 'best.pt'}")


if __name__ == "__main__":
    main()
