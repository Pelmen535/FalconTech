"""Дообучение backbone под данные организаторов (metric learning по vehicle_id).

  python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b --epochs 20 --out weights/dinov2_b_ft

Что внутри:
  * данные: fit-часть локальной open-set валидации (см. hackathon_data.build_local_validation);
    val-часть — для ранней остановки по mAP@10 по правилу жюри (см. validate);
    --val-frac 0 — финальный режим: все id в обучении, валидации нет, сохраняется последняя эпоха;
  * кропы режутся один раз в кэш с запасом 30% (crops_cache/), дальше — случайный джиттер bbox
    внутри запаса: это дёшево и даёт устойчивость к неточным bbox;
    полный путь кэша — <runs_dir>/<name>/crops_cache/;
  * аугментации под камеры: сдвиг/масштаб bbox, цвет, grayscale (ночь), понижение разрешения + JPEG
    (плохие камеры), гауссов блюр, random erasing (перекрытия), флип;
    маска зоны номера — как на инференсе;
  * модель: timm backbone → cls+mean → BNNeck; лоссы: ArcFace (по id) + triplet batch-hard
    + опциональная дистилляция геометрии сходств от учителя (--distill);
    PK-сэмплер (P машин × K кадров в батче), --cam-aware: K кадров машины с разных камер;
  * веса сохраняются в <out>/best.pt и грузятся через get_backbone("ft:<out>/best.pt").
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
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
    а bbox — положение исходного bbox внутри кропа (чтобы джиттер знал, где машина).

    pad — доля от БОЛЬШЕЙ стороны bbox, добавляется со всех сторон и подрезается границами кадра
    (у края запас меньше, bbox внутри кропа смещается); quality — JPEG-качество кэша.
    Кэш переиспользуется по имени файла: при изменении pad/quality каталог кэша надо удалить
    вручную, иначе кропы останутся от прежних настроек."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # группируем по кадру, чтобы открыть каждый JPEG ровно один раз (иначе кадр с 20 машинами
    # читается 20 раз), а индекс i храним, чтобы вернуть записи в ИСХОДНОМ порядке — на него
    # завязаны метки и метрики
    by_frame: dict[str, list[tuple[int, Record]]] = defaultdict(list)
    for i, r in enumerate(records):
        by_frame[r.path].append((i, r))
    by_index: list[Record | None] = [None] * len(records)
    for path, items in by_frame.items():
        # кэш адресуется только по key — pad/quality в имя файла не входят; меняешь pad —
        # удаляй crops_cache целиком, иначе кропы старые, а bbox пересчитан по новому pad
        need = [(i, r) for i, r in items
                if not (cache_dir / f"{_safe_filename(r.key)}.jpg").exists()]
        if need:
            # закрываем именно исходный файл: convert("RGB") всё равно возвращает новый объект
            with Image.open(path) as src:
                im = src.convert("RGB")
                W, H = im.size
                for i, r in need:
                    x, y, w, h = r.bbox
                    p = pad * max(w, h)
                    x0, y0 = _pad_origin(r.bbox, pad)
                    x1, y1 = min(W, int(x + w + p)), min(H, int(y + h + p))
                    im.crop((x0, y0, x1, y1)).save(cache_dir / f"{_safe_filename(r.key)}.jpg",
                                                   quality=quality)
        for i, r in items:
            x, y, w, h = r.bbox
            x0, y0 = _pad_origin(r.bbox, pad)
            by_index[i] = Record(str(cache_dir / f"{_safe_filename(r.key)}.jpg"), r.vid, r.cam,
                                 (x - x0, y - y0, w, h), r.key)
    return by_index


def _pad_origin(bbox, pad) -> tuple[int, int]:
    """Левый верхний угол кропа с запасом pad. Одна формула и для нарезки кропа, и для пересчёта
    bbox внутрь него: разойдутся эти два места — bbox «поедет» относительно картинки."""
    x, y, w, h = bbox
    p = pad * max(w, h)
    return max(0, int(x - p)), max(0, int(y - p))


def _safe_filename(key: str) -> str:
    """Ключ записи → имя файла кэша: в ключе бывают слеши и двоеточия из пути кадра.
    Преобразование неинъективно — уникальность имени держится на уникальности Record.key."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


# --------------------------------------------------------------------------- аугментации
class TrainTransform:
    """Аугментации обучения. __call__(im, bbox, mask_frac) → нормализованный тензор
    3×img_size×img_size; bbox задан в координатах кэш-кропа, mask_frac — в долях кропа.
    strong=False (--no-strong-aug) оставляет только джиттер bbox и флип, отключая цвет,
    grayscale, даунскейл+JPEG, блюр и random erasing."""

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

    def _jitter_crop(self, im: Image.Image, bbox) -> Image.Image:
        """Джиттер bbox внутри запаса, нарезанного prepare_crops: шаг вынесен из __call__
        как есть, без единой перестановки обращений к random."""
        x, y, w, h = bbox
        # джиттер bbox: масштаб 0.9–1.2, сдвиг ±10%
        # порядок обращений к random (scale → dx → dy) фиксирует обучающую выборку —
        # не переставлять, не объединять два uniform в один вызов, не писать dy раньше dx
        scale = random.uniform(0.9, 1.2)
        dx, dy = random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h
        cx, cy = x + w / 2 + dx, y + h / 2 + dy
        crop_w, crop_h = w * scale, h * scale
        # за границы выходим намеренно — PIL добивает чёрным, это аугментация неполного кадра;
        # clamp (max(0,...)/min(W,...)) и замена int() на round() меняют каждый кроп
        return im.crop((int(cx - crop_w / 2), int(cy - crop_h / 2),
                        int(cx + crop_w / 2), int(cy + crop_h / 2)))

    def _degrade(self, crop: Image.Image) -> Image.Image:
        """«Плохая камера»: цвет → grayscale → даунскейл+JPEG → блюр. Порядок и число обращений
        к random несущие — они определяют, какие кадры получит модель."""
        crop = self.color(crop)
        if random.random() < 0.1:
            crop = crop.convert("L").convert("RGB")
        if random.random() < 0.3:  # плохая камера: даунскейл + jpeg
            downscale = random.uniform(0.3, 0.7)
            small_w = max(8, int(crop.width * downscale))
            small_h = max(8, int(crop.height * downscale))
            small = crop.resize((small_w, small_h), Image.BILINEAR)
            buf = io.BytesIO()
            small.save(buf, "JPEG", quality=random.randint(30, 70))
            buf.seek(0)
            crop = Image.open(buf).convert("RGB")
        if random.random() < 0.15:
            crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return crop

    def __call__(self, im: Image.Image, bbox, mask_frac=None) -> "torch.Tensor":
        crop = self._jitter_crop(im, bbox)
        # random.random() специально последним: при mask_p=0 (а релиз обучен с mask_p=0)
        # генератор не расходуется вовсе; если проверку вероятности поставить первой,
        # сместится весь поток аугментаций на каждом сэмпле
        if mask_frac and self.mask_p and random.random() < self.mask_p:
            from .plate import apply_mask_frac
            crop = apply_mask_frac(crop, mask_frac)
        if random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        if self.strong:
            crop = self._degrade(crop)
        crop = crop.resize((self.size, self.size), Image.BICUBIC)
        tensor = self.norm(crop)
        return self.erase(tensor) if self.strong else tensor


class EvalTransform:
    """Валидационный/инференсный препроцесс: crop_bbox с тем же pad, что в конфиге инференса —
    иначе валидация мерит не то, что сдаётся. Маска зоны номера приходит АРГУМЕНТОМ вызова
    (из CropDataset), не через конструктор."""

    def __init__(self, img_size: int, mean, std, pad: float):
        import torchvision.transforms as T
        self.size = img_size
        self.pad = pad
        self.norm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __call__(self, im: Image.Image, bbox, mask_frac=None):
        crop = crop_bbox(im, bbox, self.pad, mask_frac) if bbox is not None else im
        return self.norm(crop.resize((self.size, self.size), Image.BICUBIC))


class CropDataset:
    """Датасет кропов. __getitem__ → (тензор, метка, camera_id). Если label_map не задан
    (валидация, извлечение эмбеддингов), метка = -1 — она там не используется, но DataLoader
    должен что-то вернуть."""

    def __init__(self, records: list[Record], transform, label_map: dict | None = None, mask=None):
        self.records = records
        self.transform = transform
        self.label_map = label_map
        self.mask = mask          # {"cache": ..., "mode": ...} из vreid/plate.py или None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        mask_frac = None
        if self.mask:
            from .plate import boxes_frac
            # Сюда приходят записи ПОСЛЕ prepare_crops: r.path — файл кропа, r.bbox — координаты
            # внутри кропа, а кеш построен по исходным кадрам (plate.cache_key = stem(path) +
            # bbox кадра). Поиск всегда промахивается, боксы пустые — аугментация «номер закрыт»
            # не срабатывает. Релиз обучен с mask_p=0, поэтому весов это не касается; чинить
            # ключ — смена рецепта обучения, только с перезапуском.
            mask_frac = boxes_frac(self.mask["cache"], r.path, r.bbox, self.mask["mode"])
        with Image.open(r.path) as im:
            x = self.transform(im.convert("RGB"), r.bbox, mask_frac)
        y = self.label_map[r.vid] if self.label_map else -1
        return x, y, r.cam


class PKSampler:
    """Батч = P машин × K кадров каждой. Машины с < K кадрами добираются повтором."""

    def __init__(self, per_record_labels: list[int], P: int, K: int, seed: int = 0):
        self.by_label = defaultdict(list)
        for idx, label in enumerate(per_record_labels):
            self.by_label[label].append(idx)
        self.P, self.K = P, K
        # rng создаётся один раз и НЕ пересеивается в __iter__: состояние переносится между
        # эпохами, иначе каждая эпоха повторяла бы один и тот же состав батчей
        self.rng = random.Random(seed)
        # число батчей считается по количеству кадров, не по числу уникальных машин
        self.n_batches = max(1, len(per_record_labels) // (P * K))

    def __iter__(self):
        # порядок = порядок первого появления id в fit; rng.sample зависит от него,
        # sorted() поменяет состав всех батчей на всём обучении
        labels = list(self.by_label)
        for _ in range(self.n_batches):
            chosen = self.rng.sample(labels, min(self.P, len(labels)))
            batch = []
            for label in chosen:
                frames = self.by_label[label]
                # sample и choice расходуют rng по-разному — ветки не сливать
                if len(frames) >= self.K:
                    batch += self.rng.sample(frames, self.K)
                else:  # кадров меньше K — добираем повтором
                    batch += [self.rng.choice(frames) for _ in range(self.K)]
            yield batch

    def __len__(self):
        return self.n_batches


class CamAwarePKSampler(PKSampler):
    """То же P×K, но K кадров машины берутся с РАЗНЫХ камер по кругу: сначала по одному кадру
    с каждой камеры, потом добор. Тогда «трудный позитив» в батче почти всегда кросс-камерный —
    ровно то, что считает жюри. Машины с одной камерой ведут себя как в обычном сэмплере."""

    def __init__(self, per_record_labels: list[int], cams: list[int], P: int, K: int,
                 seed: int = 0):
        super().__init__(per_record_labels, P, K, seed)
        self.by_label_cam: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for idx, (label, cam) in enumerate(zip(per_record_labels, cams)):
            self.by_label_cam[label][cam].append(idx)

    def _pick(self, label: int) -> list[int]:
        # камеры идут в порядке первого появления в данных; shuffle групп и shuffle внутри
        # групп расходуют rng в этом порядке — sorted() сместит поток случайных чисел
        frames_by_cam = [list(v) for v in self.by_label_cam[label].values()]
        for cam_frames in frames_by_cam:
            self.rng.shuffle(cam_frames)
        self.rng.shuffle(frames_by_cam)
        picked, round_idx = [], 0
        while len(picked) < self.K and any(frames_by_cam):
            cam_frames = frames_by_cam[round_idx % len(frames_by_cam)]
            if cam_frames:
                # pop() без аргумента — с конца уже перемешанного списка; pop(0) или обход
                # списком дадут другие кадры
                picked.append(cam_frames.pop())
            round_idx += 1
            # страховка от зацикливания, и она достижима: у машины с числом камер больше ~10·K
            # добор уходит в rng.choice ниже, то есть расход генератора другой.
            # Не удалять как «мёртвую»
            if round_idx > 10 * self.K:
                break
        while len(picked) < self.K:  # добор повтором, если кадров меньше K
            picked.append(self.rng.choice(self.by_label[label]))
        return picked

    def __iter__(self):
        # копия базового __iter__ намеренно: порядок обращений к rng (sample по машинам →
        # выбор кадров внутри машины) должен совпадать побитово в обоих сэмплерах
        # порядок = порядок первого появления id в fit; rng.sample зависит от него,
        # sorted() поменяет состав всех батчей на всём обучении
        labels = list(self.by_label)
        for _ in range(self.n_batches):
            chosen = self.rng.sample(labels, min(self.P, len(labels)))
            batch = []
            for label in chosen:
                batch += self._pick(label)
            yield batch


# --------------------------------------------------------------------------- модель и лоссы
def build_model(model_name: str, img_size: int, pretrained: bool = True):
    """Создаёт timm-backbone без классификатора (num_classes=0) и возвращает (backbone, mean, std)
    из его data_config — нормализация должна совпадать с предобучением."""
    import timm
    kwargs = {"pretrained": pretrained, "num_classes": 0}
    # у ViT позиционные эмбеддинги привязаны к родному разрешению; img_size + dynamic_img_size
    # заставляют timm интерполировать их под наш вход, иначе модель либо падает, либо считает
    # признаки не по тем позициям
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
        # у CNN нет cls_token, отсюда проверка
        self.is_vit = hasattr(self.backbone, "forward_features") and hasattr(self.backbone, "cls_token")
        # ×2 для ViT — конкатенация cls-токена и среднего по патчам (см. features)
        feat_dim = self.backbone.num_features * (2 if self.is_vit else 1)
        self.bnneck = nn.BatchNorm1d(feat_dim)
        # BNNeck: bias у BN отключён намеренно — он сдвигает эмбеддинг целиком и мешает
        # косинусной метрике, по которой идёт поиск; scale (weight) при этом учится.
        # Часть рецепта, не убирать
        self.bnneck.bias.requires_grad_(False)
        self.dim = feat_dim

    def features(self, x):
        """Возвращает (f, fb): f — признак ДО BNNeck (по нему считается triplet), fb — ПОСЛЕ
        BNNeck (по нему считается ArcFace, он же идёт в индекс на инференсе). Для ViT
        f = concat(cls, среднее по патч-токенам), поэтому dim = 2×num_features; для CNN —
        pooled-выход."""
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
        # weights_only=False нужен потому, что в чекпоинте кроме тензоров лежат метаданные
        # (model_name, img_size, mean/std) — без них не восстановить ни архитектуру,
        # ни нормализацию. Файл свой собственный, из <out>/best.pt
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ReIDModel(ck["model_name"], ck["img_size"], pretrained=False)
        m.backbone.load_state_dict(ck["backbone"])
        m.bnneck.load_state_dict(ck["bnneck"])
        return m.to(device)


class ArcFace:
    """ArcFace по id машины. __call__(f, y) → скаляр; f — признаки ПОСЛЕ BNNeck, y — номера
    классов из label_map. s — масштаб логитов, m — угловой отступ в радианах. W — отдельный
    nn.Parameter, не завёрнутый в Module: его вручную кладут в группу параметров оптимизатора
    (см. main) и вручную обнуляют градиент в find_batch_P."""

    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.3, device="cpu",
                 label_smoothing: float = 0.0):
        import torch
        import torch.nn as nn
        self.W = nn.Parameter(torch.empty(n_classes, dim, device=device))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m, self.label_smoothing = s, m, label_smoothing

    def __call__(self, f, y):
        import torch
        import torch.nn.functional as F
        # зажим обязателен: градиент acos на ±1 бесконечен, без него ArcFace даёт NaN на первых
        # же шагах, особенно в смешанной точности. Margin применяется БЕЗ стража theta > pi - m
        # и без easy_margin — сдаваемая модель обучена именно этой формой; «сверка с референсной
        # реализацией» изменит функцию потерь
        cos = F.linear(F.normalize(f), F.normalize(self.W)).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)


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
    # cdist, а не (2 - 2·f·fᵀ).sqrt(): арифметика расходится в последних разрядах, а от неё
    # зависит, какая пара окажется самой трудной — при замене расходятся уже первые шаги
    dist = torch.cdist(f, f)
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos_mask = same & ~eye
    if cams is not None:
        cross = pos_mask & (cams[:, None] != cams[None, :])
        has_cross = cross.any(dim=1, keepdim=True)
        pos_mask = torch.where(has_cross, cross, pos_mask)
    # заглушки подобраны под L2-нормированные признаки: расстояние лежит в [0, 2], поэтому -1
    # гарантированно проигрывает max (трудный позитив), а 9.0 — min (трудный негатив).
    # ±inf нельзя: строка без пары даст NaN в лоссе и GradScaler уйдёт в бесконечный пропуск шагов
    pos = dist.masked_fill(~pos_mask, -1).max(dim=1).values
    neg = dist.masked_fill(same, 9.0).min(dim=1).values
    return F.relu(pos - neg + margin).mean()


# --------------------------------------------------------------------------- автоподбор батча
def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ in ("OutOfMemoryError",)


# пробные шаги идут на ЖИВОЙ модели в train(): они обновляют running-статистики BNNeck и
# расходуют CUDA-генератор — это часть состояния, с которым стартует обучение. Оборачивать
# в no_grad или переводить в eval «раз память только меряем» нельзя: это меняет веса,
# и число пробных шагов зависит от свободной VRAM
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
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
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
        # шаг вниз на 20%; если мы уже на p_min и он не прошёл — ставим p_min-1 только чтобы
        # провалить условие while и выйти, наружу всё равно уходит p_min (строка ниже)
        P = max(p_min, int(P * 0.8)) if P > p_min else p_min - 1
    return p_min


# --------------------------------------------------------------------------- валидация
def embed_records(model: ReIDModel, records: list[Record], transform, device, batch_size=64,
                  workers=4):
    """Считает эмбеддинги записей в ТОМ ЖЕ порядке (строка i ↔ records[i]) — в validate() к ним
    применяются маски и метки, построенные по исходному порядку, поэтому shuffle=False здесь
    обязателен. Возвращает (N, dim) float32, векторы L2-нормированы — дальше скалярное
    произведение = косинус."""
    import torch
    from torch.utils.data import DataLoader
    # каждый DataLoader тянет base_seed из глобального torch-генератора и им сеет random
    # в воркерах — от ЧИСЛА созданных загрузчиков зависят все аугментации; число и порядок
    # вызовов validate (включая предобучающий), порядок query/gallery, num_workers
    # и отсутствие persistent_workers — несущие
    loader = DataLoader(CropDataset(records, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=(device == "cuda")):
        for x, _, _ in loader:
            _, fb = model.features(x.to(device))
            out.append(torch.nn.functional.normalize(fb.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def validate(model, query: Split, gallery: Split, has_match, transform, device, batch_size,
             workers) -> dict:
    """query/gallery — сплиты запросов и галереи, has_match — маска запросов, у которых в галерее
    есть хотя бы один кадр той же машины (остальные в open-set не оцениваются). Возвращает
    словарь метрик из metrics.evaluate."""
    query_emb = embed_records(model, query.records, transform, device, batch_size, workers)
    gallery_emb = embed_records(model, gallery.records, transform, device, batch_size, workers)
    sims = query_emb @ gallery_emb.T
    # Правило жюри: убираем только пары «тот же vehicle_id И та же camera_id», и считаем mAP@10.
    # Раньше здесь стоял строгий cross-camera режим без cutoff — то есть лучшая эпоха выбиралась
    # по метрике, которая завышает результат и не совпадает с оцениваемой.
    return evaluate(sims[has_match], query.vids[has_match], query.cams[has_match],
                    gallery.vids, gallery.cams, cutoff=10)


# --------------------------------------------------------------------------- обучение
def train_one_epoch(model, loader, arc, teacher, opt, sched, scaler, params, args, device,
                    throttle) -> list[float]:
    """Одна эпоха: тело цикла вынесено из main как есть, без перестановок. Порядок внутри шага
    (autocast → fp32 → scaler.step → scaler.update → sched.step) несущий. Возвращает лоссы
    по шагам — среднее по ним печатается и уходит в log.json."""
    import torch
    model.train()
    losses = []
    # клиппим по общей норме всех обучаемых параметров, включая W из ArcFace: W не принадлежит
    # ни одному модулю и в model.parameters() не попадает. Список собран один раз — объекты
    # и их порядок те же, что при пересборке на каждом шаге
    all_params = [p for g in params for p in g["params"]]
    for x, y, cam in loader:
        t_step = time.perf_counter()
        x, y = x.to(device, non_blocking=True), y.to(device)
        # None здесь — не заглушка, а переключатель режима: с тензором cams трудный позитив
        # ищется только кросс-камерный (см. triplet_batch_hard). Безусловный cam.to(device)
        # сменит функцию потерь
        cam = cam.to(device) if args.cross_cam_triplet else None
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            enabled=(device == "cuda")):
            f, fb = model.features(x)
        # лоссы считаем в fp32 вне autocast: acos в ArcFace и cdist в triplet в fp16 срываются
        # в NaN, а выигрыш по памяти даёт forward backbone, а не эти две операции.
        # Та же причина у аналогичных мест в find_batch_P и у признаков учителя
        f, fb = f.float(), fb.float()
        # ArcFace считается по fb (ПОСЛЕ BNNeck), triplet — по f (ДО него): это и есть BNNeck,
        # а не опечатка. Порядок слагаемых не менять и не собирать через sum([...]) — сложение
        # float некоммутативно по округлению, а под GradScaler это сдвигает и точки пропуска шага
        loss = arc(fb, y) + args.tri_w * triplet_batch_hard(f, y, cams=cam)
        if teacher is not None:
            with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                                 enabled=(device == "cuda")):
                _, tb = teacher.features(x)
            loss = loss + args.distill_w * similarity_distill(fb, tb.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        # порог 5.0 имеет смысл только ПОСЛЕ scaler.unscale_: до него градиенты отмасштабированы
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        # sched.step() вызывается и на шагах, которые GradScaler пропустил из-за inf/nan:
        # график lr привязан к числу итераций, а не к числу применённых шагов — не «чинить»
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        if throttle > 0:
            if device == "cuda":
                torch.cuda.synchronize()
            time.sleep(throttle * (time.perf_counter() - t_step))
    return losses


def build_parser() -> argparse.ArgumentParser:
    """Описание CLI вынесено из main, чтобы тело main начиналось с логики; порядок add_argument
    несущий — от него зависит порядок ключей vars(args) в log.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", default="dinov2_b", help="dinov2_s|dinov2_b|dinov2_l|clip_b|resnet50|<timm name>")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--P", default="auto", help="машин в батче; auto — подобрать максимум, влезающий в память GPU")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help="lr backbone")
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=1, help="эпох разогрева")
    parser.add_argument("--freeze-blocks", type=int, default=0, help="заморозить первые N блоков ViT")
    parser.add_argument("--arc-m", type=float, default=0.3)
    parser.add_argument("--arc-ls", type=float, default=0.0,
                        help="label smoothing в ArcFace; 0.1 против переобучения (loss уходит в 0.1 к 12-й эпохе)")
    parser.add_argument("--val-frac", type=float, default=None,
                        help="переопределить dataset.val_frac; 0 — обучение на ВСЕХ id без валидации (финальная модель)")
    parser.add_argument("--mask-p", type=float, default=0.0,
                        help="вероятность закрасить зону анонимизации номера при обучении (0 — не закрывать). "
                             "Учит не опираться на эту зону, при этом на инференсе закрывать ничего не нужно")
    parser.add_argument("--plate-cache", default="results/plate_boxes.json",
                        help="кеш боксов зоны анонимизации (см. scripts/build_plate_cache.py)")
    parser.add_argument("--tri-w", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-select", action="store_true",
                        help="ровно --epochs эпох, без выбора лучшей и без ранней остановки "
                             "(для валидационного двойника релиза, который обучен с --val-frac 0)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    # строка, а не store_true: по умолчанию веса предобученные, а --pretrained false должен
    # отключать их явно; store_true перевернёт умолчание, type=bool не различит "false".
    # Значение уходит в log.json как есть
    parser.add_argument("--pretrained", default="true")
    parser.add_argument("--no-strong-aug", action="store_true")
    parser.add_argument("--distill", default=None,
                        help="путь к весам учителя (weights/.../best.pt): переносим геометрию сходств")
    parser.add_argument("--distill-w", type=float, default=1.0, help="вес слагаемого дистилляции")
    parser.add_argument("--cam-aware", action="store_true",
                        help="K кадров машины брать с разных камер (кросс-камерные позитивы в батче)")
    parser.add_argument("--cross-cam-triplet", action="store_true",
                        help="в triplet трудный позитив искать только с другой камеры")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--throttle", type=float, default=None,
                        help="доля паузы после каждого шага (0.15 = GPU занят ~85%%), чтобы компьютер не лагал; "
                             "по умолчанию из переменной окружения VREID_THROTTLE")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # or 0 — на случай VREID_THROTTLE="" (переменная задана пустой): float("") падает
    # с ValueError ещё до загрузки данных
    throttle = args.throttle if args.throttle is not None else float(os.environ.get("VREID_THROTTLE", "0") or 0)
    if throttle > 0:
        print(f"[train] throttle={throttle:.2f}: после каждого шага пауза {throttle * 100:.0f}% от времени шага")

    import torch
    from torch.utils.data import DataLoader
    from .models import _REGISTRY
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(args.out or f"weights/{cfg['name']}_{args.backbone}_{args.img_size}")
    out.mkdir(parents=True, exist_ok=True)

    if args.val_frac is not None:
        cfg["dataset"] = {**cfg["dataset"], "val_frac": args.val_frac}
    # 0.2 обязано совпадать с умолчанием val_frac в hackathon_data.split_open_set — иначе
    # full_fit и фактическое разбиение разойдутся
    full_fit = float(cfg["dataset"].get("val_frac", 0.2)) <= 0
    if full_fit:
        from .hackathon_data import read_annotations
        ds_cfg = cfg["dataset"]
        root = Path(ds_cfg["root"])
        recs = read_annotations(root / ds_cfg.get("train_csv", "train.csv"),
                                root / ds_cfg.get("images_dir", "images"),
                                ds_cfg.get("cols"), ds_cfg.get("bbox_format", "xywh"))
        splits = {"fit": Split("fit", recs), "query": None, "gallery": None, "has_match": None}
        print(f"[train] ФИНАЛЬНЫЙ режим: все {len({r.vid for r in recs})} id в обучении, валидации нет — "
              f"сохраняется последняя эпоха, число эпох задай по лучшему прогону с валидацией")
    else:
        splits = build_local_validation(cfg["dataset"])
    crop_cfg = cfg.get("crop", {})
    cache = Path(cfg.get("runs_dir", "runs")) / cfg["name"] / "crops_cache"
    t0 = time.time()
    # запас 30% здесь намеренно больше crop.pad инференса (0.08): внутри этого запаса живёт
    # джиттер bbox в TrainTransform (масштаб до 1.2, сдвиг ±10%). crop.pad из конфига —
    # только для валидации
    fit = prepare_crops(splits["fit"].records, cache / "fit")
    print(f"[train] кропы fit: {len(fit)} ({time.time() - t0:.0f} c)")
    # sorted фиксирует соответствие vehicle_id → номер класса ArcFace: порядок множества зависит
    # от хеширования, а от номера класса зависит строка матрицы W. Без сортировки два запуска
    # на одних данных дают разные метки и несравнимые веса
    ids = sorted({r.vid for r in fit})
    label_map = {v: i for i, v in enumerate(ids)}

    model_name = _REGISTRY.get(args.backbone, {}).get("model_name", args.backbone)
    model = ReIDModel(model_name, args.img_size, pretrained=(args.pretrained.lower() == "true")).to(device)
    if args.freeze_blocks and hasattr(model.backbone, "blocks"):
        for p in model.backbone.patch_embed.parameters():
            p.requires_grad_(False)
        for blk in model.backbone.blocks[: args.freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad_(False)
    arc = ArcFace(model.dim, len(ids), m=args.arc_m, device=device, label_smoothing=args.arc_ls)
    print(f"[train] {model_name}, D={model.dim}, id={len(ids)}, device={device}")

    teacher = None
    if args.distill:
        teacher = ReIDModel.load(args.distill, device)
        for sub in teacher.modules():
            sub.eval()
            for prm in sub.parameters():
                prm.requires_grad_(False)
        if teacher.img_size != args.img_size:
            raise SystemExit(f"учитель обучен на входе {teacher.img_size}, ученик на {args.img_size} — "
                             f"нужен одинаковый вход, иначе кропы не совпадут")
        print(f"[train] дистилляция от {args.distill}: D учителя {teacher.dim}, вес {args.distill_w}")

    if str(args.P).lower() == "auto":
        # пишем обратно в args.P намеренно: vars(args) уходит в log.json, и там должен стоять
        # РАЗРЕШЁННЫЙ P — при --P auto он зависит от свободной памяти GPU и иначе прогон
        # не восстановим
        args.P = find_batch_P(model, arc, args.img_size, args.K, device, teacher=teacher)
    else:
        args.P = int(args.P)

    tf_train = TrainTransform(args.img_size, model.mean, model.std, float(args.mask_p), strong=not args.no_strong_aug)
    tf_eval = EvalTransform(args.img_size, model.mean, model.std, crop_cfg.get("pad", 0.08))
    mask = None
    if args.mask_p > 0:
        from .plate import load_cache
        mask = {"cache": load_cache(args.plate_cache), "mode": "det"}
        print(f"[train] аугментация «номер закрыт» с p={args.mask_p}, боксов в кеше {len(mask['cache'])}")
    ds = CropDataset(fit, tf_train, label_map, mask=mask)
    if args.cam_aware:
        sampler = CamAwarePKSampler([label_map[r.vid] for r in fit], [r.cam for r in fit], args.P, args.K, args.seed)
        print("[train] сэмплер: camera-aware (кадры одной машины — с разных камер)")
    else:
        sampler = PKSampler([label_map[r.vid] for r in fit], args.P, args.K, args.seed)
    # каждый DataLoader тянет base_seed из глобального torch-генератора и им сеет random
    # в воркерах — от ЧИСЛА созданных загрузчиков зависят все аугментации; число и порядок
    # вызовов validate (включая предобучающий), порядок query/gallery, num_workers
    # и отсутствие persistent_workers — несущие
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=(device == "cuda"))

    params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
              {"params": list(model.bnneck.parameters()) + [arc.W], "lr": args.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=args.wd)
    steps_total = args.epochs * len(sampler)
    steps_warm = args.warmup * len(sampler)
    # lr по ШАГАМ, не по эпохам: линейный разогрев первые warmup эпох (переведённых в шаги
    # строкой выше), затем косинус до нуля к steps_total. max(1, ...) — защита от деления
    # на ноль при --warmup 0 и при epochs == warmup

    def lr_factor(step_idx: int) -> float:
        warm = min(1.0, (step_idx + 1) / max(1, steps_warm))
        progress = min(1.0, max(0, step_idx - steps_warm) / max(1, steps_total - steps_warm))
        return warm * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    if full_fit:
        base = {"mAP": -1.0}
        log = []
    else:
        base = validate(model, splits["query"], splits["gallery"], splits["has_match"], tf_eval,
                        device, args.batch_size_eval, args.workers)
        print(format_metrics(base, "[val] до обучения"))
        log = [{"epoch": 0, **base}]
    best = base["mAP"]
    best_epoch = -1
    epochs_without_gain = 0
    # сохраняем ещё до обучения: тогда best.pt существует всегда — и при обрыве прогона,
    # и если ни одна эпоха не побьёт стартовую метрику; get_backbone("ft:<out>/best.pt")
    # не останется без файла
    torch.save(model.state(), out / "best.pt")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = train_one_epoch(model, loader, arc, teacher, opt, sched, scaler, params, args,
                                 device, throttle)
        if full_fit:   # валидации нет: держим последнюю эпоху
            # в mAP кладётся НОМЕР ЭПОХИ — так условие улучшения ниже истинно каждый раз
            # и best.pt всегда хранит последнюю эпоху. Внимание: в этом режиме поле mAP
            # в log.json — номер эпохи, а не метрика
            metrics = {"mAP": float(epoch)}
            print(f"[ep {epoch:02d}] loss={np.mean(losses):.3f} ({time.time() - t0:.0f} c)")
        else:
            metrics = validate(model, splits["query"], splits["gallery"], splits["has_match"],
                               tf_eval, device, args.batch_size_eval, args.workers)
            print(f"[ep {epoch:02d}] loss={np.mean(losses):.3f} {format_metrics(metrics, '')} "
                  f"({time.time() - t0:.0f} c)")
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if args.no_select:
            # Фиксированное число эпох, без отбора лучшей и без ранней остановки.
            # Нужно, чтобы валидационный двойник строился ТЕМ ЖЕ способом, что и релиз
            # (`--val-frac 0`, 18 эпох, никакого выбора эпохи). Иначе мы сравниваем модель,
            # отобранную по валидации, с моделью, которая отбора не видела, и разница между ними
            # — не свойство рецепта, а сам факт отбора.
            best, best_epoch = metrics["mAP"], epoch
            torch.save(model.state(), out / "best.pt")
        elif metrics["mAP"] > best:
            best, best_epoch, epochs_without_gain = metrics["mAP"], epoch, 0
            torch.save(model.state(), out / "best.pt")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"[train] ранняя остановка: {args.patience} эпох без улучшения")
                break
        # запись внутри цикла и после проверок намеренно: при ранней остановке break выходит
        # до сброса, и эпоха-триггер в log.json не попадает — так же, как в уже сданных
        # журналах. Выносить из цикла нельзя
        with open(out / "log.json", "w", encoding="utf-8") as log_file:
            report = {"args": vars(args), "best_mAP": best, "best_epoch": best_epoch, "log": log}
            json.dump(report, log_file, ensure_ascii=False, indent=2)
    if full_fit:
        print(f"[train] финальная модель: {args.epochs} эпох на всех id → {out / 'best.pt'} (валидации не было)")
    elif args.no_select:
        print(f"[train] без отбора: сохранена эпоха {best_epoch} из {args.epochs}, mAP на ней "
              f"{best * 100:.1f}% → {out / 'best.pt'} (метрика печаталась, но на выбор не влияла)")
    else:
        print(f"[train] лучший mAP={best * 100:.1f}% (эпоха {best_epoch}) → {out / 'best.pt'}")


if __name__ == "__main__":
    main()
