"""Скорость инференса на CPU: мс на кроп при batch=1 и batch=16, число параметров.

Меряем то, что войдёт в критерий: препроцессинг (resize+normalize) + forward backbone.
Детектор (если кропы делаем сами) добавится отдельной строкой позже.
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image


def _param_count(backbone) -> int:
    """Модель лежит по-разному: у timm-бэкбона в .model, у дообученного в .m (ReIDModel)."""
    for attr in ("model", "m"):
        mod = getattr(backbone, attr, None)
        if mod is None:
            continue
        try:
            if hasattr(mod, "modules") and not hasattr(mod, "parameters"):   # ReIDModel: список подмодулей
                return sum(p.numel() for sub in mod.modules() for p in sub.parameters())
            return sum(p.numel() for p in mod.parameters())
        except Exception:
            pass
    return 0


def bench_backbone(backbone, threads: int = 4, n_iter: int = 50, warmup: int = 10,
                   img_size=(640, 480), batch: int = 16) -> dict:
    """Латентность batch=1 и пропускная способность при batch=N — то, что меряет жюри.
    Считается только препроцессинг + forward, без чтения кадра с диска."""
    import torch
    torch.set_num_threads(threads)
    cuda = getattr(backbone, "device", "cpu") == "cuda"
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (img_size[1], img_size[0], 3), dtype=np.uint8))
            for _ in range(batch)]
    res = {"backbone": backbone.name, "threads": threads, "dim": backbone.dim,
           "device": getattr(backbone, "device", "cpu"), "batch": batch,
           "params_M": round(_param_count(backbone) / 1e6, 1)}

    # batch=1: препроцессинг + forward
    for _ in range(warmup):
        backbone.embed(imgs[:1])
    sync()
    t = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        backbone.embed(imgs[:1]); sync()
        t.append(time.perf_counter() - t0)
    res["ms_per_image_b1"] = round(float(np.median(t)) * 1000, 2)

    # пакетный режим
    for _ in range(max(1, warmup // 4)):
        backbone.embed(imgs)
    sync()
    t = []
    for _ in range(max(1, n_iter // 4)):
        t0 = time.perf_counter()
        backbone.embed(imgs); sync()
        t.append(time.perf_counter() - t0)
    res["ms_per_image_b16"] = round(float(np.median(t)) * 1000 / batch, 2)
    res["fps_batch"] = round(1000 / res["ms_per_image_b16"], 1)

    # только препроцессинг, чтобы видеть его долю
    t = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        backbone.transform(imgs[0])
        t.append(time.perf_counter() - t0)
    res["ms_preprocess"] = round(float(np.median(t)) * 1000, 2)
    return res


def format_bench(r: dict) -> str:
    return (f"{r['backbone']}: {r['params_M']} M параметров, D={r['dim']}, {r['device']}, "
            f"threads={r['threads']} | batch=1: {r['ms_per_image_b1']} мс/кроп "
            f"(препроцессинг {r['ms_preprocess']} мс) | batch={r['batch']}: "
            f"{r['ms_per_image_b16']} мс/кроп = {r['fps_batch']} кроп/с")
