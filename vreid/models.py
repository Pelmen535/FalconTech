"""Backbone'ы для извлечения эмбеддингов.

Интерфейс один: Backbone.embed(list[PIL.Image]) -> np.ndarray [N, D], L2-нормированный.

Доступные имена (get_backbone):
  dummy       — гистограмма цвета, без нейросети; только для тестов стенда на CPU
  dinov2_s    — DINOv2 ViT-S/14 (timm, веса с Hugging Face), D=384*2
  dinov2_b    — DINOv2 ViT-B/14, D=768*2
  clip_b      — CLIP ViT-B/16 (OpenAI), image tower, D=768
  resnet50    — ResNet-50 ImageNet, D=2048
Для DINOv2 эмбеддинг = concat(CLS, mean(patch tokens)) — так на re-ID обычно лучше, чем один CLS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class Backbone:
    name: str = "base"
    dim: int = 0

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, img: Image.Image):
        """Препроцессинг одного изображения → тензор (для DataLoader)."""
        raise NotImplementedError

    def embed_tensors(self, batch) -> np.ndarray:
        raise NotImplementedError


def l2n(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ---------------------------------------------------------------------------
class DummyBackbone(Backbone):
    """HSV-гистограмма + грубая сетка яркости. Нужен только чтобы гонять стенд без GPU."""
    name = "dummy"

    def __init__(self, bins: int = 8, grid: int = 4):
        self.bins, self.grid = bins, grid
        self.dim = bins * 3 + grid * grid

    def transform(self, img: Image.Image):
        import torch
        img = img.convert("RGB").resize((64, 64))
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        feats = [np.histogram(hsv[..., c], bins=self.bins, range=(0, 1))[0] for c in range(3)]
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        cell = 64 // self.grid
        grid = gray.reshape(self.grid, cell, self.grid, cell).mean(axis=(1, 3)).ravel()
        v = np.concatenate([np.concatenate(feats).astype(np.float32) / (64 * 64), grid])
        return torch.from_numpy(v.astype(np.float32))

    def embed_tensors(self, batch) -> np.ndarray:
        return l2n(batch.cpu().numpy())

    def embed(self, images):
        import torch
        return self.embed_tensors(torch.stack([self.transform(i) for i in images]))


# ---------------------------------------------------------------------------
class TimmBackbone(Backbone):
    """Любая timm-модель как экстрактор признаков."""

    def __init__(self, model_name: str, name: str, device: str | None = None,
                 img_size: int | None = None, pool: str = "cls+mean", half: bool = True):
        import timm
        import torch

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"pretrained": True, "num_classes": 0}
        if img_size is not None:
            kwargs["img_size"] = img_size
            kwargs["dynamic_img_size"] = True
        self.model = timm.create_model(model_name, **kwargs).eval().to(self.device)
        self.half = half and str(self.device).startswith("cuda")
        if self.half:
            self.model = self.model.half()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        if img_size is not None:
            cfg["input_size"] = (3, img_size, img_size)
        self._tf = timm.data.create_transform(**cfg, is_training=False)
        self.pool = pool
        self.is_vit = hasattr(self.model, "forward_features") and hasattr(self.model, "cls_token")
        with torch.no_grad():
            d = self._forward(torch.zeros(1, *cfg["input_size"], device=self.device,
                                          dtype=torch.float16 if self.half else torch.float32))
        self.dim = int(d.shape[1])
        print(f"[models] {name}: {model_name}, D={self.dim}, device={self.device}, "
              f"input={cfg['input_size'][1:]}")

    def transform(self, img: Image.Image):
        return self._tf(img.convert("RGB"))

    def _forward(self, x):
        import torch
        if self.is_vit and self.pool == "cls+mean":
            tokens = self.model.forward_features(x)          # [B, 1+P, C]
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            cls = tokens[:, 0]
            patches = tokens[:, n_prefix:].mean(dim=1)
            return torch.cat([cls, patches], dim=1)
        return self.model(x)

    def embed_tensors(self, batch) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = batch.to(self.device)
            if self.half:
                x = x.half()
            out = self._forward(x).float().cpu().numpy()
        return l2n(out)

    def embed(self, images):
        import torch
        return self.embed_tensors(torch.stack([self.transform(i) for i in images]))


_REGISTRY = {
    "dinov2_s": dict(model_name="vit_small_patch14_dinov2.lvd142m", img_size=224),
    "dinov2_b": dict(model_name="vit_base_patch14_dinov2.lvd142m", img_size=224),
    "dinov2_l": dict(model_name="vit_large_patch14_dinov2.lvd142m", img_size=224),
    "clip_b":   dict(model_name="vit_base_patch16_clip_224.openai", pool="cls"),
    "resnet50": dict(model_name="resnet50.a1_in1k", pool="avg"),
}


class FinetunedBackbone(Backbone):
    """Дообученная модель из vreid.train (best.pt): backbone → cls+mean → BNNeck."""

    def __init__(self, path: str, device: str | None = None, half: bool = True):
        import torch
        import torchvision.transforms as T
        from .train import ReIDModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = ReIDModel.load(path, self.device)
        self.m.eval()
        self.half = half and str(self.device).startswith("cuda")
        self.name = "ft:" + Path(path).parent.name
        self.dim = self.m.dim
        self.size = self.m.img_size
        self._tf = T.Compose([T.Resize((self.size, self.size), interpolation=T.InterpolationMode.BICUBIC),
                              T.ToTensor(), T.Normalize(self.m.mean, self.m.std)])
        print(f"[models] {self.name}: {self.m.model_name}, D={self.dim}, input={self.size}, device={self.device}")
        # VREID_COMPILE=1 — прогнать backbone через torch.compile. Первый батч компилируется
        # десятки секунд, дальше forward обычно на 20-50% быстрее. На Windows Triton может быть
        # недоступен — тогда просто остаёмся на обычном режиме; в Linux-контейнере (а именно его
        # запускает жюри) работает. Ускорение прямо конвертируется в балл за пропускную способность.
        import os
        if os.environ.get("VREID_COMPILE", "") not in ("", "0"):
            try:
                self.m.backbone = torch.compile(self.m.backbone, mode="max-autotune-no-cudagraphs")
                print("[models] torch.compile включён (первый батч будет долгим)")
            except Exception as e:
                print(f"[models] torch.compile недоступен, работаю как обычно: {type(e).__name__}: {e}")

    def transform(self, img: Image.Image):
        return self._tf(img.convert("RGB"))

    def embed_tensors(self, batch) -> np.ndarray:
        import torch
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self.half):
            _, fb = self.m.features(batch.to(self.device))
        return l2n(fb.float().cpu().numpy())

    def embed(self, images):
        import torch
        return self.embed_tensors(torch.stack([self.transform(i) for i in images]))


def get_backbone(name: str, device: str | None = None) -> Backbone:
    if name == "dummy":
        return DummyBackbone()
    if name.startswith("ft:"):
        return FinetunedBackbone(name[3:], device=device)
    if name in _REGISTRY:
        return TimmBackbone(name=name, device=device, **_REGISTRY[name])
    # любое имя timm напрямую
    return TimmBackbone(model_name=name, name=name, device=device)
