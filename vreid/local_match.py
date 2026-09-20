"""Локальное сопоставление патчей для ре-ранжирования топ-K и объяснимости.

Идея: глобальный вектор путает одинаковые модели одного цвета; различают их детали
(наклейка, диск, вмятина). Для пары кропов берём плотные патч-признаки ViT (DINOv2),
находим взаимные ближайшие патчи, оставляем только геометрически согласованные
(одинаковый сдвиг на сетке ± допуск) и считаем их число — это «сколько деталей совпало».

  LocalFeatures.extract(split) → tokens [N, T, d] float16 (после PCA), grid (h, w)
  match_pair(fa, fb, grid)     → {'n_mutual', 'n_inliers', 'pairs'}
  local_scores(...)            → [Q, K] число согласованных совпадений для топ-K кандидатов
  fuse(sims, order, local, beta) → новое ранжирование

Объяснимость: pairs — координаты совпавших патчей, их можно нарисовать на двух кропах.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .datasets import Split
from .extract import crop_bbox


class _CropDS:
    """Датасет кропов для DataLoader (класс верхнего уровня — иначе Windows/spawn не может его запаковать)."""

    def __init__(self, records, tf, pad, mask_frac=None):
        self.records, self.tf, self.pad, self.mask_frac = records, tf, pad, mask_frac

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r.path) as im:
            im = im.convert("RGB")
            if r.bbox is not None:
                im = crop_bbox(im, r.bbox, self.pad, self.mask_frac)
            return self.tf(im)


class LocalFeatures:
    def __init__(self, backbone, size: int = 336, pca_dim: int = 64, device: str | None = None):
        """backbone — TimmBackbone (ViT) или FinetunedBackbone; нужен доступ к forward_features."""
        import torch
        import torchvision.transforms as T
        self.bb = backbone
        self.model = getattr(backbone, "model", None) or backbone.m.backbone
        self.device = backbone.device
        self.size = size
        self.pca_dim = pca_dim
        mean = getattr(backbone, "_tf", None)
        try:  # достаём mean/std из уже собранного трансформа
            norm = [t for t in backbone._tf.transforms if isinstance(t, T.Normalize)][0]
            self.mean, self.std = norm.mean, norm.std
        except Exception:
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(self.mean, self.std)])
        patch = getattr(self.model.patch_embed, "patch_size", (14, 14))
        p = patch[0] if isinstance(patch, (tuple, list)) else patch
        self.grid = (size // p, size // p)
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        self.P = None  # PCA

    def _tokens(self, batch):
        import torch
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
            tok = self.model.forward_features(batch.to(self.device))
        return tok[:, self.n_prefix:].float().cpu().numpy()   # [B, T, C]

    def fit_pca(self, sample_tokens: np.ndarray):
        x = sample_tokens.reshape(-1, sample_tokens.shape[-1]).astype(np.float32)
        if len(x) > 50000:
            x = x[np.random.default_rng(0).choice(len(x), 50000, replace=False)]
        self.mu = x.mean(axis=0)
        u, s, vt = np.linalg.svd(x - self.mu, full_matrices=False)
        self.P = vt[: self.pca_dim].T.astype(np.float32)

    def _project(self, tok):
        z = (tok - self.mu) @ self.P
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def extract(self, split: Split, out: str | Path, pad: float = 0.08, mask_frac=None,
                batch_size: int = 32, workers: int = 4) -> dict:
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        dl = DataLoader(_CropDS(split.records, self.tf, pad, mask_frac), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
        raw = []
        for b in tqdm(dl, desc=f"local[{split.name}]", unit="batch"):
            raw.append(self._tokens(b))
            if self.P is None and sum(len(r) for r in raw) * raw[0].shape[1] >= 20000:
                self.fit_pca(np.concatenate(raw))
        raw = np.concatenate(raw)
        if self.P is None:
            self.fit_pca(raw)
        tokens = self._project(raw).astype(np.float16)
        data = {"tokens": tokens, "grid": np.array(self.grid), "keys": np.array(split.keys),
                "pca_P": self.P, "pca_mu": self.mu}
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **data)
        print(f"[local] {split.name}: tokens {tokens.shape} → {out}")
        return data


# --------------------------------------------------------------------------- сопоставление
def match_pair(fa: np.ndarray, fb: np.ndarray, grid, min_sim: float = 0.6, tol: float = 2.0) -> dict:
    """fa, fb: [T, d] нормированные. Взаимные ближайшие патчи с sim ≥ min_sim,
    затем согласованность сдвига: оставляем совпадения, чей сдвиг (dx, dy) на сетке
    лежит в пределах tol от доминирующего сдвига."""
    h, w = int(grid[0]), int(grid[1])
    s = fa.astype(np.float32) @ fb.astype(np.float32).T   # [T, T]
    ab = s.argmax(axis=1)
    ba = s.argmax(axis=0)
    ia = np.arange(len(fa))
    mutual = ba[ab] == ia
    ok = mutual & (s[ia, ab] >= min_sim)
    a_idx, b_idx = ia[ok], ab[ok]
    n_mutual = int(ok.sum())
    if n_mutual == 0:
        return {"n_mutual": 0, "n_inliers": 0, "pairs": np.zeros((0, 4), dtype=np.int64)}
    ay, ax = a_idx // w, a_idx % w
    by, bx = b_idx // w, b_idx % w
    d = np.stack([bx - ax, by - ay], axis=1).astype(np.float32)
    # доминирующий сдвиг: точка с максимальным числом соседей в радиусе tol
    dist = np.abs(d[:, None, :] - d[None, :, :]).max(axis=2)
    support = (dist <= tol).sum(axis=1)
    center = d[support.argmax()]
    inl = np.abs(d - center).max(axis=1) <= tol
    pairs = np.stack([ax[inl], ay[inl], bx[inl], by[inl]], axis=1)
    return {"n_mutual": n_mutual, "n_inliers": int(inl.sum()), "pairs": pairs,
            "shift": center.tolist()}


def local_scores(q_tokens: np.ndarray, g_tokens: np.ndarray, order: np.ndarray, grid,
                 topk: int = 30, min_sim: float = 0.6, tol: float = 2.0, verbose: bool = True) -> np.ndarray:
    """Для каждого запроса и его топ-K кандидатов (order[:, :topk]) — число согласованных совпадений."""
    Q = q_tokens.shape[0]
    out = np.zeros((Q, topk), dtype=np.float32)
    it = range(Q)
    if verbose:
        from tqdm import tqdm
        it = tqdm(it, desc="local-match", unit="q")
    for i in it:
        fa = q_tokens[i]
        for r in range(min(topk, order.shape[1])):
            j = order[i, r]
            if j < 0:
                continue
            out[i, r] = match_pair(fa, g_tokens[j], grid, min_sim, tol)["n_inliers"]
    return out


def fuse(sims: np.ndarray, order: np.ndarray, local: np.ndarray, beta: float, T: int) -> np.ndarray:
    """Новая матрица сходств: для топ-K — sim + beta·(inliers / T), остальным без изменений."""
    new = sims.copy()
    K = local.shape[1]
    rows = np.arange(sims.shape[0])[:, None]
    idx = order[:, :K]
    valid = idx >= 0
    new[rows.repeat(K, 1)[valid], idx[valid]] = sims[rows.repeat(K, 1)[valid], idx[valid]] + beta * local[valid] / float(T)
    return new


def tune_beta(sims, order, local, T, q_vids, q_cams, g_vids, g_cams, has_match,
              betas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0)) -> tuple[float, dict]:
    from .metrics import evaluate
    best = (0.0, None)
    for b in betas:
        s2 = fuse(sims, order, local, b, T)
        m = evaluate(s2[has_match], q_vids[has_match], q_cams[has_match], g_vids, g_cams, cross_camera_only=True, cutoff=10)
        if best[1] is None or m["mAP"] > best[1]["mAP"]:
            best = (b, m)
    return best
