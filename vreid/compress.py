"""Сжатие слепка: кривая «байты → качество».

Варианты (имя → байт на слепок при D_pca = d):
  f32           сырой эмбеддинг                          4·D
  f16           сырой в half                             2·D
  pca{d}        PCA до d, float16                        2·d
  pcaw{d}       PCA + whitening до d, float16            2·d
  pca{d}_i8     PCA до d, int8 с общей шкалой            d
  pca{d}_bin    PCA до d, знак → бинарный код            d/8   (сравнение по Хэммингу)
  pq{m}         product quantization, m байт            m     (faiss, 256 центроидов на подвектор)

Все варианты «обучаются» на галерее/трейне (fit) и применяются к запросам (encode/decode).
Для оценки достаточно decode → косинус, кроме bin (Хэмминг) — так качество слегка
пессимистично для PQ (симметричное расстояние), зато честно и просто.
"""
from __future__ import annotations

import re

import numpy as np


def _l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class PCA:
    def __init__(self, d: int, whiten: bool = False, eps: float = 1e-6):
        self.d, self.whiten, self.eps = d, whiten, eps

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        xc = x - self.mean
        # SVD по ковариации — устойчиво и быстро при D ≤ 2048
        cov = xc.T @ xc / max(1, len(xc) - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1][: self.d]
        self.evals = np.maximum(evals[order], 0)
        self.W = evecs[:, order]
        if self.whiten:
            self.W = self.W / np.sqrt(self.evals + self.eps)
        self.explained = float(self.evals.sum() / max(evals.sum(), 1e-12))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.W


class Codec:
    """name → (bytes_per_vector, fit, encode→decoded float / binary, similarity)."""

    def __init__(self, name: str):
        self.name = name
        self.kind, self.d, self.m, self.whiten = self._parse(name)
        self.pca = None
        self.pq = None
        self.scale = None

    @staticmethod
    def _parse(name):
        if name in ("f32", "f16"):
            return name, None, None, False
        m = re.fullmatch(r"(pcaw?)(\d+)(?:_(i8|bin))?", name)
        if m:
            kind = m.group(3) or "f16"
            return kind, int(m.group(2)), None, m.group(1) == "pcaw"
        m = re.fullmatch(r"pq(\d+)", name)
        if m:
            return "pq", None, int(m.group(1)), False
        raise ValueError(f"неизвестный кодек {name}")

    # ---- размер ----
    def bytes_per_vector(self, D: int) -> int:
        if self.kind == "f32":
            return 4 * D
        if self.kind == "f16":
            return 2 * (self.d or D)
        if self.kind == "i8":
            return self.d
        if self.kind == "bin":
            return int(np.ceil(self.d / 8))
        if self.kind == "pq":
            return self.m
        raise AssertionError

    # ---- обучение ----
    def fit(self, x: np.ndarray):
        x = x.astype(np.float32)
        if self.d is not None:
            self.pca = PCA(self.d, whiten=self.whiten).fit(x)
            z = self.pca.transform(x)
            if self.kind == "i8":
                self.scale = float(np.quantile(np.abs(z), 0.999)) / 127.0 + 1e-12
        if self.kind == "pq":
            import faiss
            D = x.shape[1]
            assert D % self.m == 0, f"D={D} не делится на m={self.m}"
            self.pq = faiss.ProductQuantizer(D, self.m, 8)
            self.pq.train(np.ascontiguousarray(x))
        return self

    # ---- кодирование ----
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Возвращает то, что хранится в базе: float16 / int8 / packed bits / uint8-коды."""
        x = x.astype(np.float32)
        if self.kind == "f32":
            return x
        if self.kind == "f16":
            z = self.pca.transform(x) if self.pca else x
            return z.astype(np.float16)
        if self.kind == "i8":
            z = self.pca.transform(x)
            return np.clip(np.round(z / self.scale), -127, 127).astype(np.int8)
        if self.kind == "bin":
            z = self.pca.transform(x)
            return np.packbits(z > 0, axis=1)
        if self.kind == "pq":
            return self.pq.compute_codes(np.ascontiguousarray(x))
        raise AssertionError

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if self.kind in ("f32", "f16"):
            return codes.astype(np.float32)
        if self.kind == "i8":
            return codes.astype(np.float32) * self.scale
        if self.kind == "bin":
            return np.unpackbits(codes, axis=1)[:, : self.d].astype(np.float32)
        if self.kind == "pq":
            return self.pq.decode(codes)
        raise AssertionError

    # ---- сходство ----
    def similarity(self, q_codes: np.ndarray, g_codes: np.ndarray) -> np.ndarray:
        """[Q, G], больше = ближе. Для bin — 1 - hamming/d, иначе косинус декодированных."""
        if self.kind == "bin":
            q = np.unpackbits(q_codes, axis=1)[:, : self.d].astype(np.int32)
            g = np.unpackbits(g_codes, axis=1)[:, : self.d].astype(np.int32)
            ham = (q[:, None, :] != g[None, :, :]).sum(axis=2) if q.shape[0] * g.shape[0] < 5e7 else _ham_blocked(q, g)
            return 1.0 - ham / float(self.d)
        q, g = _l2n(self.decode(q_codes)), _l2n(self.decode(g_codes))
        return q @ g.T


def _ham_blocked(q, g, block=256):
    out = np.empty((q.shape[0], g.shape[0]), dtype=np.int32)
    for i in range(0, q.shape[0], block):
        out[i:i + block] = (q[i:i + block, None, :] != g[None, :, :]).sum(axis=2)
    return out


DEFAULT_VARIANTS = ["f32", "f16", "pca256", "pcaw256", "pca128_i8", "pca64_i8",
                    "pca256_bin", "pca128_bin", "pq32", "pq16"]


def evaluate_variants(q_emb, g_emb, q_vids, q_cams, g_vids, g_cams,
                      fit_emb=None, variants=None) -> list[dict]:
    """Таблица: вариант → байт/слепок, rank-1, mAP (оба протокола)."""
    from .metrics import evaluate
    fit_emb = g_emb if fit_emb is None else fit_emb
    D = q_emb.shape[1]
    rows = []
    for name in variants or DEFAULT_VARIANTS:
        codec = Codec(name)
        if codec.d is not None and codec.d > min(D, len(fit_emb) - 1):
            continue
        if codec.kind == "pq" and D % codec.m != 0:
            continue
        try:
            codec.fit(fit_emb)
        except Exception as e:  # faiss может отказать на слишком маленькой выборке
            print(f"[compress] {name}: пропущен ({e})")
            continue
        sims = codec.similarity(codec.encode(q_emb), codec.encode(g_emb))
        std = evaluate(sims, q_vids, q_cams, g_vids, g_cams)
        cc = evaluate(sims, q_vids, q_cams, g_vids, g_cams, cross_camera_only=True)
        rows.append({"variant": name, "bytes": codec.bytes_per_vector(D),
                     "rank1": std["rank1"], "mAP": std["mAP"],
                     "rank1_cross": cc["rank1"], "mAP_cross": cc["mAP"],
                     "pca_explained": getattr(codec.pca, "explained", None)})
    return rows


def format_table(rows: list[dict]) -> str:
    lines = [f"{'вариант':<12}{'байт':>6}  {'rank-1':>7} {'mAP':>7}   {'rank-1 cc':>9} {'mAP cc':>7}"]
    for r in rows:
        lines.append(f"{r['variant']:<12}{r['bytes']:>6}  {r['rank1'] * 100:6.1f}% {r['mAP'] * 100:6.1f}%   "
                     f"{r['rank1_cross'] * 100:8.1f}% {r['mAP_cross'] * 100:6.1f}%")
    return "\n".join(lines)
