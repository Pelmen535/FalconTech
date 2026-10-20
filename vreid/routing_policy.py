"""Supplement legacy per-image routing using an immutable fixed-gallery profile.

The numeric gate sees only source-projected embeddings. Labels, keys and query
batch statistics are deliberately absent from this interface.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class GalleryProfile:
    mean_support: float
    use_supplement: bool


class GalleryRoutingPolicy:
    def __init__(self, gate, gallery_threshold, query_threshold):
        self.mean = np.array(gate['mean'], dtype=np.float64, copy=True)
        self.std = np.array(gate['std'], dtype=np.float64, copy=True)
        self.coef = np.array(gate['coef'], dtype=np.float64, copy=True)
        self.intercept = float(gate['intercept'])
        self.gallery_threshold = float(gallery_threshold)
        self.query_threshold = float(query_threshold)
        if self.mean.ndim != 1 or not (self.mean.shape == self.std.shape == self.coef.shape):
            raise ValueError('Invalid routing gate dimensions')
        if not all(np.isfinite(x).all() for x in [self.mean, self.std, self.coef]) or not np.isfinite(self.intercept) or np.any(self.std <= 0):
            raise ValueError('Invalid routing gate coefficients')
        if not (0 <= self.gallery_threshold <= 1 and 0 <= self.query_threshold <= 1):
            raise ValueError('Invalid routing thresholds')
        for value in [self.mean, self.std, self.coef]:
            value.setflags(write=False)

    @classmethod
    def load(cls, config_path):
        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        path = config_path.parent / config['gate_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != config['gate_SHA256']:
            raise ValueError('Routing gate checksum mismatch')
        with np.load(path, allow_pickle=False) as gate:
            return cls(gate, config['gallery_threshold'], config['query_threshold'])

    def support(self, source_embeddings):
        x = np.asarray(source_embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Expected finite source-projected embedding matrix')
        # Row reduction avoids GEMM batch-size dependent summation in the gate.
        logit = np.sum(((x - self.mean) / self.std) * self.coef, axis=1) + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700)))

    def gallery_profile(self, source_embeddings):
        scores = self.support(source_embeddings)
        if len(scores) == 0:
            raise ValueError('Empty gallery')
        # Sorted reduction makes the fixed profile invariant to gallery order.
        mean = float(np.mean(np.sort(scores)))
        return GalleryProfile(mean, mean >= self.gallery_threshold)

    def query_mask(self, source_embeddings, legacy_probability, profile):
        legacy = np.asarray(legacy_probability, dtype=np.float64)
        support = self.support(source_embeddings)
        if legacy.shape != support.shape or not np.isfinite(legacy).all():
            raise ValueError('Invalid legacy routing probabilities')
        return (legacy >= .95) | (profile.use_supplement & (support >= self.query_threshold))

    def gallery_mask(self, source_embeddings, legacy_probability, profile):
        legacy = np.asarray(legacy_probability, dtype=np.float64)
        if legacy.shape != (len(source_embeddings),) or not np.isfinite(legacy).all():
            raise ValueError('Invalid legacy gallery routing probabilities')
        return (legacy >= .95) | profile.use_supplement
