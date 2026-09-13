from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .data import Record, image_to_array, load_vehicle_image


class Embedder(Protocol):
    def embed(self, records: Sequence[Record]) -> np.ndarray: ...


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


class ColorGridEmbedder:
    """Dependency-light smoke-test baseline, not a competitive Re-ID model."""

    def __init__(self, bins: int = 12, grid: tuple[int, int] = (2, 2)) -> None:
        self.bins = bins
        self.grid = grid

    def _one(self, image: Image.Image) -> np.ndarray:
        array = image_to_array(image.resize((128, 128)))
        features: list[np.ndarray] = []
        rows, cols = self.grid
        for row in range(rows):
            for col in range(cols):
                cell = array[
                    row * 128 // rows : (row + 1) * 128 // rows,
                    col * 128 // cols : (col + 1) * 128 // cols,
                ]
                for channel in range(3):
                    histogram, _ = np.histogram(cell[..., channel], bins=self.bins, range=(0.0, 1.0), density=False)
                    histogram = histogram.astype(np.float32)
                    histogram /= max(float(histogram.sum()), 1.0)
                    features.append(histogram)
        return np.concatenate(features)

    def embed(self, records: Sequence[Record]) -> np.ndarray:
        return l2_normalize(np.stack([self._one(load_vehicle_image(record)) for record in records]))


class OnnxEmbedder:
    def __init__(self, model_path: str | Path, spec_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('ONNX backend requires: pip install -e ".[onnx]"') from exc
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        self.size = tuple(spec.get("size", [224, 224]))
        self.mean = np.asarray(spec.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(spec.get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 3, 1, 1)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _prepare(self, record: Record) -> np.ndarray:
        image = load_vehicle_image(record).resize(self.size, Image.Resampling.BILINEAR)
        array = image_to_array(image).transpose(2, 0, 1)[None]
        return (array - self.mean) / self.std

    def embed(self, records: Sequence[Record]) -> np.ndarray:
        batch = np.concatenate([self._prepare(record) for record in records], axis=0).astype(np.float32)
        output = self.session.run(None, {self.input_name: batch})[0]
        return l2_normalize(np.asarray(output).reshape(len(records), -1))
