# -*- coding: utf-8 -*-
"""SHA-256 предобученных весов DINOv2 из локального кеша Hugging Face.

    python scripts/hash_pretrained.py

Ответ 46 требует однозначного источника внешних весов. Имя модели и URL записаны в
handoff/SOURCES.md; этот скрипт добавляет к ним хэш конкретного файла, который реально
скачался и от которого шло дообучение, — чтобы проверяющий мог сверить байты, а не поверить
названию. Кеш в поставку не входит: релизный чекпойнт самодостаточен.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

NAMES = ["vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m"]


def roots():
    for env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        v = os.environ.get(env)
        if v:
            yield Path(v)
    yield Path.home() / ".cache" / "huggingface"
    yield Path.home() / ".cache" / "torch"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    out = {}
    seen = set()
    for root in roots():
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        for name in NAMES:
            for p in root.rglob("*"):
                if not p.is_file() or p.suffix not in (".safetensors", ".bin", ".pth"):
                    continue
                if name.split(".")[0] not in str(p).replace("_", "_"):
                    continue
                if name in out:
                    continue
                if name.replace(".", "--") in str(p) or name in str(p):
                    out[name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
    if not out:
        print("[hash] кеш DINOv2 не найден. Это не ошибка сдачи: релизный чекпойнт "
              "самодостаточен (pretrained=False). Но для SOURCES.md хэш стоит получить — "
              "запусти на машине, где шло обучение.")
        return
    for k, v in out.items():
        print(f"{k}\n  файл   {v['path']}\n  размер {v['bytes'] / 1024**2:.1f} МиБ\n  sha256 {v['sha256']}")
    Path("results").mkdir(exist_ok=True)
    Path("results/pretrained_hashes.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[hash] → results/pretrained_hashes.json")


if __name__ == "__main__":
    main()
