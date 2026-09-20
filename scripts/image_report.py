# -*- coding: utf-8 -*-
"""Что реально стоит внутри собранного образа. Запускать ВНУТРИ контейнера:

    docker run --rm --entrypoint python vreid-release /app/scripts/image_report.py

Нужен потому, что pip дважды отчитался об установке typing-extensions 4.12.1, скачав 4.16.0.
Верить логу сборки нельзя — надо спросить у самого образа. Заодно это и есть артефакт
для ответа 39: точные версии того, что поедет на стенд.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

print("[image] python", sys.version.split()[0])
for name in ("torch", "torchvision", "timm", "numpy", "PIL", "typing_extensions",
             "huggingface_hub", "safetensors", "yaml", "tqdm"):
    try:
        m = __import__(name)
        print(f"[image] {name:<18} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"[image] {name:<18} НЕ ИМПОРТИРУЕТСЯ: {type(e).__name__}: {e}")

lock = Path("/app/installed-packages.txt")
print(f"\n[image] pip freeze --all ({'есть' if lock.exists() else 'НЕТ ФАЙЛА'}):")
if lock.exists():
    print(lock.read_text(encoding="utf-8").strip())

try:
    sys.path.insert(0, "/app")
    from vreid.artifacts import weight_inventory
    inv = weight_inventory("/app/release")
    print(f"\n[image] веса: {json.dumps(inv, ensure_ascii=False)}")
except Exception as e:
    print(f"\n[image] инвентарь весов не собрался: {type(e).__name__}: {e}")
