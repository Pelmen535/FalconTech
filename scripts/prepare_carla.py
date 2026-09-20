"""Подготовка VeRi-CARLA.

1. Скачай архив (Dropbox, ~ несколько ГБ):
   https://www.dropbox.com/s/cg1etrs22y2xb62/VeRi_CARLA_dataset.zip?dl=1
   и положи его в data/ (имя VeRi_CARLA_dataset.zip).
2. python scripts/prepare_carla.py
   — распакует в data/VeRi_CARLA_dataset, покажет структуру папок и примеры имён файлов,
   и проверит, что регулярка из configs/carla.yaml разбирает имена.
Если папки называются иначе, чем query/gallery/train — скрипт подскажет, что поправить в конфиге.
"""
from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "VeRi_CARLA_dataset.zip"
OUT = ROOT / "data" / "VeRi_CARLA_dataset"
CFG = ROOT / "configs" / "carla.yaml"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    if not OUT.exists():
        if not ZIP.exists():
            print(f"Нет {ZIP}. Скачай архив по ссылке в шапке скрипта и положи в data/.")
            sys.exit(1)
        print(f"Распаковываю {ZIP} → {OUT} ...")
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(OUT)
    # если внутри архива один корневой каталог — поднимаем его содержимое
    kids = [p for p in OUT.iterdir() if not p.name.startswith("__MACOSX")]
    if len(kids) == 1 and kids[0].is_dir():
        inner = kids[0]
        print(f"В архиве один каталог {inner.name}; использую его как корень")
        root = inner
    else:
        root = OUT

    print("\nСтруктура (папка: число изображений):")
    folders = {}
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            n = sum(1 for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
            if n:
                folders[p] = n
                print(f"  {p.relative_to(root)}: {n}")
    if not folders:
        print("Изображений не найдено — посмотри содержимое data/VeRi_CARLA_dataset вручную")
        sys.exit(1)

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    pattern = cfg["dataset"]["pattern"]
    rx = re.compile(pattern)
    print(f"\nПроверка регулярки {pattern!r}:")
    for folder, n in folders.items():
        files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT][:5]
        ok = sum(1 for f in files if rx.search(f.name))
        print(f"  {folder.relative_to(root)}: примеры {[f.name for f in files[:3]]} — подошло {ok}/{len(files)}")
        if ok:
            vids = Counter(rx.search(f.name).group("vid") for f in folder.iterdir()
                           if f.suffix.lower() in IMG_EXT and rx.search(f.name))
            print(f"      уникальных vid: {len(vids)}")

    want = cfg["dataset"]["splits"]
    print("\nОжидаемые сплиты из configs/carla.yaml:")
    for k, v in want.items():
        exists = (root / v).exists()
        print(f"  {k}: {v} — {'OK' if exists else 'НЕТ ТАКОЙ ПАПКИ — поправь splits в конфиге'}")
    if root != OUT:
        print(f"\nВ configs/carla.yaml поставь root: {root.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
