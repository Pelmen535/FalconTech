"""Сплиты re-ID датасетов.

Один сплит = список записей (путь к кропу, id машины, id камеры).
Формат данных абстрагирован: сегодня CARLA, завтра VeRi-776 или кадры организаторов —
меняется только конфиг (папки и регулярное выражение для имени файла).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Record:
    path: str
    vid: int   # id транспортного средства; -1 = неизвестен (тест / дистрактор без пары)
    cam: int   # id камеры / точки съёмки (в данных организаторов — псевдокамера)
    bbox: tuple | None = None   # (x, y, w, h) в пикселях полного кадра; None = файл уже кроп
    key: str = ""               # идентификатор строки (image_id + номер bbox) для файлов сдачи


@dataclass
class Split:
    name: str
    records: list[Record]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def vids(self):
        import numpy as np
        return np.array([r.vid for r in self.records], dtype=np.int64)

    @property
    def cams(self):
        import numpy as np
        return np.array([r.cam for r in self.records], dtype=np.int64)

    @property
    def paths(self) -> list[str]:
        return [r.path for r in self.records]

    def num_ids(self) -> int:
        return len({r.vid for r in self.records})

    @property
    def keys(self) -> list[str]:
        return [r.key or r.path for r in self.records]

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "vid", "cam", "x", "y", "w", "h", "key"])
            for r in self.records:
                b = r.bbox or ("", "", "", "")
                w.writerow([r.path, r.vid, r.cam, *b, r.key])

    @staticmethod
    def from_csv(path: str | Path, name: str | None = None) -> "Split":
        recs = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bbox = None
                if row.get("x", "") != "":
                    bbox = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
                recs.append(Record(row["path"], int(row["vid"]), int(row["cam"]), bbox, row.get("key", "")))
        return Split(name or Path(path).stem, recs)


def _iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            yield p


def split_from_folder(folder: str | Path, pattern: str, name: str,
                      vid_group: str = "vid", cam_group: str = "cam",
                      strict: bool = False) -> Split:
    """Собирает сплит из папки, вытаскивая vid и cam из имени файла регуляркой.

    pattern — регулярное выражение с именованными группами (?P<vid>...) и (?P<cam>...).
    Нечисловые id (например "c001") переводятся в int по цифрам внутри.
    Файлы, не подошедшие под pattern, пропускаются (strict=True — ошибка).
    """
    folder = Path(folder)
    rx = re.compile(pattern)
    recs, skipped = [], 0
    for p in _iter_images(folder):
        m = rx.search(p.name)
        if not m:
            skipped += 1
            if strict:
                raise ValueError(f"{p.name} не подходит под {pattern}")
            continue
        recs.append(Record(str(p), _to_int(m.group(vid_group)), _to_int(m.group(cam_group))))
    if skipped:
        print(f"[datasets] {name}: пропущено {skipped} файлов, не подошедших под шаблон")
    if not recs:
        raise RuntimeError(f"[datasets] {name}: в {folder} не найдено ни одного файла под {pattern}")
    return Split(name, recs)


def _to_int(s: str) -> int:
    digits = re.sub(r"\D", "", s)
    if digits:
        return int(digits)
    # нечисловой id — стабильный хэш
    return abs(hash(s)) % (10 ** 9)


# ---- готовые форматы -------------------------------------------------------

# VeRi-776: 0002_c002_00030600_0.jpg  → vid=2, cam=2
VERI_PATTERN = r"^(?P<vid>\d{4})_c(?P<cam>\d{3})_"

# VeRi-CARLA (sekilab): "<vehicleID>_<cameraID>..." — точный формат уточняется
# скриптом scripts/prepare_carla.py, который печатает примеры имён.
CARLA_PATTERN = r"^(?P<vid>\d+)_(?P<cam>\d+)"


def load_dataset(cfg: dict) -> dict[str, Split]:
    """cfg (из YAML):
        root: путь к датасету
        pattern: регулярка для имён файлов
        splits: {query: "image_query", gallery: "image_test", train: "image_train"}
    Возвращает словарь имя_сплита → Split.
    """
    root = Path(cfg["root"])
    pattern = cfg.get("pattern", VERI_PATTERN)
    out = {}
    for split_name, sub in cfg["splits"].items():
        folder = root / sub
        if not folder.exists():
            print(f"[datasets] сплит {split_name}: папка {folder} не найдена, пропускаю")
            continue
        out[split_name] = split_from_folder(folder, pattern, split_name)
        s = out[split_name]
        print(f"[datasets] {split_name}: {len(s)} изображений, {s.num_ids()} id, "
              f"{len(set(s.cams.tolist()))} камер")
    if "query" not in out or "gallery" not in out:
        raise RuntimeError("нужны сплиты query и gallery")
    return out
