"""Validate HTTP uploads; archive members are read directly, never extracted."""
import csv
from io import BytesIO, StringIO
import json
import math
from pathlib import PurePosixPath
import warnings
import zipfile
from PIL import Image, UnidentifiedImageError


def image_box(content, value, settings):
    if not content or len(content) > settings.max_image_bytes:
        raise ValueError('Пустое изображение или превышен лимит 20 МиБ.')
    try:
        box = json.loads(value) if isinstance(value, str) else value
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(isinstance(v, bool) for v in box):
            raise ValueError('bbox должен содержать четыре числа x,y,w,h.')
        box = tuple(float(v) for v in box)
        if not all(math.isfinite(v) for v in box) or min(box[:2]) < 0 or min(box[2:]) <= 0:
            raise ValueError('bbox: координаты неотрицательные, ширина и высота положительные.')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as im:
                w, h = im.size
                fmt = im.format
                if fmt not in ('JPEG', 'PNG', 'WEBP') or w * h > settings.max_image_pixels:
                    raise ValueError('Допустимы JPEG/PNG/WebP до 40 миллионов пикселей.')
                if box[0] + box[2] > w + 1e-5 or box[1] + box[3] > h + 1e-5:
                    raise ValueError('bbox выходит за пределы изображения.')
                im.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Не удалось прочитать изображение.') from exc
    except (TypeError, OverflowError) as exc:
        raise ValueError('Некорректный bbox.') from exc
    return box


def gallery_archive(content, settings):
    if not content or len(content) > settings.max_archive_bytes:
        raise ValueError('Пустой или слишком большой ZIP.')
    try:
        with zipfile.ZipFile(BytesIO(content)) as z:
            entries = z.infolist()
            if len(entries) > 2 * settings.max_gallery_items + 30:
                raise ValueError('Слишком много файлов в ZIP.')
            names = {}
            for info in entries:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in info.filename or ':' in info.filename:
                    raise ValueError('Недопустимый путь в ZIP.')
                if info.filename in names:
                    raise ValueError('Повторяющееся имя файла в ZIP.')
                names[info.filename] = info
            if sum(i.file_size for i in entries) > settings.max_expanded_bytes:
                raise ValueError('ZIP слишком велик после распаковки.')
            manifests = [n for n in names if PurePosixPath(n).name == 'test_gallery.csv']
            if len(manifests) != 1:
                raise ValueError('В ZIP нужен один test_gallery.csv и каталог images/.')
            manifest = manifests[0]
            if names[manifest].file_size > 2 * 1024 * 1024:
                raise ValueError('Слишком большой CSV.')
            reader = csv.DictReader(StringIO(z.read(manifest).decode('utf-8-sig')))
            if not {'image_id', 'x', 'y', 'w', 'h'} <= set(reader.fieldnames or []):
                raise ValueError('CSV: обязательны image_id,x,y,w,h.')
            rows = list(reader)
            if not 1 <= len(rows) <= settings.max_gallery_items:
                raise ValueError('Недопустимое количество строк галереи.')
            base = PurePosixPath(manifest).parent / 'images'
            result = []; seen = set()
            for row in rows:
                key = row['image_id']
                if not key or key in seen or any(c in key for c in '/\\\\'):
                    raise ValueError('Пустой, повторяющийся или некорректный image_id.')
                seen.add(key)
                matches = [n for n, info in names.items() if not info.is_dir() and PurePosixPath(n).parent == base
                           and (PurePosixPath(n).name == key or PurePosixPath(n).stem == key)]
                if len(matches) != 1:
                    raise ValueError('Не найдено однозначное изображение для ' + key)
                info = names[matches[0]]
                if info.file_size > settings.max_image_bytes:
                    raise ValueError('Превышен размер изображения.')
                raw = z.read(info)
                box = image_box(raw, [row[k] for k in ('x','y','w','h')], settings)
                result.append(dict(image_id=key, image_bytes=raw, bbox_xywh=box))
            return result
    except (zipfile.BadZipFile, UnicodeError, RuntimeError) as exc:
        raise ValueError('Не удалось прочитать ZIP/CSV.') from exc
