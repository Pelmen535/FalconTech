"""Скачивание весов модели из GitHub Release с проверкой контрольных сумм.

Зачем отдельный скрипт, а не Git LFS. Веса весят 743 МБ, а бесплатная квота GitHub на LFS —
1 ГБ хранилища и 1 ГБ трафика В МЕСЯЦ НА АККАУНТ. Одно клонирование съело бы её целиком, и
второй проверяющий получил бы вместо весов два текстовых указателя по 134 байта. Поэтому
веса лежат как файлы релиза: там лимит 2 ГБ на файл и трафик не ограничен.

Проверка обязательна и делается всегда. Совпадение SHA-256 означает, что скачано ровно то,
чем сняты все числа в отчётах: те же суммы записаны в reports/MANIFEST.json и печатаются
`python -m vreid.bench_full`, а сам рецепт (release/recipe.json) идёт в репозитории и в
проверке не нуждается.

    python scripts/fetch_weights.py                 # в release/ рядом с recipe.json
    python scripts/fetch_weights.py --out /tmp/rel  # в другое место
    python scripts/fetch_weights.py --check         # только проверить, что уже лежит

Зависимостей нет: только стандартная библиотека, чтобы скрипт работал до установки
requirements и внутри пустого окружения.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release" / "weights.json"


def load_manifest(path: Path = MANIFEST) -> tuple[str, str, dict]:
    """Какие файлы и с какими суммами ждать - записано рядом с recipe.json, а не в коде.

    Так смена релиза - это правка одного JSON, который коммитится вместе с рецептом, и
    рецепт с весами не могут разойтись: оба лежат в одном коммите. Имя файла → (SHA-256,
    размер, описание). Совпадение суммы означает, что скачано ровно то, чем сняты числа."""
    data = json.loads(path.read_text(encoding="utf-8"))
    files = {name: (spec["sha256"], int(spec["size"]), spec.get("what", ""))
             for name, spec in data["files"].items()}
    return data["tag"], f"https://github.com/{data['repo']}/releases/download", files


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str, size: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, "файла нет"
    actual_size = path.stat().st_size
    if actual_size != size:
        return False, f"размер {actual_size} вместо {size}"
    actual = sha256(path)
    if actual != expected:
        return False, f"sha256 {actual[:16]}… вместо {expected[:16]}…"
    return True, "совпал"


def download(url: str, target: Path, size: int) -> None:
    """Скачивание во временный файл рядом с целью: прерванная загрузка не оставит
    после себя файл, который следующий запуск примет за готовый."""
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if size:
                print(f"\r    {done / 2**20:7.1f} / {size / 2**20:.1f} МиБ "
                      f"({100 * done / size:5.1f}%)", end="", flush=True)
    print()
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "release",
                        help="куда класть веса (по умолчанию release/ рядом с recipe.json)")
    parser.add_argument("--tag", default=None, help="тег релиза (по умолчанию из release/weights.json)")
    parser.add_argument("--check", action="store_true",
                        help="ничего не скачивать, только проверить уже лежащее")
    parser.add_argument("--force", action="store_true",
                        help="перекачать, даже если файл на месте и сумма совпадает")
    args = parser.parse_args()

    tag, release_url, weights = load_manifest()
    tag = args.tag or tag
    bad = 0
    for name, (expected, size, what) in weights.items():
        target = args.out / name
        ok, reason = verify(target, expected, size)
        if ok and not args.force:
            print(f"[weights] {name}: на месте, sha256 {reason} — {what}")
            continue
        if args.check:
            print(f"[weights] {name}: ПРОБЛЕМА — {reason}")
            bad += 1
            continue
        url = f"{release_url}/{tag}/{name}"
        print(f"[weights] {name}: {reason}, качаю {size / 2**20:.0f} МиБ из {url}")
        try:
            download(url, target, size)
        except urllib.error.HTTPError as error:
            print(f"[weights] не скачалось ({error.code} {error.reason}). Проверь, что релиз "
                  f"{tag} опубликован: {release_url.rsplit('/', 1)[0]}/releases")
            bad += 1
            continue
        except urllib.error.URLError as error:
            print(f"[weights] сеть недоступна: {error.reason}")
            bad += 1
            continue
        ok, reason = verify(target, expected, size)
        # Скачалось не то — это не «почти получилось», а непригодный файл: числа в отчётах
        # сняты с конкретных весов, и подменённые к ним отношения не имеют.
        print(f"[weights] {name}: sha256 {reason}" if ok
              else f"[weights] {name}: ПОСЛЕ ЗАГРУЗКИ НЕ СОВПАЛО — {reason}")
        bad += 0 if ok else 1

    if bad:
        print(f"[weights] не в порядке файлов: {bad}")
        return 1
    print(f"[weights] всё на месте в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
