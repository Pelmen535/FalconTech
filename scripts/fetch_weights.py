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
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_URL = "https://github.com/Pelmen535/FalconTech/releases/download"
TAG = "v1.0"

# Имя файла → (SHA-256, размер в байтах, что это такое).
# Суммы не «для галочки»: model.pt — та модель, чей путь меряет жюри, reranker.pt — вторая
# ступень каскада. Расхождение суммы означает, что числа в отчётах к этим файлам не относятся.
WEIGHTS = {
    "model.pt": (
        "608042f64c259b97590e57b81f6c03e7143fba43d9a9c320b4a2a346e9f5fd1c",
        171995879,
        "DINOv2 ViT-B/14 336, основная модель: её вектор идёт в embeddings.npy",
    ),
    "reranker.pt": (
        "6ac3f746aebb20c04526c81708224ebc90b715015d9e99e2e94af74580b66299",
        607231914,
        "DINOv2 ViT-L/14 336, ре-ранкер второй ступени: уточняет порядок внутри топ-30",
    ),
}


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
    parser.add_argument("--tag", default=TAG, help="тег релиза на GitHub")
    parser.add_argument("--check", action="store_true",
                        help="ничего не скачивать, только проверить уже лежащее")
    parser.add_argument("--force", action="store_true",
                        help="перекачать, даже если файл на месте и сумма совпадает")
    args = parser.parse_args()

    bad = 0
    for name, (expected, size, what) in WEIGHTS.items():
        target = args.out / name
        ok, reason = verify(target, expected, size)
        if ok and not args.force:
            print(f"[weights] {name}: на месте, sha256 {reason} — {what}")
            continue
        if args.check:
            print(f"[weights] {name}: ПРОБЛЕМА — {reason}")
            bad += 1
            continue
        url = f"{RELEASE_URL}/{args.tag}/{name}"
        print(f"[weights] {name}: {reason}, качаю {size / 2**20:.0f} МиБ из {url}")
        try:
            download(url, target, size)
        except urllib.error.HTTPError as error:
            print(f"[weights] не скачалось ({error.code} {error.reason}). Проверь, что релиз "
                  f"{args.tag} опубликован: {RELEASE_URL.rsplit('/', 1)[0]}/releases")
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
