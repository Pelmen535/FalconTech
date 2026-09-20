"""Привести логи прогонов к UTF-8.

PowerShell `Tee-Object` без `-Encoding` пишет UTF-16LE с BOM. Файл при этом корректен, но
любой читатель, ожидающий UTF-8 — git diff, просмотрщик на GitHub, grep, жюри — видит
мусор. Перевод в UTF-8 обратим и ничего не теряет: UTF-16 и UTF-8 кодируют один и тот же
текст.

Скрипт идемпотентен: файл, уже лежащий в UTF-8, не трогается. По умолчанию ничего не
меняет и только показывает, что собирается сделать.

    python scripts/normalize_logs.py            # показать
    python scripts/normalize_logs.py --apply    # переписать
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOMS = ((b"\xff\xfe\x00\x00", "utf-32-le"), (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"), (b"\xef\xbb\xbf", "utf-8-sig"))


def detect(raw: bytes) -> str | None:
    """Кодировка файла, если она не UTF-8. None — трогать не нужно."""
    for bom, name in BOMS:
        if raw.startswith(bom):
            return name
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        # Без BOM остаётся однобайтовая кодировка консоли Windows.
        return "cp1251"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=ROOT / "logs")
    parser.add_argument("--pattern", default="*.log")
    parser.add_argument("--apply", action="store_true", help="переписать файлы, а не только показать")
    args = parser.parse_args()

    changed = kept = failed = 0
    for path in sorted(args.path.rglob(args.pattern)):
        raw = path.read_bytes()
        encoding = detect(raw)
        if encoding is None:
            kept += 1
            continue
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            print(f"[logs] НЕ РАЗОБРАН {path.name}: не {encoding}")
            failed += 1
            continue
        # BOM после декодирования остаётся символом U+FEFF в начале строки — убираем.
        text = text.lstrip("\ufeff")
        changed += 1
        if args.apply:
            path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
        else:
            print(f"[logs] {path.name}: {encoding} → utf-8")
    verb = "переписано" if args.apply else "будет переписано"
    print(f"[logs] {verb} {changed}, уже в UTF-8 {kept}, не разобрано {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
