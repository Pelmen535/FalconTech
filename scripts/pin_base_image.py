# -*- coding: utf-8 -*-
"""Зафиксировать базовый образ Dockerfile по SHA-256 дайджесту вместо тега.

    python scripts/pin_base_image.py --dockerfile Dockerfile

Организаторы требуют точных версий и запрещают ссылаться на подвижные теги (ответ 39).
Тег вида 2.3.1-cuda12.1-cudnn8-runtime формально версионный, но владелец репозитория
вправе перезалить его: тогда сборка у жюри соберётся из ДРУГИХ байтов, чем у нас, и
«воспроизводимо» перестанет быть правдой. Дайджест этого не позволяет.

Дайджест берётся у локального демона docker (образ уже скачан сборкой), поэтому это
ровно тот образ, на котором мы мерили, а не «какой сейчас лежит в реестре».
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TAG_RE = re.compile(r"^ARG BASE_IMAGE=(?P<ref>\S+)\s*$", re.M)


def digest_of(ref: str) -> str:
    try:
        out = subprocess.run(["docker", "inspect", "--format", "{{json .RepoDigests}}", ref],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise SystemExit(f"не смог спросить docker про {ref}: {e}\n"
                         f"сначала docker pull {ref}")
    import json
    digs = json.loads(out)
    if not digs:
        raise SystemExit(f"у {ref} нет RepoDigests — образ собран локально, а не скачан. "
                         f"Сделай docker pull {ref} и повтори.")
    return digs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dockerfile", default="Dockerfile")
    ap.add_argument("--check", action="store_true", help="только проверить, ничего не писать")
    a = ap.parse_args()

    p = Path(a.dockerfile)
    s = p.read_text(encoding="utf-8")
    m = TAG_RE.search(s)
    if not m:
        raise SystemExit(f"в {p} нет строки ARG BASE_IMAGE=")
    ref = m.group("ref")
    if "@sha256:" in ref:
        print(f"[pin] уже по дайджесту: {ref}")
        return 0
    pinned = digest_of(ref)
    print(f"[pin] {ref}\n  -> {pinned}")
    if a.check:
        print("[pin] --check: файл не менялся")
        return 1
    s = s.replace(f"ARG BASE_IMAGE={ref}",
                  f"# Тег, из которого получен дайджест ниже: {ref}\n"
                  f"# Дайджест зафиксирован scripts/pin_base_image.py — тег владелец репозитория\n"
                  f"# вправе перезалить, дайджест нет (ответ 39: точные версии, никаких подвижных ссылок).\n"
                  f"ARG BASE_IMAGE={pinned}")
    p.write_text(s, encoding="utf-8")
    print(f"[pin] {p} обновлён")
    return 0


if __name__ == "__main__":
    sys.exit(main())
