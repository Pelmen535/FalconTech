"""Дождаться, пока адрес начнёт отвечать 200. Нужен в очередях после `docker compose up`.

    python scripts/wait_http.py http://localhost:8080/health --timeout 300

Возвращает 0, когда ответ получен, и 1 по истечении времени. Никаких сторонних библиотек:
скрипт должен работать и внутри контейнера, где стоит только стандартная библиотека.
"""
from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=300.0, help="секунд всего")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                if response.status == 200:
                    waited = args.timeout - (deadline - time.monotonic())
                    print(f"[wait] {args.url} отвечает через {waited:.0f} с")
                    return 0
                last = f"HTTP {response.status}"
        except (urllib.error.URLError, OSError, TimeoutError) as reason:
            last = f"{type(reason).__name__}: {reason}"
        time.sleep(args.interval)
    print(f"[wait] {args.url} не ответил за {args.timeout:.0f} с; последняя причина — {last}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
