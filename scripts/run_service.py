"""Запуск HTTP-сервиса поверх ядра из этого же репозитория.

    python scripts/run_service.py --device cuda            # локально, с GPU
    python scripts/run_service.py --device cpu             # без GPU
    docker compose up                                      # всё вместе, одной командой

По умолчанию слушает только 127.0.0.1. Привязка к другому адресу требует VREID_API_KEY —
иначе сервис с загруженной моделью оказался бы открыт всей сети без единой проверки.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from service.config import Settings

    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--core-dir', type=Path, default=defaults.core_dir,
                        help='папка, внутри которой лежит vreid/ (по умолчанию корень репозитория)')
    parser.add_argument('--release-dir', type=Path, default=None,
                        help='папка с model.pt и recipe.json (по умолчанию <core-dir>/release)')
    parser.add_argument('--data-dir', type=Path, default=defaults.data_dir,
                        help='постоянное хранилище галереи')
    parser.add_argument('--device', default=defaults.device, help='cpu или cuda')
    parser.add_argument('--backend', choices=['release', 'robust', 'october'], default=defaults.backend,
                        help='october — v2.0 DINOv3 + CNN (Linux); release — v1.4; robust — Windows research backend')
    parser.add_argument('--runtime-manifest', type=Path, default=defaults.runtime_manifest,
                        help='единый manifest October runtime (--backend october)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--allow-remote', action='store_true',
                        help='разрешить привязку к нелокальному адресу без ключа; '
                             'так запускается контейнер, наружу его порт не публикуется')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error('Недопустимый порт или число потоков')
    remote = args.host not in ('127.0.0.1', 'localhost', '::1')
    if remote and not defaults.api_key and not args.allow_remote:
        parser.error('Для привязки к нелокальному адресу задай VREID_API_KEY '
                     'или передай --allow-remote (внутренняя сеть контейнеров)')
    if remote and not defaults.api_key:
        print('ВНИМАНИЕ: сервис слушает ' + args.host + ' без ключа доступа. Это допустимо '
              'только во внутренней сети контейнеров; для выхода наружу задай VREID_API_KEY.',
              flush=True)

    # Рантайм офлайн: ни один путь не должен пытаться сходить в сеть за весами.
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('VREID_COMPILE', '0')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)

    core = args.core_dir.resolve()
    release = (args.release_dir or (defaults.release_dir if os.environ.get('VREID_RELEASE_DIR')
                                   else core / 'release')).resolve()
    demo = defaults.demo_dir if os.environ.get('VREID_DEMO_DIR') else core / 'demo'
    # Preserve environment settings, including demo_ui; CLI overrides only its fields.
    settings = replace(defaults, core_dir=core, release_dir=release,
                       data_dir=args.data_dir.resolve(), device=args.device, demo_dir=demo,
                       backend=args.backend, runtime_manifest=args.runtime_manifest)

    import uvicorn
    from service.app import create_app
    from service.storage import GalleryIdentityError

    print(f'ReID service: http://{args.host}:{args.port}  (OpenAPI: /docs)', flush=True)
    storage = ('PostgreSQL/pgvector'
               if settings.database_url else 'SQLite в ' + str(settings.data_dir))
    print(f'  ядро     {core}\n  релиз    {release}\n  галерея  {storage}'
          f'\n  device   {args.device}', flush=True)
    try:
        application = create_app(settings)
    except GalleryIdentityError as mismatch:
        # Это не сбой, а сработавшая защита: в хранилище лежат векторы, посчитанные другой
        # моделью или другим кропом. Молча переиспользовать их нельзя, поэтому объясняем
        # человеку, что делать, и выходим без трассировки и без бесконечного перезапуска.
        print('Галерея посчитана другой моделью или другими параметрами извлечения: '
              f'{mismatch}', flush=True)
        print('Варианты: указать другое хранилище (--data-dir или VREID_DATABASE_URL) '
              'или очистить текущее — например, docker compose down -v.', flush=True)
        return 3
    uvicorn.run(application, host=args.host, port=args.port, workers=1, log_level='info', access_log=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
