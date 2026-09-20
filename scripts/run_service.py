"""Запуск HTTP-сервиса поверх ядра из этого же репозитория.

    python scripts/run_service.py --device cuda            # локально, с GPU
    python scripts/run_service.py --device cpu             # без GPU
    docker compose up                                      # всё вместе, одной командой

По умолчанию слушает только 127.0.0.1. Привязка к другому адресу требует VREID_API_KEY —
иначе сервис с загруженной моделью оказался бы открыт всей сети без единой проверки.
"""
from __future__ import annotations

import argparse
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
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--allow-remote', action='store_true',
                        help='разрешить привязку к нелокальному адресу без ключа; '
                             'так запускается контейнер, наружу его порт не публикуется')
    args = parser.parse_args()

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
    release = (args.release_dir or core / 'release').resolve()
    settings = Settings(core, release, args.data_dir.resolve(), args.device, defaults.api_key,
                        demo_dir=(core / 'demo'),
                        # Строка подключения приходит из окружения: пустая — SQLite в
                        # data_dir, непустая — PostgreSQL с pgvector (стек docker compose).
                        database_url=defaults.database_url)

    import uvicorn
    from service.app import create_app
    from service.storage import GalleryIdentityError

    print(f'ReID service: http://{args.host}:{args.port}  (OpenAPI: /docs)', flush=True)
    storage = ('PostgreSQL/pgvector ' + settings.database_url.split('@')[-1]
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
    uvicorn.run(application, host=args.host, port=args.port, workers=1, log_level='info')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
