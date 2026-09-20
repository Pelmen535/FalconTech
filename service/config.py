"""Настройки сервиса. Всё берётся из окружения или из аргументов запуска —
ничего не зашито в код, кроме безопасных значений по умолчанию."""
from dataclasses import dataclass
from pathlib import Path
import os

# service/ лежит в корне репозитория, рядом с vreid/ и release/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    core_dir: Path
    release_dir: Path
    data_dir: Path
    device: str = 'cpu'
    api_key: str = ''
    max_image_bytes: int = 20 * 1024 * 1024
    max_archive_bytes: int = 128 * 1024 * 1024
    max_expanded_bytes: int = 512 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_gallery_items: int = 1500
    # Небольшой набор настоящих кадров организаторов, чтобы сервис можно было показать
    # в два клика, не имея под рукой датасета. На качество модели не влияет.
    demo_dir: Path = PROJECT_ROOT / 'demo'
    # Пустая строка — хранить галерею в SQLite внутри data_dir. Непустая — строка
    # подключения к PostgreSQL с pgvector: так работает стек из docker compose.
    database_url: str = ''

    @classmethod
    def from_env(cls):
        core = Path(os.environ.get('VREID_CORE_DIR', str(PROJECT_ROOT))).expanduser().resolve()
        release = Path(os.environ.get('VREID_RELEASE_DIR', str(core / 'release'))).expanduser().resolve()
        data = Path(os.environ.get('VREID_SERVICE_DATA', str(PROJECT_ROOT / 'var/service'))).expanduser().resolve()
        demo = Path(os.environ.get('VREID_DEMO_DIR', str(core / 'demo'))).expanduser().resolve()
        return cls(core, release, data,
                   os.environ.get('VREID_DEVICE', 'cpu'),
                   os.environ.get('VREID_API_KEY', ''),
                   demo_dir=demo,
                   database_url=os.environ.get('VREID_DATABASE_URL', ''))
