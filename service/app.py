"""HTTP-сервис поверх неизменного ядра vreid: галерея, поиск, отказ, объяснение.

Разделение ответственности (ТЗ §6): здесь только транспорт и правила доступа.
Сходство, ре-ранжирование и порог отказа живут в ядре и в замороженном рецепте —
сервис их не пересчитывает и не переинтерпретирует. Совпадение ответа сервиса с
конкурсным CLI проверяется тестом `tests/service/test_core.py` на реальной сдаче.
"""
from contextlib import contextmanager
from io import BytesIO
import base64
import json
import secrets
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image

from .ann import AnnUnavailable, ShortlistIndex
from .config import Settings
from .core import CoreAdapter
from .storage import DuplicateImageError, open_store
from .uploads import gallery_archive, image_box

STATIC = Path(__file__).parent / "static"

DESCRIPTION = """
Сервис безномерной идентификации транспортного средства. Команда «Связанные одной цепью».

* **Галерея** — наблюдения (кадр, bbox, вектор) в постоянном хранилище.
* **Поиск** — один запрос против всей галереи: топ-10 и решение об отказе.
* **Объяснение** — точное разложение косинуса пары по участкам обоих кадров.
* **Масштабируемость** — режим `ann` использует HNSW-шортлист вместо полного перебора.

Уверенность — **сырой косинус**, не вероятность. Отказ означает, что лучший кандидат
не прошёл замороженный порог релиза; ранжирование при этом всё равно возвращается.
"""

TAGS = [
    {"name": "Состояние", "description": "Здоровье процесса и параметры загруженного релиза."},
    {"name": "Галерея", "description": "Наблюдения: добавление, импорт, просмотр, удаление."},
    {"name": "Поиск", "description": "Идентификация и объяснение решения."},
]


def create_app(settings=None, core=None):
    settings = settings or Settings.from_env()
    core = core or CoreAdapter(settings.core_dir, settings.release_dir, settings.device)
    # Галерея привязывается к отпечатку ИЗВЛЕЧЕНИЯ, а не к хэшу всего рецепта:
    # см. CoreAdapter._embedding_identity. Новый порог отказа не обесценивает векторы,
    # новые веса или иной кроп — обесценивают, и тогда база отвергается.
    store = open_store(settings, core.model_sha256,
                       getattr(core, 'embedding_sha256', core.recipe_sha256),
                       core.dimension)
    app = FastAPI(title='Связанные одной цепью · ReID API', version='1.1.0',
                  description=DESCRIPTION, openapi_tags=TAGS,
                  docs_url=None, redoc_url=None)
    app.state.core = core
    app.state.store = store
    app.state.settings = settings
    app.state.index = ShortlistIndex()
    gate = threading.Lock()

    @contextmanager
    def processing():
        if not gate.acquire(blocking=False):
            raise HTTPException(503, 'Модель занята другим запросом. Повторите после его завершения.', headers={'Retry-After':'2'})
        try: yield
        finally: gate.release()

    @app.middleware('http')
    async def limits_and_auth(request: Request, call_next):
        if request.url.path.startswith('/v1/') and settings.api_key:
            supplied = request.headers.get('x-api-key', '')
            if not secrets.compare_digest(supplied, settings.api_key):
                return JSONResponse({'detail':'Требуется корректный X-API-Key.'}, status_code=401)
        length = request.headers.get('content-length')
        if length:
            try:
                if int(length) > settings.max_archive_bytes + 1024 * 1024:
                    return JSONResponse({'detail':'Превышен размер запроса.'}, status_code=413)
            except ValueError:
                return JSONResponse({'detail':'Некорректный Content-Length.'}, status_code=400)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.exception_handler(DuplicateImageError)
    async def duplicate_error(request, exc):
        return JSONResponse({'detail':str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({'detail':str(exc)}, status_code=422)

    def item_url(item):
        url = '/v1/gallery/items/' + quote(item['image_id'], safe='') + '/image'
        return {**item, 'image_url':url, 'thumbnail_url':url + '?crop=true'}

    @contextmanager
    def as_file(raw: bytes, name: str = 'frame.img'):
        """Ядро читает кадр с диска — отдаём ему временный файл, а не переписываем ядро."""
        with tempfile.TemporaryDirectory(prefix='vreid-') as folder:
            path = Path(folder) / name
            path.write_bytes(raw)
            yield path

    def encode(raw, box):
        with as_file(raw) as path:
            return core.encode(path, box)

    # ------------------------------------------------------------------ страницы
    @app.get('/', include_in_schema=False)
    def page():
        return FileResponse(STATIC / 'index.html', media_type='text/html')

    @app.get('/static/{name:path}', include_in_schema=False)
    def static_file(name: str):
        """Отдаём только файлы из service/static: никаких путей наружу."""
        target = (STATIC / name).resolve()
        if not target.is_file() or not target.is_relative_to(STATIC.resolve()):
            raise HTTPException(404, 'Файл не найден.')
        media = {'.js':'application/javascript', '.css':'text/css', '.png':'image/png',
                 '.svg':'image/svg+xml', '.json':'application/json',
                 '.html':'text/html'}.get(target.suffix, 'application/octet-stream')
        return FileResponse(target, media_type=media)

    @app.get('/docs', include_in_schema=False)
    def docs():
        """Swagger UI со скачанными заранее файлами: страница работает без интернета."""
        return HTMLResponse(
            '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<title>ReID API · OpenAPI</title>'
            '<link rel="stylesheet" href="/static/vendor/swagger-ui.css">'
            '<style>body{margin:0;background:#fff}</style></head><body>'
            '<div id="swagger"></div>'
            '<script src="/static/vendor/swagger-ui-bundle.js"></script>'
            '<script>window.ui=SwaggerUIBundle({url:"/openapi.json",dom_id:"#swagger",'
            'deepLinking:true,docExpansion:"list",defaultModelsExpandDepth:0});</script>'
            '</body></html>')

    # ------------------------------------------------------------------ состояние
    @app.get('/health', tags=['Состояние'], summary='Здоровье процесса')
    def health():
        meta = core.metadata()
        return dict(status='ok', gallery_count=store.count(), model_loaded=meta.get('loaded', False),
                    dimension=core.dimension, model_sha256=core.model_sha256, recipe_sha256=core.recipe_sha256,
                    device=settings.device, storage=getattr(store, 'backend', 'unknown'),
                    busy=gate.locked())

    @app.get('/v1/model', tags=['Состояние'], summary='Параметры загруженного релиза')
    def model():
        return {**core.metadata(), 'bbox_format':'xywh', 'confidence_scale':'raw_cosine'}

    @app.get('/v1/index/stats', tags=['Состояние'], summary='Индекс поиска и его масштабируемость')
    def index_stats():
        """Текущий режим поиска и измеренные цифры ANN на галереях до 10^6.

        Числа берутся из results/ann_scalability.json (scripts/ann_benchmark.py). Если файла
        нет, поле benchmark пустое — сервис не выдумывает измерений, которых не делал.
        """
        count = store.count()
        report = None
        candidate = settings.core_dir / 'results/ann_scalability.json'
        if candidate.is_file():
            try:
                report = json.loads(candidate.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                report = None
        return {
            'gallery_count': count,
            'active_mode': 'exact' if count < app.state.index.stats()['min_items_for_ann'] else 'ann_available',
            'exact_search': 'полный перебор float32; результат совпадает с конкурсным CLI',
            'ann': app.state.index.stats(),
            'benchmark': report,
        }

    # ------------------------------------------------------------------ демо
    def demo_query_path() -> Path:
        meta = settings.demo_dir / 'query.json'
        if not meta.is_file():
            raise HTTPException(404, 'Демонстрационный набор не поставлен с этой сборкой.')
        return meta

    @app.get('/v1/demo', tags=['Галерея'], summary='Есть ли демонстрационный набор')
    def demo_info():
        """Небольшой демонстрационный набор: 20 наблюдений и один запрос.

        Нужен, чтобы сервис можно было показать сразу после запуска. Это проверка
        исполнения, а не измерение качества модели.

        В репозиторий архив не кладётся: он собирается из кадров, выданных организаторами,
        а публиковать их мы не вправе. Собрать на месте — одна команда, она же в подсказке
        ниже: у проверяющего этот набор есть, он его и выдавал.
        """
        meta = settings.demo_dir / 'query.json'
        archive = settings.demo_dir / 'gallery.zip'
        if not meta.is_file() or not archive.is_file():
            return {'available': False,
                    'hint': 'python scripts/make_demo_gallery.py --data <папка с данными>',
                    'why': ('демонстрационный набор собирается из выданных организаторами '
                            'кадров и в репозиторий не выкладывается')}
        query = json.loads(meta.read_text(encoding='utf-8'))
        return {'available': True, 'image_url': '/v1/demo/query.jpg',
                'image_id': query.get('image_id'), 'bbox': query.get('bbox'),
                'gallery_items': 20,
                'note': '20 кадров из выданного набора; демонстрация работы, не оценка точности'}

    @app.get('/v1/demo/query.jpg', tags=['Галерея'], summary='Демонстрационный кадр-запрос')
    def demo_query_image():
        meta = json.loads(demo_query_path().read_text(encoding='utf-8'))
        target = (settings.demo_dir / meta['image']).resolve()
        if not target.is_file() or not target.is_relative_to(settings.demo_dir.resolve()):
            raise HTTPException(404, 'Демонстрационный кадр не найден.')
        return FileResponse(target, media_type='image/jpeg')

    @app.post('/v1/demo/gallery', status_code=201, tags=['Галерея'],
              summary='Загрузить демонстрационную галерею')
    def demo_gallery():
        """Импортирует demo/gallery.zip. Если галерея уже не пуста — не трогает её."""
        archive = settings.demo_dir / 'gallery.zip'
        if not archive.is_file():
            raise HTTPException(404, 'Демонстрационного архива нет: он не поставляется с '
                                     'репозиторием, потому что собран из кадров организаторов. '
                                     'Соберите его одной командой: '
                                     'python scripts/make_demo_gallery.py --data <папка с данными>')
        if store.count():
            raise HTTPException(409, 'Галерея не пуста: демонстрационный набор не добавляю, '
                                     'чтобы не смешать его с вашими данными.')
        records = gallery_archive(archive.read_bytes(), settings)
        with processing():
            for record in records:
                record['embedding'] = encode(record['image_bytes'], record['bbox_xywh'])
            store.add_many(records)
        return dict(added=len(records), count=store.count())

    # ------------------------------------------------------------------ галерея
    @app.get('/v1/gallery', tags=['Галерея'], summary='Список наблюдений')
    def gallery(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
        return dict(count=store.count(), items=[item_url(x) for x in store.list_items(offset, limit)])

    @app.post('/v1/gallery/items', status_code=201, tags=['Галерея'],
              summary='Добавить одно наблюдение')
    def add_item(image: UploadFile = File(...), bbox: str = Form(...), image_id: str | None = Form(None)):
        raw = image.file.read(settings.max_image_bytes + 1)
        box = image_box(raw, bbox, settings)
        key = image_id or uuid4().hex
        with processing():
            if store.count() >= settings.max_gallery_items:
                raise HTTPException(409, 'Достигнут лимит галереи сервиса.')
            ids, _ = store.snapshot()
            if key in ids: raise DuplicateImageError('image_id уже есть в галерее: ' + key)
            vector = encode(raw, box)
            return item_url(store.add(key, raw, box, vector))

    @app.post('/v1/gallery/import', status_code=201, tags=['Галерея'],
              summary='Импортировать ZIP с test_gallery.csv и images/')
    def import_gallery(archive: UploadFile = File(...)):
        raw = archive.file.read(settings.max_archive_bytes + 1)
        records = gallery_archive(raw, settings)
        with processing():
            if store.count() + len(records) > settings.max_gallery_items:
                raise HTTPException(409, 'Импорт превышает лимит галереи.')
            ids, _ = store.snapshot()
            if set(ids).intersection(r['image_id'] for r in records):
                raise DuplicateImageError('Часть image_id уже есть в галерее; импорт отменён целиком.')
            for record in records:
                record['embedding'] = encode(record['image_bytes'], record['bbox_xywh'])
            store.add_many(records)
            return dict(added=len(records), count=store.count())

    @app.get('/v1/gallery/items/{image_id}/image', tags=['Галерея'],
             summary='Кадр наблюдения или миниатюра автомобиля')
    def gallery_image(image_id: str, crop: bool = Query(False)):
        try:
            raw = store.image(image_id)
            item = store.get_item(image_id) if crop else None
        except KeyError: raise HTTPException(404, 'Изображение не найдено.')
        with Image.open(BytesIO(raw)) as im:
            if crop:
                x, y, w, h = item['bbox_xywh']
                view = im.crop((int(x), int(y), int(x + w), int(y + h))).convert('RGB')
                view.thumbnail((512, 512))
                output = BytesIO(); view.save(output, format='JPEG', quality=85)
                return Response(output.getvalue(), media_type='image/jpeg', headers={'Cache-Control':'no-store'})
            media = {'JPEG':'image/jpeg', 'PNG':'image/png', 'WEBP':'image/webp'}.get(im.format, 'application/octet-stream')
        return Response(raw, media_type=media, headers={'Cache-Control':'no-store'})

    @app.delete('/v1/gallery/items/{image_id}', status_code=204, tags=['Галерея'],
                summary='Удалить наблюдение')
    def delete_item(image_id: str):
        with processing():
            if not store.delete(image_id): raise HTTPException(404, 'Запись не найдена.')
        return Response(status_code=204)

    # ------------------------------------------------------------------ поиск
    @app.post('/v1/search', tags=['Поиск'], summary='Найти автомобиль в галерее')
    def search(image: UploadFile = File(...), bbox: str = Form(...),
               query_id: str | None = Form(None),
               mode: str = Query('exact', pattern='^(exact|ann)$',
                                 description='exact — полный перебор (как в конкурсном CLI); '
                                             'ann — HNSW-шортлист, затем тот же точный косинус')):
        raw = image.file.read(settings.max_image_bytes + 1)
        box = image_box(raw, bbox, settings)
        started = time.perf_counter()
        notes = []
        with processing():
            ids, vectors = store.snapshot()
            if len(ids) < 10:
                raise HTTPException(409, 'Для поиска нужны минимум 10 изображений в галерее.')
            embedding = encode(raw, box)
            subset, shortlist_seconds = None, None
            if mode == 'ann':
                try:
                    picked, shortlist_seconds = app.state.index.shortlist(
                        ids, vectors, embedding, size=max(200, 10 * 10))
                    subset = sorted(int(i) for i in picked)
                except AnnUnavailable as reason:
                    notes.append(f'ANN не применён: {reason}. Отдан точный перебор.')
            if subset is not None and len(subset) >= 10:
                # Ре-ранжирование внутри шортлиста: сходство считается так же, меняется
                # только состав кандидатов, дошедших до этого шага.
                result = core.rank(embedding, vectors[subset], [ids[i] for i in subset],
                                   query_id=query_id)
                notes.append(f'Шортлист HNSW: {len(subset)} из {len(ids)} записей.')
            else:
                result = core.rank(embedding, vectors, ids, query_id=query_id)
        return {**result, 'ranking':[item_url(x) for x in result['ranking']],
                'candidate':item_url(result['candidate']) if result['candidate'] else None,
                'confidence_scale':'raw_cosine', 'gallery_count':len(ids),
                'search_mode': 'ann' if subset is not None and len(subset) >= 10 else 'exact',
                'shortlist_seconds': None if shortlist_seconds is None else round(shortlist_seconds, 5),
                'notes': notes,
                'model_sha256':core.model_sha256, 'recipe_sha256':core.recipe_sha256,
                'seconds':round(time.perf_counter() - started, 4)}

    @app.post('/v1/explain', tags=['Поиск'],
              summary='Чем именно похожи запрос и кандидат')
    def explain(image: UploadFile = File(...), bbox: str = Form(...),
                gallery_id: str = Form(...), regions: int = Query(3, ge=0, le=16)):
        """Точное аддитивное разложение косинуса по участкам обоих кадров.

        Возвращает по PNG на сторону (наложение на кроп) и самые весомые клетки сетки.
        Это не Grad-CAM: сумма вкладов патчей плюс постоянная часть в точности равна
        косинусу, и поле reconstruction_error это показывает.
        """
        explainer = getattr(core, 'explain', None)
        if explainer is None:
            raise HTTPException(501, 'Загруженное ядро не поддерживает объяснение.')
        raw = image.file.read(settings.max_image_bytes + 1)
        box = image_box(raw, bbox, settings)
        try:
            partner_bytes = store.image(gallery_id)
            partner = store.get_item(gallery_id)
        except KeyError:
            raise HTTPException(404, 'Наблюдение галереи не найдено.')
        started = time.perf_counter()
        with processing():
            with as_file(raw, 'query.img') as query_path, \
                 as_file(partner_bytes, 'gallery.img') as partner_path:
                result = explainer(query_path, box, partner_path, partner['bbox_xywh'],
                                   regions=regions)
        sides = {name: {**{k: v for k, v in side.items() if k != 'png'},
                        'heatmap_png_base64': base64.b64encode(side['png']).decode('ascii')}
                 for name, side in result['sides'].items()}
        return {**{k: v for k, v in result.items() if k != 'sides'},
                'gallery_id': gallery_id, 'sides': sides,
                'seconds': round(time.perf_counter() - started, 4)}

    return app
