"""HTTP-сервис поверх неизменного ядра vreid: галерея, поиск, отказ, объяснение.

Разделение ответственности (ТЗ §6): здесь только транспорт и правила доступа.
Сходство, ре-ранжирование и порог отказа живут в ядре и в замороженном рецепте —
сервис их не пересчитывает и не переинтерпретирует. Совпадение ответа сервиса с
пакетным прогоном проверяется тестом `tests/service/test_core.py` на реальной сдаче.
"""
from contextlib import contextmanager, asynccontextmanager
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
from .robust import ResourceUnavailableError
from .storage import DuplicateImageError, open_store, validate_image_id
from .uploads import gallery_archive, image_box
from .observability import REQUEST_ID, event, operation
from .schemas import (responses, HealthResponse, ModelResponse, IndexResponse,
                      GalleryResponse, GalleryItemResponse, ImportResponse,
                      DemoResponse, SearchResponse, ExplainResponse, CorrespondenceResponse)

STATIC = Path(__file__).parent / "static"
BBOX_DESCRIPTION = 'Строка JSON [x,y,width,height] в пикселях исходного кадра; ширина и высота положительны.'

DESCRIPTION = """
Сервис безномерной идентификации транспортного средства. Команда «Скованные одной цепью».

* **Галерея** — наблюдения (кадр, bbox, вектор) в постоянном хранилище.
* **Поиск** — один запрос против всей галереи: топ-10 и решение об отказе.
* **Объяснение** — точное разложение косинуса пары по участкам обоих кадров.
* **Большие галереи** — режим `ann`: приближённый отбор кандидатов (HNSW), затем тот же
  точный косинус.

Уверенность — **косинусное сходство**, не вероятность. Отказ означает, что лучший кандидат
не прошёл порог уверенности модели; порог зафиксирован в её конфигурации и во время работы не
меняется. Десять ближайших наблюдений возвращаются и при отказе — для просмотра оператором.
"""

TAGS = [
    {"name": "Состояние", "description": "Состояние сервиса и параметры модели."},
    {"name": "Галерея", "description": "Наблюдения: добавление, импорт, просмотр, удаление."},
    {"name": "Поиск", "description": "Идентификация и объяснение решения."},
]


def create_app(settings=None, core=None):
    settings = settings or Settings.from_env()
    if core is None:
        if settings.backend == 'robust':
            from .robust import RobustAdapter
            core = RobustAdapter(settings.core_dir, settings.data_dir, settings.device)
        elif settings.backend == 'october':
            from .october import OctoberAdapter
            core = OctoberAdapter(settings.core_dir, settings.data_dir, settings.device,
                                  manifest=settings.runtime_manifest)
        elif settings.backend == 'release':
            core = CoreAdapter(settings.core_dir, settings.release_dir, settings.device)
        else:
            raise ValueError('Unknown service backend')
    # Галерея привязывается к отпечатку ИЗВЛЕЧЕНИЯ, а не к хэшу всего рецепта:
    # см. CoreAdapter._embedding_identity. Новый порог отказа не обесценивает векторы,
    # новые веса или иной кроп — обесценивают, и тогда база отвергается.
    store = open_store(settings, core.model_sha256,
                       getattr(core, 'embedding_sha256', core.recipe_sha256),
                       core.dimension, references_only=getattr(core,'references_only',False))
    @asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            closer = getattr(core, 'close', None)
            if closer is not None:
                closer()

    description = DESCRIPTION
    if getattr(core, 'confidence_scale', '') == 'calibrated_acceptance_score':
        description = ('Сервис безномерной идентификации ТС: галерея и независимый поиск по одному запросу. '
            'Уверенность — **калиброванный score принятия**, не вероятность. '
            'Поле cosine показывает сходство; решение об отказе определяется отдельным confidence. '
            'При отказе top-10 сохраняется. Галерейные преобразования фиксированы. '
            'ANN ограничивает пул кандидатов, сохраняя полный галерейный контекст. '
            'Диагностика чувствительности показывает эффекты закрытия областей, отдельно от ранжирования и отказа. '
            'Экспериментальные визуальные соответствия могут связывать разные детали или фон; это не доказательство идентичности. '
            'Фактическая доступность функций указана в /v1/model. Порог заморожен в конфигурации модели.')
    app = FastAPI(title='Скованные одной цепью · ReID API', version=('1.4.0' if settings.backend == 'release' else
                            '2.0.0' if settings.backend == 'october' else 'robust-20261017'),
                  description=description, openapi_tags=TAGS, lifespan=lifespan,
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

    @app.exception_handler(ResourceUnavailableError)
    async def resource_error(request, exc):
        return JSONResponse({'detail':str(exc)}, status_code=503, headers={'Retry-After':'10'})

    @app.middleware('http')
    async def request_log(request: Request, call_next):
        token = REQUEST_ID.set(uuid4().hex)
        started = time.perf_counter()
        status, error = 500, None
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers['X-Request-ID'] = REQUEST_ID.get()
            return response
        except Exception as exc:
            error = type(exc).__name__
            # Keep internal details in operation/error-type logs. The request
            # has already unwound its processing gate and storage transaction.
            return JSONResponse({'detail': 'Внутренняя ошибка сервиса.'}, status_code=500,
                                headers={'X-Request-ID': REQUEST_ID.get(),
                                         'X-Content-Type-Options': 'nosniff'})
        finally:
            route = request.scope.get('route')
            event('http.request', method=request.method, route=getattr(route, 'path', '<unmatched>'),
                  status=status, duration_ms=round((time.perf_counter()-started)*1000, 3), error_type=error)
            REQUEST_ID.reset(token)

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

    def encode(raw, box, gallery=False):
        with operation('gallery.encode' if gallery else 'query.encode'), as_file(raw) as path:
            encoder = getattr(core, 'encode_gallery', core.encode) if gallery else core.encode
            return encoder(path, box)

    def encode_records(records):
        encoder = getattr(core, 'encode_gallery_many', None)
        if encoder is None:
            for record in records:
                record['embedding'] = encode(record['image_bytes'], record['bbox_xywh'], gallery=True)
            return
        with tempfile.TemporaryDirectory(prefix='vreid-gallery-') as folder:
            rows = []
            for i, record in enumerate(records):
                path = Path(folder)/f'{i}.img'
                path.write_bytes(record['image_bytes'])
                rows.append(dict(path=str(path), bbox=record['bbox_xywh']))
            with operation('gallery.encode_batch', count=len(rows)):
                vectors = encoder(rows)
            if len(vectors) != len(records):
                raise RuntimeError('Gallery encoder returned an invalid count')
            for record, vector in zip(records, vectors):
                record['embedding'] = vector

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
            '<title>Документация API · Безномерная идентификация ТС</title>'
            '<link rel="icon" href="data:image/svg+xml,<svg xmlns=%27http://www.w3.org/2000/svg%27 '
            'viewBox=%270 0 32 32%27><text y=%2726%27 font-size=%2726%27>%F0%9F%9A%97</text></svg>">'
            '<link rel="stylesheet" href="/static/vendor/swagger-ui.css">'
            '<style>body{margin:0;background:#fff}</style></head><body>'
            '<div id="swagger"></div>'
            '<script src="/static/vendor/swagger-ui-bundle.js"></script>'
            '<script>window.ui=SwaggerUIBundle({url:"/openapi.json",dom_id:"#swagger",'
            'deepLinking:true,docExpansion:"list",defaultModelsExpandDepth:0});</script>'
            '</body></html>')

    # ------------------------------------------------------------------ состояние
    @app.get('/health', tags=['Состояние'], summary='Состояние сервиса',
             responses=responses(HealthResponse, authenticated=False))
    def health():
        meta = core.metadata()
        return dict(status='ok', gallery_count=store.count(), model_loaded=meta.get('loaded', False),
                    dimension=core.dimension, model_sha256=core.model_sha256, recipe_sha256=core.recipe_sha256,
                    device=settings.device, storage=getattr(store, 'backend', 'unknown'),
                    busy=gate.locked())

    @app.get('/v1/model', tags=['Состояние'], summary='Параметры загруженной модели',
             responses=responses(ModelResponse))
    def model():
        return {**core.metadata(), 'bbox_format':'xywh',
                'confidence_scale':getattr(core, 'confidence_scale', 'raw_cosine')}

    @app.get('/v1/index/stats', tags=['Состояние'], summary='Индекс поиска и его масштабируемость',
             responses=responses(IndexResponse))
    def index_stats():
        """Текущий режим поиска и измеренные цифры ANN на галереях до 10^6.

        Числа берутся из results/ann_scalability.json (scripts/ann_benchmark.py). Если файла
        нет, поле benchmark пустое — сервис не выдумывает измерений, которых не делал.
        """
        count = store.count()
        if getattr(core,'manages_ann',False):
            stats=core.index_stats()
            return {'gallery_count':count,'active_mode':'ann_available' if stats['available'] and count>=stats['min_items_for_ann'] else 'exact',
                    'exact_search':'full fixed-gallery October ensemble','ann':stats,'benchmark':None}
        if not getattr(core, 'supports_ann', True):
            return {'gallery_count': count, 'active_mode': 'exact',
                    'exact_search': 'полный поиск по замороженной галерее',
                    'ann': {**app.state.index.stats(), 'available': False}, 'benchmark': None}
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
            'exact_search': 'полный перебор float32 по всей галерее',
            'ann': app.state.index.stats(),
            'benchmark': report,
        }

    # ------------------------------------------------------------------ демо
    def require_demo_ui() -> None:
        if not settings.demo_ui:
            raise HTTPException(404, 'Демонстрационные кнопки выключены; включаются VREID_DEMO_UI=1.')

    def demo_query_path() -> Path:
        require_demo_ui()
        meta = settings.demo_dir / 'query.json'
        if not meta.is_file():
            raise HTTPException(404, 'Демонстрационный набор не поставлен с этой сборкой.')
        return meta

    # Демо-эндпоинты в спецификации только тогда, когда включены: иначе это шум в Swagger
    @app.get('/v1/demo', tags=['Галерея'], summary='Есть ли демонстрационный набор',
             include_in_schema=settings.demo_ui, responses=responses(DemoResponse, errors=[404]))
    def demo_info():
        """Небольшой демонстрационный набор: 20 наблюдений и один запрос.

        Нужен, чтобы сервис можно было показать сразу после запуска. Это проверка
        исполнения, а не измерение качества модели.

        В репозиторий архив не кладётся: он собирается из кадров, выданных организаторами,
        а публиковать их мы не вправе. Собрать на месте — одна команда, она же в подсказке
        ниже: у того, кому выданы данные, этот набор и так есть.
        """
        if not settings.demo_ui:
            # Выключено — интерфейс молчит: ни кнопок, ни подсказки собрать набор
            return {'available': False}
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

    @app.get('/v1/demo/query.jpg', tags=['Галерея'], summary='Демонстрационный кадр-запрос',
             include_in_schema=settings.demo_ui, responses=responses(errors=[404], binary=True))
    def demo_query_image():
        meta = json.loads(demo_query_path().read_text(encoding='utf-8'))
        target = (settings.demo_dir / meta['image']).resolve()
        if not target.is_file() or not target.is_relative_to(settings.demo_dir.resolve()):
            raise HTTPException(404, 'Демонстрационный кадр не найден.')
        return FileResponse(target, media_type='image/jpeg')

    @app.post('/v1/demo/gallery', status_code=201, tags=['Галерея'],
              summary='Загрузить демонстрационную галерею', include_in_schema=settings.demo_ui,
              responses=responses(ImportResponse, status=201, errors=[404,409,422,503]))
    def demo_gallery():
        """Импортирует demo/gallery.zip. Если галерея уже не пуста — не трогает её."""
        require_demo_ui()
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
            if store.count():
                raise HTTPException(409, 'Галерея не пуста: демонстрационный набор не добавляю.')
            encode_records(records)
            store.add_many(records)
        return dict(added=len(records), count=store.count())

    # ------------------------------------------------------------------ галерея
    @app.get('/v1/gallery', tags=['Галерея'], summary='Список наблюдений',
             responses=responses(GalleryResponse, errors=[422]))
    def gallery(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
        return dict(count=store.count(), items=[item_url(x) for x in store.list_items(offset, limit)])

    @app.post('/v1/gallery/items', status_code=201, tags=['Галерея'],
              summary='Добавить одно наблюдение',
              responses=responses(GalleryItemResponse, status=201, errors=[409,413,422,503]))
    def add_item(image: UploadFile = File(...), bbox: str = Form(..., description=BBOX_DESCRIPTION), image_id: str | None = Form(None)):
        raw = image.file.read(settings.max_image_bytes + 1)
        box = image_box(raw, bbox, settings)
        key = validate_image_id(image_id if image_id is not None else uuid4().hex)
        with processing():
            if store.count() >= settings.max_gallery_items:
                raise HTTPException(409, 'Достигнут лимит галереи сервиса.')
            if store.exists(key): raise DuplicateImageError('image_id уже есть в галерее: ' + key)
            vector = encode(raw, box, gallery=True)
            return item_url(store.add(key, raw, box, vector))

    @app.post('/v1/gallery/import', status_code=201, tags=['Галерея'],
              summary='Импортировать ZIP с test_gallery.csv и images/',
              responses=responses(ImportResponse, status=201, errors=[409,413,422,503]))
    def import_gallery(archive: UploadFile = File(...)):
        raw = archive.file.read(settings.max_archive_bytes + 1)
        records = gallery_archive(raw, settings)
        with processing():
            if store.count() + len(records) > settings.max_gallery_items:
                raise HTTPException(409, 'Импорт превышает лимит галереи.')
            if store.existing_ids(r['image_id'] for r in records):
                raise DuplicateImageError('Часть image_id уже есть в галерее; импорт отменён целиком.')
            encode_records(records)
            store.add_many(records)
            return dict(added=len(records), count=store.count())

    @app.get('/v1/gallery/items/{image_id}/image', tags=['Галерея'],
             summary='Кадр наблюдения или миниатюра автомобиля',
             responses=responses(errors=[404,422], binary=True))
    def gallery_image(image_id: str, crop: bool = Query(False)):
        try:
            raw,frame_bbox = store.frame(image_id)
        except KeyError: raise HTTPException(404, 'Изображение не найдено.')
        with Image.open(BytesIO(raw)) as im:
            if crop:
                x, y, w, h = frame_bbox
                view = im.crop((int(x), int(y), int(x + w), int(y + h))).convert('RGB')
                view.thumbnail((512, 512))
                output = BytesIO(); view.save(output, format='JPEG', quality=85)
                return Response(output.getvalue(), media_type='image/jpeg', headers={'Cache-Control':'no-store'})
            media = {'JPEG':'image/jpeg', 'PNG':'image/png', 'WEBP':'image/webp'}.get(im.format, 'application/octet-stream')
        return Response(raw, media_type=media, headers={'Cache-Control':'no-store'})

    @app.delete('/v1/gallery/items/{image_id}', status_code=204, tags=['Галерея'],
                summary='Удалить наблюдение', responses=responses(status=204, errors=[404,422,503]))
    def delete_item(image_id: str):
        with processing():
            if not store.delete(image_id): raise HTTPException(404, 'Запись не найдена.')
        return Response(status_code=204)

    # ------------------------------------------------------------------ поиск
    @app.post('/v1/search', tags=['Поиск'], summary='Найти автомобиль в галерее',
              responses=responses(SearchResponse, errors=[409,413,422,503]))
    def search(image: UploadFile = File(...), bbox: str = Form(..., description=BBOX_DESCRIPTION),
               query_id: str | None = Form(None, description='ID наблюдения запроса. Если задан, кадры с тем же префиксом до # исключаются из кандидатов.'),
               mode: str = Query('exact', pattern='^(exact|ann)$',
                                 description='exact — полный перебор галереи; '
                                             'ann — HNSW-шортлист, затем тот же точный косинус')):
        raw = image.file.read(settings.max_image_bytes + 1)
        box = image_box(raw, bbox, settings)
        started = time.perf_counter()
        notes = []
        if mode == 'ann' and not getattr(core, 'supports_ann', True):
            raise HTTPException(422, 'Эта модель требует полную галерею: используйте mode=exact.')
        with processing():
            ids, vectors = store.snapshot()
            if len(ids) < 10:
                raise HTTPException(409, 'Для поиска в галерее должно быть не меньше 10 наблюдений.')
            embedding = encode(raw, box)
            subset, shortlist_seconds = None, None
            if mode == 'ann' and not getattr(core,'manages_ann',False):
                try:
                    picked, shortlist_seconds = app.state.index.shortlist(
                        ids, vectors, embedding, size=max(200, 10 * 10))
                    subset = sorted(int(i) for i in picked)
                except AnnUnavailable as reason:
                    notes.append(f'Выполнен точный поиск: {reason}.')
            if getattr(core,'manages_ann',False):
                with operation('search.rank',gallery_count=len(ids)):
                    result=core.rank(embedding,vectors,ids,query_id=query_id,mode=mode)
                notes.extend(result.get('notes',[]));shortlist_seconds=result.get('shortlist_seconds')
            elif subset is not None and len(subset) >= 10:
                # Ре-ранжирование внутри шортлиста: сходство считается так же, меняется
                # только состав кандидатов, дошедших до этого шага.
                with operation('search.rank', gallery_count=len(subset)):
                    result = core.rank(embedding, vectors[subset], [ids[i] for i in subset], query_id=query_id)
                notes.append(f'Быстрый поиск: отобрано {len(subset)} из {len(ids)} наблюдений.')
            else:
                with operation('search.rank', gallery_count=len(ids)):
                    result = core.rank(embedding, vectors, ids, query_id=query_id)
        event('search.result', accepted=bool(result['accepted']), gallery_count=len(ids),
              confidence_scale=getattr(core, 'confidence_scale', 'raw_cosine'))
        return {**result, 'ranking':[item_url(x) for x in result['ranking']],
                'candidate':item_url(result['candidate']) if result['candidate'] else None,
                'confidence_scale':getattr(core, 'confidence_scale', 'raw_cosine'), 'gallery_count':len(ids),
                'search_mode': result.get('search_mode','ann' if subset is not None and len(subset) >= 10 else 'exact'),
                'shortlist_seconds': None if shortlist_seconds is None else round(shortlist_seconds, 5),
                'notes': notes,
                'supports_correspondence':getattr(core,'supports_correspondence',False),
                'model_sha256':core.model_sha256, 'recipe_sha256':core.recipe_sha256,
                'seconds':round(time.perf_counter() - started, 4)}

    @app.post('/v1/correspondences', tags=['Поиск'],
              summary='Экспериментальные визуальные соответствия участков',
              responses=responses(CorrespondenceResponse, errors=[404,413,422,501,503]))
    def correspondences(image: UploadFile = File(...), bbox: str = Form(..., description=BBOX_DESCRIPTION),
                        gallery_id: str = Form(...)):
        """CPU DINOv3 patch matching. Может ошибаться; не доказывает идентичность
        и не объясняет confidence/refusal. Не меняет поиск и не сохраняет признаки.
        """
        matcher = getattr(core,'correspondences',None)
        if matcher is None or not getattr(core,'supports_correspondence',False):
            raise HTTPException(501,'Загруженное ядро не поддерживает экспериментальные соответствия.')
        raw = image.file.read(settings.max_image_bytes + 1)
        box = image_box(raw,bbox,settings)
        started = time.perf_counter()
        with processing():
            try:
                partner_bytes,partner_bbox = store.frame(gallery_id)
            except KeyError:
                raise HTTPException(404,'Наблюдение галереи не найдено.')
            with as_file(raw,'query.img') as query_path, as_file(partner_bytes,'gallery.img') as partner_path:
                with operation('diagnostic.correspondence'):
                    result = matcher(query_path,box,partner_path,partner_bbox)
        return {**result,'gallery_id':gallery_id,'seconds':round(time.perf_counter()-started,4)}

    @app.post('/v1/explain', tags=['Поиск'],
              summary='Диагностическая чувствительность сходства пары',
              responses=responses(ExplainResponse, errors=[404,413,422,501,503]))
    def explain(image: UploadFile = File(...), bbox: str = Form(..., description=BBOX_DESCRIPTION),
                gallery_id: str = Form(...), regions: int = Query(3, ge=0, le=16)):
        """Объяснение сходства пары через рабочую модель.

        October возвращает независимые эффекты закрытия областей при фиксированной
        галерее; они неаддитивны и не являются соответствиями деталей двух кадров.
        Карта относится только к сходству, отдельно от ранжирования и отказа.
        """
        explainer = getattr(core, 'explain', None)
        if explainer is None:
            raise HTTPException(501, 'Загруженное ядро не поддерживает объяснение.')
        raw = image.file.read(settings.max_image_bytes + 1)
        box = image_box(raw, bbox, settings)
        started = time.perf_counter()
        with processing():
            try:
                partner_bytes,partner_bbox=store.frame(gallery_id)
                partner={'bbox_xywh':partner_bbox}
            except KeyError:
                raise HTTPException(404, 'Наблюдение галереи не найдено.')
            with as_file(raw, 'query.img') as query_path, \
                 as_file(partner_bytes, 'gallery.img') as partner_path:
                if getattr(core,'manages_ann',False):
                    ids,vectors=store.snapshot()
                    result=explainer(query_path,box,partner_path,partner['bbox_xywh'],regions=regions,
                                     gallery_vectors=vectors,gallery_ids=ids,gallery_id=gallery_id)
                else:
                    result = explainer(query_path, box, partner_path, partner['bbox_xywh'],regions=regions)
        sides = {name: {**{k: v for k, v in side.items() if k != 'png'},
                        'heatmap_png_base64': base64.b64encode(side['png']).decode('ascii')}
                 for name, side in result['sides'].items()}
        return {**{k: v for k, v in result.items() if k != 'sides'},
                'gallery_id': gallery_id, 'sides': sides,
                'seconds': round(time.perf_counter() - started, 4)}

    original_openapi = app.openapi

    def documented_openapi():
        schema = original_openapi()
        if settings.api_key:
            schemes = schema.setdefault('components', {}).setdefault('securitySchemes', {})
            schemes['ApiKeyAuth'] = {'type': 'apiKey', 'in': 'header', 'name': 'X-API-Key'}
            for path, operations in schema['paths'].items():
                if path.startswith('/v1/'):
                    for method, definition in operations.items():
                        if method in {'get','post','put','patch','delete','head','options'}:
                            definition.setdefault('security', [{'ApiKeyAuth': []}])
        return schema

    app.openapi = documented_openapi
    return app
