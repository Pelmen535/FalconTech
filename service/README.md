# HTTP-сервис и хранение v2.0

Публичный запуск из корня: `docker compose up --build`, интерфейс `http://localhost:8080`, документация `/docs`, схема `/openapi.json`. Канонический backend — `october`; точная выбранная модель и ресурсы определены в `release_current/runtime-manifest.json`.

PostgreSQL в Compose хранит наблюдения и ссылки `feature_ref` схемы `raw-pair-le-v2`; сырые признаки лежат в каталоге данных и связаны namespace/контрольными суммами. Хранить одну только SQL-БД для восстановления недостаточно.

Основные операции: импорт ZIP или одиночного изображения с XYWH, поиск с top‑10 и отказом, explain выбранной пары, экспорт CSV/JSON, удаление наблюдения. Действующий контракт полей и кодов ошибок задаёт OpenAPI. Неподдерживаемые малые/sparse multibox-галереи отклоняются явно; нет обещания корректного ответа на любом числе допустимых соседей. Ошибки воркера возвращаются в JSON с request ID; перегрузка и нехватка ресурсов не превращаются в фиктивный результат.

## Резервная копия

Команда читает SQL согласованным readonly-снимком и сохраняет необходимые raw-features и manifest:

```bash
docker compose exec api python -B scripts/gallery_admin.py backup \
  --postgres --data-dir /var/lib/vreid \
  --runtime-manifest /app/release_current/runtime-manifest.json \
  --out /var/lib/vreid/backups/gallery-001.zip
```

Имя выходного файла должно быть новым. Для SQLite уберите `--postgres` и укажите её каталог данных.

## Холодное восстановление и перенос

```bash
python -B scripts/gallery_admin.py restore \
  --backup /backups/gallery-001.zip --target-dir /data/new-gallery
```

Это восстановление в **новый** каталог SQLite. Для PostgreSQL добавьте `--postgres` и задайте `VREID_DATABASE_URL` отдельной пустой целевой БД. После проверки подключайте новый каталог вместе с его целевой БД; прежние каталог и БД остаются для отката.

Миграция v1.4 переизвлекает новые признаки из сохранённых изображений, не передаёт старые вектора новой модели:

```bash
python -B scripts/gallery_admin.py migrate \
  --source-dir /data/v14-gallery --target-dir /data/v2-gallery \
  --runtime-manifest release_current/runtime-manifest.json --device cuda
```

Миграция читает источник и создаёт отдельный результат. CPU и GPU используют разные численные пространства; перенос между ними также требует явной миграции. Эти команды запускаются в зафиксированной Linux-среде релиза, с доступом к соответствующим каталогам.

Подтверждённые реальные проверки: GPU HTTP/CLI на688G, PostgreSQL20G, перезапуск, сохранение raw-references, readonly backup и cold restore в SQLite и отдельный PostgreSQL, миграция галереи v1.4 на 16 наблюдениях. Полный охват и незакрытые условия — [отчёт](../docs/V2_A5000_REPORT_RU.md). Большие произвольные галереи и точный CUDA12.2-стенд организаторов этим не подтверждены.
