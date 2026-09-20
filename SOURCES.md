# Происхождение внешних весов и данных

Требование организаторов (ответ 46): для любых внешних весов и датасетов нужны однозначный
URL, автор и версия/тег; приватные или недоступные проверяющим источники запрещены. Ответ 39
дополнительно требует точных версий пакетов и запрещает ссылки на подвижные теги.

## 1. Предобученные веса, от которых шло дообучение

| | |
|---|---|
| Модель | DINOv2 ViT-B/14 (distilled), вход 518, патч 14 |
| Идентификатор в timm | `vit_base_patch14_dinov2.lvd142m` |
| Источник весов | Hugging Face Hub, репозиторий `timm/vit_base_patch14_dinov2.lvd142m` |
| URL | https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m |
| Автор модели | Meta AI Research (FAIR) |
| Публикация | Oquab et al., «DINOv2: Learning Robust Visual Features without Supervision», arXiv:2304.07193 |
| Обучающий набор | LVD-142M (курированный набор Meta, без разметки) |
| Лицензия | Apache 2.0 (DINOv2 переведён с CC BY-NC на Apache 2.0 в августе 2023) |
| Как получено | `timm.create_model(..., pretrained=True)` при первом обучении; в релизе `pretrained=False` |

Учитель дистилляции — та же линейка, `vit_large_patch14_dinov2.lvd142m`,
https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m, тот же автор и лицензия.
Учитель обучался на тех же 1233 fit-личностях (подтверждается `weights/hack_dinov2_l_336_cam/log.json`:
`val_frac: null`) и в поставку не входит — он нужен был только на обучении.

**Важно для проверки лимита 2 ГБ.** В релизном образе лежит ровно один файл весов,
`release/model.pt`, 164 МиБ. Он самодостаточен: архитектура собирается timm с `pretrained=False`,
из сети в рантайме ничего не тянется (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
Веса учителя и промежуточные чекпойнты в образ не попадают.

## 2. Библиотеки и образы

Все версии закреплены точно, ни одной ссылки на подвижный тег (ответ 39). Полный список
пакетов, фактически установленных в образ, снимается изнутри собранного образа:
`docker run --rm --entrypoint python vreid-release /app/scripts/image_report.py`.

### 2.1 Модель и инференс — `requirements-infer.txt`

| пакет | версия | автор | URL | лицензия |
|---|---|---|---|---|
| timm | 1.0.20 | Ross Wightman и участники | https://github.com/huggingface/pytorch-image-models | Apache 2.0 |
| torch, torchvision | 2.3.1 / 0.18.1 (из базового образа) | Meta AI / PyTorch | https://github.com/pytorch/pytorch | BSD-3-Clause |
| numpy | 1.26.4 | сообщество NumPy | https://github.com/numpy/numpy | BSD-3-Clause |
| pillow | 10.4.0 | Jeffrey A. Clark и участники | https://github.com/python-pillow/Pillow | MIT-CMU |
| huggingface-hub | 1.32.0 | Hugging Face | https://github.com/huggingface/huggingface_hub | Apache 2.0 |
| safetensors | 0.8.0 | Hugging Face | https://github.com/huggingface/safetensors | Apache 2.0 |

### 2.2 Сервис — `requirements-service.txt`

| пакет | версия | автор | URL | лицензия |
|---|---|---|---|---|
| fastapi | 0.115.12 | Sebastián Ramírez | https://github.com/fastapi/fastapi | MIT |
| starlette | 0.46.2 | Encode | https://github.com/encode/starlette | BSD-3-Clause |
| uvicorn | 0.34.2 | Encode | https://github.com/encode/uvicorn | BSD-3-Clause |
| pydantic | 2.13.5 | Samuel Colvin и участники | https://github.com/pydantic/pydantic | MIT |
| python-multipart | 0.0.20 | Andrew Dunham и участники | https://github.com/Kludex/python-multipart | Apache 2.0 |
| psycopg | 3.2.12 | Daniele Varrazzo и участники | https://github.com/psycopg/psycopg | LGPL-3.0 |
| faiss-cpu | 1.9.0.post1 | Meta AI Research | https://github.com/facebookresearch/faiss | MIT |

### 2.3 Клиентский слой

| компонент | версия | URL | лицензия |
|---|---|---|---|
| nginx (базовый образ) | 1.27.4-alpine | https://hub.docker.com/_/nginx | BSD-2-Clause |
| Swagger UI (файлы в `service/static/vendor/`) | 5.17.14 | https://github.com/swagger-api/swagger-ui | Apache 2.0 |

Swagger UI скачан на этапе подготовки и лежит в репозитории: страница спецификации обязана
открываться без интернета. Контрольные суммы файлов:

```
92b9842ad2dc4d85fd74bbd3d4ca85147777bba34751d0e6a45d52493c8d6219  swagger-ui-bundle.js
7eb73e8c93bb2add112824cb5a19a25dd65f489227b7b5b273a07cbee5824fb1  swagger-ui.css
```

### 2.4 Базовые образы

| образ | дайджест или тег | назначение |
|---|---|---|
| `pytorch/pytorch` | `sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a` (тег `2.3.1-cuda12.1-cudnn8-runtime`) | релизный: конкурсный прогон и сервис на стенде жюри |
| `pytorch/pytorch` | `2.11.0-cuda12.8-cudnn9-runtime` | только локальная сборка на карте sm_120, в сдачу не идёт |
| `pgvector/pgvector` | `pg16` | СУБД галереи |
| `nginx` | `1.27.4-alpine` | тонкий клиент |

Дайджест релизной базы зафиксирован `scripts/pin_base_image.py`: тег владелец репозитория
вправе перезалить, дайджест — нет.

## 3. Данные

Использованы только данные организаторов: `train.csv`, `test_query.csv`, `test_gallery.csv`
и `images/`. Внешние re-ID датасеты (VeRi-776, VehicleID, CityFlow, CARLA) **не использовались**
ни для обучения, ни для калибровки. В репозитории остались конфиги `configs/carla.yaml` и
`configs/veri776.yaml` и скрипт `scripts/prepare_carla.py` — это нерабочие черновики раннего
этапа, ни один вес из них не участвовал; проверяется по `log.json` каждого чекпойнта, где
записан фактический конфиг обучения.

## 4. Чего здесь нет

Юридического заключения. Приведены заявленные авторами лицензии и ссылки; проверка на
соответствие требованиям конкурса — за организаторами.
