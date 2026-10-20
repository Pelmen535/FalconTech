# Внешние источники и происхождение v2.0

Этот список относится к October A / `vric-road-verifier-v1/current-query-v1`.
Историю DINOv2 v1.4 можно посмотреть в [теге v1.4](https://github.com/Pelmen535/FalconTech/tree/v1.4).

## Модели и данные

| Источник и версия | Официальная страница / получение | Как использован |
|---|---|---|
| DINOv3 ViT-L/16 LVD-1689M — Meta; model family 2025; runtime timm 1.0.29 | [Источник](https://github.com/facebookresearch/dinov3) · [Получение](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | pretrained foundation → own full fine-tuning → interpolation25 → immutable CPU/TRT artifacts |
| FastReID R50-IBN/GeM — JDAI-CV; v0.1.1; commit c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d | [Источник](https://github.com/JDAI-CV/fast-reid/tree/c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d) · [Получение](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth) | pretrained CNN, own final-layer/residual fine-tuning; upstream SHA f591e4f7f31ce0be2f4079171cfc03fcbecab5ea5b40f9148c753b9c6f061e8e |
| VERI-Wild — Lou et al.; original 2019 dataset, author TRAIN split | [Источник](https://github.com/PKU-IMRE/VERI-Wild) · [Получение](https://github.com/PKU-IMRE/VERI-Wild) | foundation ReID training, source transformations/heads, historical CAL/VAL; CNN author pretraining |
| NiVe1303 — Ruozheng LI; Mendeley v1; DOI 10.17632/42wv2svztx.1 | [Источник](https://data.mendeley.com/datasets/42wv2svztx/1) · [Получение](https://doi.org/10.17632/42wv2svztx.1) | night backbone fine-tuning, transforms/routing/confidence; 500 FIT IDs, 12319 FIT images |
| DN-348 — Li et al.; CVPR2024 DN-ReID dataset | [Источник](https://github.com/chenjingong/DN-ReID) · [Получение](https://drive.google.com/uc?export=download&id=1I7khNI6upkvZqJmhygQ2CQUlHzLGFEFO) | CNN fine-tuning: 1800 images; additional development/CAL/VAL |
| CFV / Car Full View — Electronics 2023 12(24),4947; public train/test archives | [Источник](https://www.mdpi.com/2079-9292/12/24/4947) · [Получение](https://huggingface.co/datasets/fort-cyber/CFV-Dataset) | historical transfer diagnostics; reused as downstream head development/training; NOT a fresh final holdout |
| VRAI — Wang et al.; ICCV2019; author train/dev | [Источник](https://github.com/JiaoBL1234/VRAI-Dataset) · [Получение](https://drive.google.com/drive/folders/1S8BCBDHxXJeqyv8v6LFWS5EZHvVdrHeN) | expanded head FIT/CAL and historical downstream development |
| RGBNT100 — 100-vehicle multispectral dataset; RGB only | [Источник](https://github.com/lsh-ahu/ICPL-ReID) · [Получение](https://github.com/lsh-ahu/ICPL-ReID#datasets) | expanded head and downstream development; aliases with WMVeID excluded where known |
| Vehicle-Rear — de Oliveira et al.; 2019 author dataset | [Источник](https://github.com/icarofua/vehicle-rear) · [Получение](https://github.com/icarofua/vehicle-rear) | historical transfer and later exposed downstream-head development |
| WMVeID863 — author public RGB/NIR/TIR dataset; visible modality only | [Источник](https://github.com/lsh-ahu/ICPL-ReID) · [Получение](https://drive.google.com/uc?export=download&id=186Ep0YgHY0a8BQ1Z59HP1D2-kZazbH02) | 398 filtered author TRAIN IDs for road refusal; known RGBNT aliases/train-test aliases excluded |
| VRIC — Kanaci, Zhu, Gong; GCPR2018; author TRAIN/TEST | [Источник](https://qmul-vric.github.io/index.html) · [Получение](https://qmul-vric.github.io/index.html) | 600 author TRAIN IDs / 7826 raw/blackout FIT images for road head; 661-query diagnostic separate |

Организаторский набор предоставлен участникам через личный кабинет; собственного публичного URL на его изображения нет. Данные, разметка, ТЗ, FAQ и эталонный scorer организаторов в открытый репозиторий не включены. Численные результаты и инструкции воспроизведения доступны без распространения чужих материалов.

**История использования данных.** Две эпохи последующего дообучения DINOv3 использовали 8 000 VERI-Wild и 12 319 NiVe FIT-снимков; CNN — дополнительно 1 800 DN-348. До этого DINOv3 проходил предобучение на VERI-Wild и обучение на организаторском TRAIN. Последняя голова отказа обучалась на 63 445 строках эпизодов, 9 415 консервативных группах ID: организаторские данные, VERI-Wild, NiVe, DN-348, CFV, VRAI, RGBNT100, Vehicle-Rear, WMVeID и VRIC. Это количество строк, а не независимых машин. Для WMVeID добавлены 398 отфильтрованных author-TRAIN ID; для VRIC — 600 author-TRAIN ID. Известные точные дубликаты и некоторые межнаборные aliases исключались; отсутствие всех физических/перекодированных совпадений и пересечений с foundation-pretraining не доказано.

Исторические CAL/VAL/transfer-наборы неоднократно использовались для разработки, а часть позднее вошла в обучение головы отказа. Поэтому опубликованные сравнения — **локальные результаты разработки**, не независимый финальный holdout и не оценка закрытого теста организаторов. Нового обучения при подготовке v2.0 не было.

## Версии и точные файлы весов

- Архитектура DINOv3: `vit_large_patch16_dinov3.lvd1689m`, 384×384, 2 048D CLS + mean patch; финальный исходный checkpoint `interpolation25.pt`, SHA-256 `9e9940418ed3b8b945d61befd26fff2fd571305467e29a2fc1d44b5575c37748`.
- FP32 CPU-представление содержит те же 323 тензора после обратимого byte-shuffle и сжатия, без квантования: SHA-256 `76c07501e972cc0bc19e2115fd02dd2438e64e40ff143818de32b670ec869335`.
- CNN: ResNet50-IBN/GeM + residual head, собственный `specialist-cnn.pt`, SHA-256 `ef3cf8824ee6ccafa7e2ec886957dd4951073df32500f0d5e4d2001e19a96907`.
- Исходная CNN: FastReID `v0.1.1/veriwild_bot_R50-ibn.pth`, SHA-256 `f591e4f7f31ce0be2f4079171cfc03fcbecab5ea5b40f9148c753b9c6f061e8e`.
- Все **16 используемых файлов**, их URL, размеры и SHA-256: [release/weights.json](release/weights.json). Сумма **1 838 245 659 байт** при лимите 2 000 000 000 байт. В комплект включены оба TensorRT engine, оба CPU-экстрактора и все transforms/классификаторы; скрытых сетевых загрузок при инференсе нет.
- Отпечатки исполняемых исходников, model/recipe/embedding identity, выбранные параметры и TensorRT/SM: [release_current/runtime-manifest.json](release_current/runtime-manifest.json). [Релиз и файлы весов](https://github.com/Pelmen535/FalconTech/releases/tag/v2.0).

При восстановлении инференса исходные pretrained-файлы из внешних репозиториев не загружаются: используются опубликованные финальные файлы с SHA-256. Точный upstream revision исходных Meta/timm pretrained-весов в старых материалах не установлен; он не подменяется сегодняшней версией. Это ограничение истории обучения, а не ссылка на плавающий `latest` для запуска сервиса.

## Зависимости

Среда Linux задаётся неизменяемым [PyTorch base image](https://hub.docker.com/r/pytorch/pytorch) `pytorch/pytorch@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be` и точными [основными зависимостями](requirements-robust-linux.txt), [lock-файлом](requirements-robust-linux.lock.txt). FAISS ставится в отдельное окружение поверх базового: `faiss-cpu==1.15.0`.

| Компонент | Зафиксированная версия | Источник |
|---|---|---|
| PyTorch | 2.11.0+cu128 | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| timm | 1.0.29 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| NumPy | 2.5.3 | [numpy/numpy](https://github.com/numpy/numpy) |
| Pillow | 11.3.0 | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| scikit-learn / joblib / SciPy | 1.9.1 / 1.5.3 / 1.18.1 | [scikit-learn](https://github.com/scikit-learn/scikit-learn), [joblib](https://github.com/joblib/joblib), [SciPy](https://github.com/scipy/scipy) |
| TensorRT CUDA12 | 10.9.0.34 | [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) |
| FAISS CPU | 1.15.0 | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| FastAPI / Starlette / Uvicorn | 0.115.12 / 0.46.2 / 0.34.2 | [FastAPI](https://github.com/fastapi/fastapi), [Starlette](https://github.com/encode/starlette), [Uvicorn](https://github.com/encode/uvicorn) |
| psycopg / psycopg-binary | 3.2.12 / 3.2.12 | [psycopg](https://github.com/psycopg/psycopg) |
| PostgreSQL + pgvector | образ SHA ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b | [pgvector/pgvector](https://github.com/pgvector/pgvector) |
| nginx | образ SHA 4ff102c5d78d254a6f0da062b3cf39eaf07f01eec0927fd21e219d0af8bc0591 | [nginx](https://nginx.org/) |

Остальные транзитивные пакеты фиксируются базовым образом и lock-файлом; Dockerfile выполняет `pip check`, runtime проверяет ключевые версии до запуска. Установленные версии и фактическая проверочная среда сохраняются в отчёте A5000.

## Условия внешних материалов

Собственный код сохраняет [Apache-2.0](LICENSE); это не меняет условия внешних весов и данных. Для DINOv3 действует [отдельная DINOv3 License](third_party/DINOv3_LICENSE.md), её полный текст включён, использование Meta DINOv3 явно указано. Для кода FastReID включена [исходная Apache-2.0 license](third_party/FastReID_LICENSE). У VERI-Wild и VRAI есть исследовательские/некоммерческие условия доступа; VRIC сохраняет права владельцев исходных UA-DETRAC-изображений. Условия репозитория с кодом не означают автоматически ту же лицензию на датасет. Исходные сторонние изображения не распространяются. Полная юридическая совместимость коммерческого применения и передачи производных весов здесь не сертифицируется; см. условия авторов по ссылкам выше.

[Машиночитаемый реестр](third_party/model-dataset-registry.json) перечисляет роль каждого источника и известные условия. AICITY2021 Track2 DMT рассматривался как внешняя идея; его веса в выбранном runtime отсутствуют.
