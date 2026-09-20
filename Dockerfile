# Офлайн-инференс ML-ядра (ТЗ §7: запуск одной командой, без доступа в сеть).
#
# СТЕНД ЖЮРИ (ответ 30/32): RTX A5000 24 ГБ, CUDA driver 12.2, 2x Xeon Gold 6338.
# A5000 — Ampere (sm_86), и драйвер под CUDA 12.2 (ветка 535) НЕ запустит контейнер, собранный
# под CUDA 12.8: версия CUDA в образе не может быть новее той, что поддерживает драйвер хоста.
# Поэтому релизная база — CUDA 12.1, она работает на драйвере 530+ и покрывает Ampere.
# Успешный прогон на CPU совместимость с CUDA не доказывает: нужен прогон на самой A5000.
#
# На машине разработки карта RTX 5060 Ti — Blackwell, sm_120, которого в CUDA 12.1 нет.
# Для локального прогона база переключается одним аргументом:
#     docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev .
#
# Тег, из которого получен дайджест ниже: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# Дайджест зафиксирован scripts/pin_base_image.py — тег владелец репозитория вправе перезалить,
# дайджест нет (ответ 39: точные версии, никаких подвижных ссылок).
ARG BASE_IMAGE=pytorch/pytorch@sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a
FROM ${BASE_IMAGE}

# VREID_COMPILE=1 включает torch.compile. Замерено 17.09: на нашей связке он ЗАМЕДЛЯЕТ
# (76.2 -> 64.3 FPS на Windows без triton; в контейнере прироста тоже нет), поэтому выключен.
# VREID_THROTTLE читается в боевом пути — прибиваем к нулю, чтобы стенд не зависел от окружения.
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=8 \
    VREID_COMPILE=0 \
    VREID_THROTTLE=0

WORKDIR /app
# Зависимости ставятся на этапе сборки; в рантайме сеть не нужна (проверено прогоном
# с --network none). --no-deps + полный список транзитивных пакетов: организаторы требуют
# точных версий (ответ 39), а pip без --no-deps вправе подтянуть что-то новее закреплённого.
# pip check тут же ловит несогласованный набор — ломаться должно здесь, а не на стенде жюри.
#
# PIP_FLAGS нужен только для локальной сборки на dev-базе: образы pytorch 2.9+ помечены
# PEP 668 (externally-managed-environment), и pip отказывается ставить в системный python
# без --break-system-packages. Релизная база (CUDA 12.1) такой пометки не имеет, поэтому
# по умолчанию флаг пуст и релизная команда сборки остаётся ровно такой, как в README.
ARG PIP_FLAGS=""
# REQUIREMENTS переключается только для dev-базы: см. шапку requirements-infer-dev.txt.
ARG REQUIREMENTS=requirements-infer.txt
COPY requirements-infer.txt requirements-infer-dev.txt ./
RUN python -m pip install --no-cache-dir --no-deps ${PIP_FLAGS} -r ${REQUIREMENTS} \
 && python -m pip check \
 && python -m pip freeze --all > /app/installed-packages.txt

COPY vreid/ /app/vreid/
COPY scripts/ /app/scripts/
COPY release/ /app/release/

# Проверки на этапе сборки: импорт связки, инвентаризация весов против лимита 2 ГБ
# (ответы 34/37 — считается СУММА всех файлов весов) и разбор рецепта с его запретами.
RUN python -c "import torch, torchvision, timm; from vreid.artifacts import weight_inventory; from vreid.predict import load_recipe; print('[build] torch', torch.__version__, '| torchvision', torchvision.__version__, '| timm', timm.__version__); print('[build] weights:', weight_inventory('/app/release')); print('[build] recipe ok:', load_recipe('/app/release')['threshold'])"

# одна команда: читает /data, пишет /out
ENTRYPOINT ["python", "-m", "vreid.predict", "--data", "/data", "--release", "/app/release", "--out", "/out"]
