# Цифровой отпечаток автомобиля

Идентификация транспортного средства по визуальным признакам без опоры на
государственный регистрационный знак. Система строит устойчивый признак
автомобиля, связывает наблюдения с разных камер и возвращает ранжированный
список кандидатов с диагностикой качества результата.

Главный документ проекта — [PROJECT.md](PROJECT.md). Контракты кода находятся
в [contracts/schema.py](contracts/schema.py), данные описаны в
[docs/data.md](docs/data.md), решения — в [docs/decisions.md](docs/decisions.md),
а протокол метрик — в [docs/metrics.md](docs/metrics.md).

## Что уже запускается

В репозитории есть воспроизводимый smoke baseline. Он создаёт синтетический
набор, проверяет отсутствие утечки идентичностей и дубликатов, маскирует номер
до извлечения признаков, выполняет cross-camera поиск и считает Recall@1,
Recall@5 и mAP. Простые цветовые признаки нужны только для проверки конвейера;
их результат не является качеством конкурсной модели. Для обученной модели
предусмотрен локальный ONNX-адаптер.

Текущий smoke harness принимает отдельный CSV-манифест оценки. Канонический
обмен между модулями проекта остаётся JSONL и структурами из
`contracts/schema.py`; адаптер к этому контракту ещё предстоит добавить.

## Быстрый smoke test в PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m vehicle_reid make-demo --out data/demo
python -m vehicle_reid audit --manifest data/demo/manifest.csv
python -m vehicle_reid evaluate --manifest data/demo/manifest.csv --out outputs/demo_metrics.json
python -m pytest
```

Полный прогон можно выполнить одной командой:

```powershell
.\scripts\run_pipeline.ps1 -Manifest data\demo\manifest.csv -Output outputs\demo_metrics.json -Python .\.venv\Scripts\python.exe
```

Для ONNX-модели установите `python -m pip install -e ".[onnx]"` и передайте
`--backend onnx --model path/to/model.onnx --model-spec configs/model_spec.example.json`.
Веса, датасеты, видео, кропы и результаты прогонов в Git не добавляются.

## Среда для основной модели

PyTorch ставится отдельно под целевую видеокарту:

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_capability())"
```

Ожидаемая capability для целевой Blackwell-карты — `(12, 0)`.
