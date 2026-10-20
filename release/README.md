# Артефакты v2.0

- [runtime-manifest.json](runtime-manifest.json) задаёт модель, конфигурацию и контрольные суммы. Его копия используется в [release_current](../release_current/runtime-manifest.json).
- [weights.json](weights.json) содержит URL, размеры и SHA-256 всех 16 файлов весов. Загрузка: `python scripts/fetch_weights.py` из корня проекта.
- `recipe.json` и `model.pt` относятся к совместимости с v1.4. Точки входа v2.0 используют runtime manifest.

[Запуск сервиса и CLI](../README.md) · [Происхождение весов](../docs/TRAINING_PROVENANCE_V2_RU.md) · [Источники и лицензии](../SOURCES.md).
