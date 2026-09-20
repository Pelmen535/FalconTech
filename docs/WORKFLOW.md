# Команды проверки для Николая

Все команды ниже выполняются из корня solution в существующей среде обучения/инференса. Это инструкции для следующего запуска, не утверждение о полученных метриках.

## Замороженный отказ

Использовать кеши **validation-модели**, полученные с нужным decoder; имя файла само по себе этого не доказывает:

```text
python scripts/refusal_audit.py --query-npz runs/hack/ft_soup_b336_val/val_query_track_fast.npz --gallery-npz runs/hack/ft_soup_b336_val/val_gallery_track_fast.npz --recipe release/recipe.json --seed 0 --out results/refusal_audit_fixed.json
```

NPZ содержит emb, vids, cams, keys. Если пути другие — заменить их на реальные. Приложить extraction log, SHA validation checkpoint, recipe и split. Результат записывает frozen .55 и selected_on_A отдельно, с фактическими candidate CSV. Старые просмотры validation учитываются при описании независимости B.

## Контроль размера галереи

Сначала выбрать query cohort по ID, сохранить в JSON-массив строк. Все gallery того же vehicle_id сохраняются при каждом размере. Добавлять только distractors. Размеры выбрать после подсчёта доступных anchors/distractors; пример 500/600/688 допустим только если anchors ≤500 и полный набор содержит 688 строк.

```text
python scripts/gallery_size_effect.py --query-npz runs/hack/ft_soup_b336_val/val_query_track_fast.npz --gallery-npz runs/hack/ft_soup_b336_val/val_gallery_track_fast.npz --recipe release/recipe.json --query-ids-json results/fixed_cohort.json --sizes 500 600 688 --repeats 5 --out results/gallery_size_effect_fixed.json
```

При всех query может не остаться distractors: правильный результат — not_estimable. Не добавлять обучающие ID или дубликаты, не выбирать cohort по удачному падению метрики.

## Парная абляция

Сначала получить baseline и варианты на одних query. Варианты: зона номера, крупная сплошная заливка, заранее выбранные контроли равной площади. Хранить параметры и изображения масок. Для каждого query записать AP@10 в долях, применив правильный junk-фильтр. JSON:

```json
{
  "model_sha256": "<sha>", "recipe_sha256": "<sha>",
  "split_sha256": "<sha>", "mask_design_sha256": "<sha>",
  "predeclared_controls": true,
  "query_ids": ["q1", "q2"], "vehicle_ids": [1, 2],
  "baseline_ap10": [0.8, 0.7],
  "variants": {
    "plate_large": {"ap10": [0.6, 0.6], "area_fraction": [0.08, 0.09]},
    "control_upper": {"ap10": [0.7, 0.6], "area_fraction": [0.08, 0.09]}
  },
  "control_groups": [{"mask": "plate_large", "controls": ["control_upper"]}]
}
```

Числа выше вымышлены исключительно для описания формата; не использовать их в отчёте. В реальном JSON нужны все query и контроли.

```text
python scripts/ablation_report.py --paired-json results/paired_input.json --out results/ablation_paired.json
```

Bootstrap группирует по identity. Итог показывает изменения и неопределённость, не доказывает отсутствие зависимости от номера.

## Benchmark

В целевом контейнере:

```text
python -m vreid.bench_full --data /data --release /app/release --out /out/bench_full.json --device cuda:0 --n 300 --warmup 50 --min-seconds 10 --batches 1 8 16 32 --workers 4
```

Включены чтение, decode, crop, preprocessing, сеть и нормализация; поиск/rerank не включены. Сохранить сырой JSON, stdout, драйвер, image digest и installed-packages.txt. Проверяются также CPU/диск/методика стенда; один флаг target_gpu=true не доказывает совпадение всего оборудования.

## Официальный scorer

После получения от организатора записать URL и SHA-256. Обёртка не придумывает CLI: всё после -- передаётся evaluate.py без изменений.

```text
python scripts/run_official_scorer.py --script organizer/evaluate.py --source-url "<официальный URL>" --expected-sha256 "<SHA256>" --out results/official -- <документированные аргументы организатора>
```

Успешный exit code сам по себе не означает высокий балл. Хранить исходные stdout/stderr и актуальный example_submission.zip.

