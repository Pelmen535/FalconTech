# Queue 19: честная цифра каскада, ре-ранкер для релиза и ученик от трёх учителей.
#     powershell -ExecutionPolicy Bypass -File journal\run_queue19_cascade.ps1
# ASCII only in code. Около 8 часов, три обучения подряд.
#
# ЗАЧЕМ. Точность - единственный слабый блок (33.4 из 45). Измерено сегодня:
#
#   1. Каскад "ViT-B ищет - ViT-L проверяет топ-30" даёт 74.21 -> 78.51 mAP@10, режим
#      отказа не трогает (96.40 -> 96.55 при том же пороге 0.55), а замеряемая жюри
#      латентность не меняется: rerank в неё не входит (ответы 31/32), веса ре-ранкера
#      считаются в лимит 2 ГБ (ответы 34/37). НО нынешний ViT-L выбирал лучшую эпоху по
#      той же валидации, на которой снят прирост, - цифра завышена на неизвестную величину.
#      Шаг 1 делает двойника с --no-select и даёт честное число.
#   2. Релизный ре-ранкер должен быть обучен на ВСЕХ личностях, как soup_b336_all для B.
#      Шаг 4.
#   3. Дистилляция от трёх учителей вместо одного - чистый выигрыш без каскада, на
#      инференсе бесплатный. Код проверен на эквивалентность: с одним учителем поведение
#      совпадает до бита (reports/equivalence_before_multiteacher.json).
#
# Порядок не случаен: если шаг 3 покажет, что прирост каскада - артефакт отбора эпохи,
# шаг 4 не нужен, и его надо снимать с очереди.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue19 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }
$global:failed = @()
function Step($name, $cmd) {
    Log "START $name"; $t0 = Get-Date
    $global:LASTEXITCODE = 0
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log"
    $code = $LASTEXITCODE
    $log = "logs\$name.log"
    $empty = -not (Test-Path $log) -or ((Get-Item $log).Length -eq 0)
    $verdict = if ($code -ne 0) { "FAILED exit=$code" } elseif ($empty) { "FAILED no output" } else { "ok" }
    if ($verdict -ne "ok") { $global:failed += $name }
    Log "END   $name  $verdict  $([int]((Get-Date) - $t0).TotalMinutes) min"
}

Log "WAIT  for running vreid processes"
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 30
}
docker compose down | Out-Null
Log "WAIT  done - GPU free"

# Рецепт учителя взят из weights/hack_dinov2_l_336_cam/log.json без единого изменения,
# кроме режима сохранения: меняем ровно одно, иначе сравнивать будет не с чем.
$L = "--config configs/hackathon.yaml --backbone dinov2_l --img-size 336 --epochs 20 " +
     "--P 5 --K 4 --freeze-blocks 8 --cam-aware --cross-cam-triplet --workers 4"

# 1-3. Честная цифра каскада: двойник ре-ранкера, не видевший валидацию при выборе эпохи
Step "q19_train_l_fit" "python -m vreid.train $L --no-select --seed 0 --out weights/l336cam_fit"
Step "q19_val_l_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/l336cam_fit/best.pt --workers 4 --rerank"
Step "q19_cascade_honest" "python scripts/cascade_verify.py --light runs/hack/ft_soup_b336_fit --heavy runs/hack/ft_l336cam_fit --topk 30 --out results/cascade_verify_honest.json"

# 4. Ре-ранкер, который поедет в релиз: все личности, последняя эпоха
Step "q19_train_l_all" "python -m vreid.train $L --val-frac 0 --seed 0 --out weights/l336cam_all"

# 5-6. Ученик от трёх учителей. Рецепт fit18 без изменений, кроме списка учителей:
#      P=11 задан явно, а не auto, - иначе размер батча уплывёт и сравнивать будет не с чем.
$T = "weights/hack_dinov2_l_336_cam/best.pt,weights/hack_dinov2_l_336/best.pt,weights/hack_dinov2_l_336_ls/best.pt"
$B = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 336 --epochs 18 " +
     "--P 11 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --distill-w 20 " +
     "--no-select --workers 4"
Step "q19_train_b_multi" "python -m vreid.train $B --distill $T --seed 0 --out weights/fit18mt_s0"
Step "q19_val_b_multi" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18mt_s0/best.pt --workers 4 --rerank"

# 7. Сводка: что с чем сравнивать
Step "q19_compare" "python scripts/compare_training.py --baseline runs/hack/ft_fit18_s0 --candidate runs/hack/ft_fit18mt_s0 --out results/training_multiteacher.json"

if ($global:failed.Count -eq 0) {
    Log "queue19 finished: все шаги прошли"
} else {
    Log "queue19 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
