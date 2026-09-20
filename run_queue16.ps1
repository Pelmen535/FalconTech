# Queue 16: релиз с новыми параметрами ре-ранжирования. Пересобрать и переснять всё.
#     powershell -ExecutionPolicy Bypass -File run_queue16.ps1
# ASCII only in code, about 25 minutes, no training.
#
# Why this queue exists. Прежние параметры k-reciprocal 6/2/0.3 были ПЕРВОЙ строкой грубой
# сетки 6/10/15/20 - то есть краем диапазона, а не найденным оптимумом. Полная сетка из 93
# точек (scripts/rerank_grid.py) показала, что оптимум внутри: k1=3, k2=2, lambda=0.2.
# Точка выбрана на кеше двойника релиза и независимо подтверждена на кеше валидационной
# модели: лучшая точка совпала, ранговая корреляция поверхностей 0.957.
#
#   mAP@10   двойник 73.77 -> 74.33,  вторая модель 74.88 -> 75.97
#   rank-1   двойник 69.59 -> 70.72,  вторая модель 71.65 -> 72.68
#   отказ    не меняется вовсе: ре-ранжирование не сдвигает топ-1 ни у одного из 1213
#            запросов, значит уверенность и набор принятых запросов те же
#   цена     нулевая: поиск и ре-ранжирование не входят в замер латентности (ответ 31)
#
# lambda раньше была вшита числом 0.3 в трёх местах (predict, eval_common, адаптер сервиса).
# Теперь она в рецепте, как и остальные параметры постобработки, и проверяется на входе.
#
# Оговорка, которую надо называть вслух: оба кеша имеют галерею 688, а закрытый тест 750.
# Перенос оптимума k1 на другой размер галереи мы не измеряли.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
# Compose on this machine cannot use bake: the project path is not ASCII and buildkit
# rejects the session header it derives from it. Per-service build works.
$env:COMPOSE_DOCKER_CLI_BUILD = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue16 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }
# Windows PowerShell 5.1: у Tee-Object НЕТ параметра -Encoding, он появился только в PS 7.
# Пишем как есть, а в конце очереди прогоняем scripts/normalize_logs.py.
#
# Шаг считается провалившимся не только по ненулевому коду возврата. Первая версия этой
# очереди молча "прошла" целиком за три секунды: Tee-Object падал на разборе параметров,
# ни одна команда не запускалась, а $LASTEXITCODE оставался нулём с прошлого шага. Ровно
# так же в очередях 12 и 13 "зелёными" оказывались docker-шаги на старом образе. Поэтому
# проверяется ещё и то, что шаг вообще что-то написал.
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
$global:failed = @()

Log "WAIT  for running vreid processes"
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)|run_service" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 30
}
docker compose down | Out-Null
Log "WAIT  done - GPU free, service stack down"

$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"

Step "tests16" "python -m pytest tests -q"

# 1. Все четыре образа на текущем коде и текущем рецепте
Step "build16_release" "docker build -t vreid-release ."
Step "build16_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "build16_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "build16_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
Step "packages16" "docker run --rm --entrypoint python vreid-release /app/scripts/image_report.py"

# 2. Сдача и её форма
Step "predict16" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "form16" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "indep16" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "ties16" "python scripts/check_tie_independence.py"
Step "replay16" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

# 3. Контейнер: дважды, офлайн, без разделяемой памяти
Step "docker16_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_r16_a:/out vreid-dev"
Step "docker16_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_r16_b:/out vreid-dev"
Step "cmp16_det" "python scripts/compare_outputs.py out_r16_a out_r16_b"
Step "docker16_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_r16_offline:/out vreid-dev"
Step "cmp16_offline" "python scripts/compare_outputs.py out_r16_a out_r16_offline"
Step "docker16_noshm" "docker run --rm --gpus all -v %VREID_DATA%:/data:ro -v %cd%\out_r16_noshm:/out vreid-dev"
Step "cmp16_noshm" "python scripts/compare_outputs.py out_r16_a out_r16_noshm"
Step "cmp16_native" "python scripts/compare_outputs.py submission out_r16_a"
Step "form16_docker" "python scripts/check_submission.py --submission out_r16_a --data %VREID_DATA%"

# 4. Производительность по методике организаторов, на образе, который существует сейчас
Step "bench16" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# 5. Отказ, разбор ошибок, карточка и манифест
Step "stress16" "python scripts/refusal_stress.py --run runs/hack/ft_soup_b336_fit --release release --out results/refusal_stress_fit.json"
Step "audit16" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe release/recipe.json --out results/refusal_audit_fit.json"
Step "errors16" "python scripts/error_analysis.py --run runs/hack/ft_soup_b336_fit --release release --data %VREID_DATA%"
Step "confidence16" "python scripts/confidence_stress.py --run runs/hack/ft_soup_b336_fit --out results/confidence_stress.json"
Step "scorecard16" "python scripts/scorecard.py --release release --submission submission"
Step "manifest16" "python scripts/make_manifest.py --release release --submission submission --run runs/hack/ft_soup_b336_fit --out reports/MANIFEST.json"

# 6. Продукт целиком: стек поднимается и отвечает настоящей моделью
Step "compose16" "docker compose up -d"
Step "waitapi16" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "smoke16" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"
Step "logs16" "python scripts/normalize_logs.py --apply"

if ($global:failed.Count -eq 0) {
    Log "queue16 finished: все шаги прошли"
} else {
    Log "queue16 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
