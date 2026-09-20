# Queue 15: финальный прогон на замороженном рецепте. Всё, что сдаётся, снимается заново.
#     powershell -ExecutionPolicy Bypass -File run_queue15.ps1
# ASCII only in code, about 25 minutes, no training.
#
# Why this queue exists. Queue 14 rebuilt the images and proved the merge, but после неё
# recipe.json ещё раз менялся: в него записано обоснование порога (threshold_choice) и
# точные числа стресс-проверки. Рецепт входит в образ и в run_info.json, поэтому все
# артефакты, снятые до этой правки, относятся к другому файлу. Пересобираем и переснимаем,
# чтобы каждый отпечаток в сдаче указывал на то, что реально лежит в release/.
#
# Кроме того, здесь впервые проверяется весь продукт целиком: конкурсный образ, образ
# сервиса, веб-слой, стек compose и сквозной HTTP-прогон на настоящей модели.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
# Compose on this machine cannot use bake: the project path is not ASCII and buildkit
# rejects the session header it derives from it. Per-service build works.
$env:COMPOSE_DOCKER_CLI_BUILD = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue15 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

Step "tests15" "python -m pytest tests -q"

# 1. Все четыре образа на текущем коде и текущем рецепте
Step "build15_release" "docker build -t vreid-release ."
Step "build15_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "build15_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "build15_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
Step "packages15" "docker run --rm --entrypoint python vreid-release /app/scripts/image_report.py"

# 2. Сдача и её форма
Step "predict15" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "form15" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "indep15" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "ties15" "python scripts/check_tie_independence.py"
Step "replay15" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

# 3. Контейнер: дважды, офлайн, без разделяемой памяти
Step "docker15_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_final_a:/out vreid-dev"
Step "docker15_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_final_b:/out vreid-dev"
Step "cmp15_det" "python scripts/compare_outputs.py out_final_a out_final_b"
Step "docker15_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_final_offline:/out vreid-dev"
Step "cmp15_offline" "python scripts/compare_outputs.py out_final_a out_final_offline"
Step "docker15_noshm" "docker run --rm --gpus all -v %VREID_DATA%:/data:ro -v %cd%\out_final_noshm:/out vreid-dev"
Step "cmp15_noshm" "python scripts/compare_outputs.py out_final_a out_final_noshm"
Step "cmp15_native" "python scripts/compare_outputs.py submission out_final_a"
Step "form15_docker" "python scripts/check_submission.py --submission out_final_a --data %VREID_DATA%"

# 4. Производительность по методике организаторов, на образе, который существует сейчас
Step "bench15" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# 5. Отказ, разбор ошибок, карточка и манифест
Step "stress15" "python scripts/refusal_stress.py --run runs/hack/ft_soup_b336_fit --release release --out results/refusal_stress_fit.json"
Step "audit15" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe release/recipe.json --out results/refusal_audit_fit.json"
Step "errors15" "python scripts/error_analysis.py --run runs/hack/ft_soup_b336_fit --release release --data %VREID_DATA%"
Step "scorecard15" "python scripts/scorecard.py --release release --submission submission"
Step "manifest15" "python scripts/make_manifest.py --release release --submission submission --run runs/hack/ft_soup_b336_fit --out reports/MANIFEST.json"

# 6. Продукт целиком: стек поднимается и отвечает настоящей моделью
Step "compose15" "docker compose up -d"
Step "waitapi15" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "smoke15" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"
Step "logs15" "python scripts/normalize_logs.py --apply"

if ($global:failed.Count -eq 0) {
    Log "queue15 finished: все шаги прошли"
} else {
    Log "queue15 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
