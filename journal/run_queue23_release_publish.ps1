# Queue 23: релиз на входе 392, полная пересъёмка артефактов и публикация на GitHub.
#     powershell -ExecutionPolicy Bypass -File journal\run_queue23_release_publish.ps1
# ASCII only in code. Около 45 минут, без обучения. Запускать ПОСЛЕ queue22.
#
# ЗАЧЕМ. Разрешение входа 392 - единственный крупный рычаг по точности, который ничего не
# стоит по баллу: шкала производительности опубликована (ответ 34), и запас у нас был.
# Измерено на настоящем конвейере, а не спроецировано:
#     336: 28.5 мс, 195.7 FPS   запас по пропускной 96%
#     392: 31.3 мс, 148.7 FPS   запас 49%
#     448: 33.8 мс, 111.3 FPS   запас 11%  <- слишком близко к обрыву, отвергнуто
#
# Двойник релиза на 392 (суп трёх сидов) уже провалидирован: mAP@10 с k-reciprocal 3/2/0.2
# даёт 77.60 против 74.21 у двойника на 336, rank-1 74.85 против 70.72. Прирост +3.39 п.п.
#
# БЕЗ КАСКАДА. Каскад с ре-ранкером ViT-L даёт больше, но его балл держится на трактовке
# правил: rerank не входит в замер латентности (ответы 31/32). Если жюри решит иначе,
# производительность падает с 20 до 10.9 из 20. Разрешение даёт меньше, зато при нулевом
# риске - и это решение владельца проекта, а не моё. Веса ре-ранкера и код каскада остаются
# в репозитории: включается одним полем рецепта (cascade: true).
#
# ЧТО ПЕРЕСЧИТЫВАЕТСЯ. Порог отказа берётся не старый: у модели с другим входом косинусы
# распределены иначе, оптимум валидации сместился с 0.6026 на 0.5840. Замороженный порог
# остаётся ниже оптимума - та же страховка, что и раньше, - но сам выбор делается по новому
# стресс-отчёту (q22_stress), а не переносится по инерции.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue23 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

if (-not (Test-Path "weights\soup_r392_all\best.pt")) {
    Log "queue23 ABORT: нет weights\soup_r392_all\best.pt - сначала queue22"
    Write-Host "Нет боевой модели на 392. Сначала journal\run_queue22_release392.ps1"
    exit 1
}

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

Step "q23_tests" "python -m pytest tests -q"

# 1. Релиз: модель на 392, без ре-ранкера. Порог 0.55 остаётся ниже оптимума валидации
#    (0.5840), то есть страховка та же; проверяется шагом q23_audit ниже.
Step "q23_export" ("python scripts/export_release.py --weights weights/soup_r392_all/best.pt " +
    "--val results/hack_ft_soup_r392_fit_val.json --confidence top1 --threshold 0.55 " +
    "--kr --k1 3 --k2 2 --lam 0.2 --crop-pad 0.05 --out release")

# 2. Сдача боевым прогоном и все проверки формы и независимости
Step "q23_predict" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "q23_form" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "q23_indep" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "q23_ties" "python scripts/check_tie_independence.py"
Step "q23_replay" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

# 3. Образы и детерминизм
Step "q23_build_release" "docker build -t vreid-release ."
Step "q23_build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "q23_build_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "q23_build_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
Step "q23_docker_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q23_a:/out vreid-dev"
Step "q23_docker_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q23_b:/out vreid-dev"
Step "q23_cmp_det" "python scripts/compare_outputs.py out_q23_a out_q23_b"
Step "q23_docker_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q23_offline:/out vreid-dev"
Step "q23_cmp_offline" "python scripts/compare_outputs.py out_q23_a out_q23_offline"

# 4. Производительность НА НОВОМ ВХОДЕ - главное число этой очереди
Step "q23_bench" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# 5. Отчёты: порог проверяется на переносимость, метрика - эталонным scorer
Step "q23_official" "python scripts/score_validation_with_official.py"
Step "q23_audit" "python scripts/refusal_audit.py --run runs/hack/ft_soup_r392_fit --recipe release/recipe.json --out results/refusal_audit_r392.json"
Step "q23_errors" "python scripts/error_analysis.py"
Step "q23_scorecard" "python scripts/scorecard.py --release release --submission submission"
Step "q23_manifest" "python scripts/make_manifest.py --release release --submission submission --run runs/hack/ft_soup_r392_fit --out reports/MANIFEST.json"

# 6. Прототип обратно
Step "q23_compose" "docker compose up -d"
Step "q23_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q23_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"
Step "q23_logs" "python scripts/normalize_logs.py --apply"

if ($global:failed.Count -eq 0) {
    Log "queue23 finished: все шаги прошли - можно публиковать v1.1"
} else {
    Log "queue23 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
