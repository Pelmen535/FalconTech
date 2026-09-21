# Queue 20: пересборка релиза с каскадом и полная пересъёмка артефактов.
#     powershell -ExecutionPolicy Bypass -File journal\run_queue20_release.ps1
# ASCII only in code. Около 40 минут, без обучения. Запускать ПОСЛЕ queue19.
#
# ЗАЧЕМ. Каскад меняет ранжирование, а значит - submission.csv, candidates.csv, все отчёты
# и оба образа. Пересъёмка обязана быть полной: половина артефактов от старого рецепта и
# половина от нового - худший из возможных вариантов, потому что выглядит согласованно.
#
# ЧТО НОВОГО В ПРОВЕРКАХ.
#   * check_cached_replay теперь требует reranker_embeddings.npy: без признаков ре-ранкера
#     сдачу из артефактов не пересобрать, и молча проверять половину пайплайна нельзя.
#   * bench_full меряет ДВЕ латентности - замеряемую жюри (только основная модель,
#     ответы 31/32) и настоящую, с форвардом ре-ранкера. Обе идут в отчёт.
#   * weight_inventory при сборке образа считает СУММУ весов против 2 000 000 000 байт.
#
# ПРЕДУСЛОВИЕ: weights/l336cam_all/best.pt существует (шаг q19_train_l_all).

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue20 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

if (-not (Test-Path "weights\l336cam_all\best.pt")) {
    Log "queue20 ABORT: нет weights\l336cam_all\best.pt - сначала queue19"
    Write-Host "Нет весов ре-ранкера. Сначала journal\run_queue19_cascade.ps1"
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

Step "q20_tests" "python -m pytest tests -q"

# 1. Релиз: основная модель прежняя, добавляется ре-ранкер. Порог и параметры
#    ре-ранжирования переносятся как есть - их выбирали отдельно и не пересматривали.
Step "q20_export" ("python scripts/export_release.py --weights weights/soup_b336_all/best.pt " +
    "--reranker weights/l336cam_all/best.pt --val results/hack_ft_soup_b336_fit_val.json " +
    "--confidence top1 --threshold 0.55 --kr --k1 3 --k2 2 --lam 0.2 --crop-pad 0.05 " +
    "--cascade-topk 30 --cascade-alpha 0.9 --out release")

# 2. Сдача боевым прогоном
Step "q20_predict" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "q20_form" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "q20_indep" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "q20_ties" "python scripts/check_tie_independence.py"
Step "q20_replay" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

# 3. Образы на текущем коде и текущем релизе
Step "q20_build_release" "docker build -t vreid-release ."
Step "q20_build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "q20_build_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "q20_build_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."

# 4. Контейнер: дважды, офлайн, сравнение бит в бит
Step "q20_docker_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q20_a:/out vreid-dev"
Step "q20_docker_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q20_b:/out vreid-dev"
Step "q20_cmp_det" "python scripts/compare_outputs.py out_q20_a out_q20_b"
Step "q20_docker_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q20_offline:/out vreid-dev"
Step "q20_cmp_offline" "python scripts/compare_outputs.py out_q20_a out_q20_offline"
Step "q20_form_docker" "python scripts/check_submission.py --submission out_q20_a --data %VREID_DATA%"

# 5. Производительность: обе цифры - замеряемая и настоящая
Step "q20_bench" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# 6. Отчёты
Step "q20_official" "python scripts/score_validation_with_official.py"
Step "q20_stress" "python scripts/refusal_stress.py --out results/refusal_stress_fit.json"
Step "q20_audit" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe release/recipe.json --out results/refusal_audit_fit.json"
Step "q20_errors" "python scripts/error_analysis.py"
Step "q20_scorecard" "python scripts/scorecard.py --release release --submission submission"
Step "q20_manifest" "python scripts/make_manifest.py --release release --submission submission --run runs/hack/ft_soup_b336_fit --out reports/MANIFEST.json"

# 7. Прототип обратно в рабочее состояние
Step "q20_compose" "docker compose up -d"
Step "q20_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q20_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"
Step "q20_logs" "python scripts/normalize_logs.py --apply"

if ($global:failed.Count -eq 0) {
    Log "queue20 finished: все шаги прошли"
} else {
    Log "queue20 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
