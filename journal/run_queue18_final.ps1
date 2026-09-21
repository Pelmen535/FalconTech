# Queue 18: пересъёмка всех артефактов после сверки с эталонным scorer.
#     powershell -ExecutionPolicy Bypass -File journal\run_queue18_final.ps1
# ASCII only in code, около 15 минут, без обучения.
#
# ЗАЧЕМ. За день изменились три вещи, и каждая делает прежние артефакты неактуальными:
#
#   1. release/recipe.json. В нём теперь параметры ре-ранжирования 3/2/0.2, обоснование
#      порога и провенанс сверки с эталоном. Хэш рецепта входит и в submission/run_info.json,
#      и в results/bench_full.json - оба стали указывать на файл, которого больше нет.
#   2. vreid/. Вычищены train.py и hack_cli.py (эквивалентность доказана харнессом),
#      исправлены метрика mAP@10 и mINP, submission.csv переведён на официальный формат
#      без строки заголовка. Образы собраны из старого кода.
#   3. Метрика. mAP@10 теперь считается ровно как в organizer/evaluate.py: обрезание до
#      десяти ДО junk-фильтра. Все отчётные числа пересняты.
#
# Отдельно: submission.csv в submission/ был пересобран из сохранённых эмбеддингов
# (scripts/check_cached_replay.py, 0 изменённых ячеек), но не боевым прогоном. Эта
# очередь делает именно боевой прогон, чтобы в сдаче лежало то, что выдаёт predict.py.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue18 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)|run_service" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 30
}
docker compose down | Out-Null
Log "WAIT  done - GPU free, service stack down"

$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"

Step "tests18" "python -m pytest tests -q"

# 1. Образы на текущем коде и текущем рецепте
Step "build18_release" "docker build -t vreid-release ."
Step "build18_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "build18_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "build18_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."

# 2. Сдача боевым прогоном, в официальном формате
Step "predict18" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "form18" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "indep18" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "ties18" "python scripts/check_tie_independence.py"
Step "replay18" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

# 3. Контейнер: дважды, офлайн, без разделяемой памяти
Step "docker18_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q18_a:/out vreid-dev"
Step "docker18_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q18_b:/out vreid-dev"
Step "cmp18_det" "python scripts/compare_outputs.py out_q18_a out_q18_b"
Step "docker18_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q18_offline:/out vreid-dev"
Step "cmp18_offline" "python scripts/compare_outputs.py out_q18_a out_q18_offline"
Step "form18_docker" "python scripts/check_submission.py --submission out_q18_a --data %VREID_DATA%"

# 4. Производительность по методике организаторов
Step "bench18" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# 5. Отчёты и сверка с эталоном
Step "official18" "python scripts/score_validation_with_official.py"
Step "stress18" "python scripts/refusal_stress.py --out results/refusal_stress_fit.json"
Step "audit18" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe release/recipe.json --out results/refusal_audit_fit.json"
Step "errors18" "python scripts/error_analysis.py"
Step "scorecard18" "python scripts/scorecard.py --release release --submission submission"
Step "manifest18" "python scripts/make_manifest.py --release release --submission submission --run runs/hack/ft_soup_b336_fit --out reports/MANIFEST.json"

# 6. Прототип обратно в рабочее состояние
Step "compose18" "docker compose up -d"
Step "waitapi18" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "smoke18" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"
Step "logs18" "python scripts/normalize_logs.py --apply"

if ($global:failed.Count -eq 0) {
    Log "queue18 finished: все шаги прошли"
} else {
    Log "queue18 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
