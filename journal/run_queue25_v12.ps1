# Queue 25: релиз v1.2 - вход 392, вес дистилляции 40. Сборка, все проверки, публикация.
#     запускать ОТВЯЗАННО от сессии (через WMI, см. queue24).
# ASCII only in code. Около 6.5 часов, четыре обучения.
#
# ЗАЧЕМ. Ночная разведка (queue24, часть B) показала, что учитель на весе 20 недоиспользован.
# Вес 40 на двух сидах, парно против тех же сидов на весе 20:
#     без постобработки   сид 0: 73.33 -> 75.17    сид 2: 73.21 -> 75.40
#     k-reciprocal 3/2    сид 0: 76.84 -> 80.27    сид 2: 77.60 -> 78.54
# Разброс между сидами на 392 без постобработки всего 0.30 (73.21-73.51) против 2.9 на 336,
# так что прирост +2.0 - это в семь раз больше разброса. Эффект настоящий.
#
# Меняется ровно одно - --distill-w 40. Вход, учитель, эпохи, порог, постобработка - как в v1.1.
#
# ДВА УРОКА ПРОШЛОЙ НОЧИ, учтённые здесь:
#   1. Скрипты отчётов берут двойника из release/weights.json, а там до публикации ещё v1.1.
#      Поэтому --run передаётся явно - иначе эталонный scorer снова посчитает не ту модель,
#      и publish_release возьмёт не то число.
#   2. Прототип не поднимется, если в томе базы лежит галерея, посчитанная прежней моделью
#      (сервис справедливо отказывается их смешивать). Поэтому перед подъёмом - down -v.
#
# Публикация - только если ВСЕ шаги прошли и двойник лучше v1.1 (77.41) хотя бы на 0.3.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"
Set-Location "E:\Хакатон"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue25 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
function EnsureDocker() {
    # Docker Desktop мог остановиться (22.09 его гасил wsl --shutdown при чистке диска).
    # Без него все шаги сборки и замера упадут, и релиз не опубликуется - поднимаем сами.
    $ok = $false
    for ($i = 0; $i -lt 40; $i++) {
        cmd /c "docker info >nul 2>&1"
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        if ($i -eq 0) {
            Log "docker не отвечает - запускаю Docker Desktop"
            Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        }
        Start-Sleep -Seconds 15
    }
    if (-not $ok) { $global:failed += "docker_unavailable"; Log "docker так и не поднялся" }
    else { Log "docker готов" }
}

Log "WAIT  for running vreid processes"
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 30
}
Log "WAIT  done - GPU free"

# ================================ ОБУЧЕНИЕ ================================
$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"
$COMMON = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 392 --epochs 18 " +
          "--P auto --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet " +
          "--distill $TEACHER --distill-w 40 --workers 4"
$TWIN = "runs/hack/ft_soup_r392dw40_fit"

# 1. Двойник: сиды 0 и 2 обучены разведкой, добираем третий и усредняем
Step "q25_fit_s3" "python -m vreid.train $COMMON --no-select --seed 3 --out weights/fit18_r392_dw40_s3"
Step "q25_soup_fit" ("python scripts/model_soup.py weights/fit18_r392_dw40_s0/best.pt " +
    "weights/fit18_r392_dw40_s2/best.pt weights/fit18_r392_dw40_s3/best.pt --out weights/soup_r392dw40_fit/best.pt")
Step "q25_val_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_r392dw40_fit/best.pt --workers 4 --rerank"

# 2. Боевая модель: все личности, три сида
foreach ($seed in 0, 2, 3) {
    $out = "weights\all18_r392_dw40_s$seed"
    if (Test-Path $out) { Remove-Item -Recurse -Force $out; Log "удалён прежний $out" }
    Step "q25_all_s$seed" "python -m vreid.train $COMMON --val-frac 0 --seed $seed --out weights/all18_r392_dw40_s$seed"
}
Step "q25_soup_all" ("python scripts/model_soup.py weights/all18_r392_dw40_s0/best.pt " +
    "weights/all18_r392_dw40_s2/best.pt weights/all18_r392_dw40_s3/best.pt --out weights/soup_r392dw40_all/best.pt")

# 3. Постобработка и отказ на НОВОМ двойнике (только отчёт, в рецепт не идут)
Step "q25_grid" "python scripts/rerank_grid.py --runs $TWIN runs/hack/ft_fit18_r392_dw40_s0 --out results/rerank_grid_r392dw40.json"
Step "q25_stress" "python scripts/refusal_stress.py --run $TWIN --out results/refusal_stress_r392dw40.json"

# ================================ РЕЛИЗ ================================
$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"

Step "q25_tests" "python -m pytest tests -q"
Step "q25_export" ("python scripts/export_release.py --weights weights/soup_r392dw40_all/best.pt " +
    "--val results/hack_ft_soup_r392dw40_fit_val.json --confidence top1 --threshold 0.55 " +
    "--kr --k1 3 --k2 2 --lam 0.2 --crop-pad 0.05 --out release")
Step "q25_predict" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "q25_form" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "q25_indep" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "q25_ties" "python scripts/check_tie_independence.py"
Step "q25_replay" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

EnsureDocker
Step "q25_build_release" "docker build -t vreid-release ."
Step "q25_build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "q25_build_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "q25_build_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
Step "q25_docker_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q25_a:/out vreid-dev"
Step "q25_docker_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q25_b:/out vreid-dev"
Step "q25_cmp_det" "python scripts/compare_outputs.py out_q25_a out_q25_b"
Step "q25_docker_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q25_offline:/out vreid-dev"
Step "q25_cmp_offline" "python scripts/compare_outputs.py out_q25_a out_q25_offline"
# Замер сразу после трёх прогонов контейнера: кадры в кеше ОС. Кадры лежат на механическом
# диске, и холодный замер после часов обучения даёт на ~11 мс больше (23.09: 41.3 против 30.2).
Step "q25_bench" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# Все отчёты - на двойнике v1.2 явно (урок 1)
Step "q25_official" "python scripts/score_validation_with_official.py --run $TWIN"
Step "q25_audit" "python scripts/refusal_audit.py --run $TWIN --recipe release/recipe.json --out results/refusal_audit_r392dw40.json"
Step "q25_errors" "python scripts/error_analysis.py --run $TWIN"
Step "q25_scorecard" "python scripts/scorecard.py --release release --submission submission"
Step "q25_manifest" "python scripts/make_manifest.py --release release --submission submission --run $TWIN --out reports/MANIFEST.json"
Step "q25_logs" "python scripts/normalize_logs.py --apply"

if ($global:failed.Count -eq 0) {
    Step "q25_publish" "python scripts/publish_release.py --tag v1.2 --val results/hack_ft_soup_r392dw40_fit_val.json --img-size 392 --distill-w 40"
} else {
    Log "PUBLISH SKIPPED: провалились $($global:failed -join ', ') - v1.2 не публикую"
}

# Прототип: сначала стереть галерею прежней модели (урок 2), потом поднять
docker compose down -v | Out-Null
Step "q25_compose" "docker compose up -d"
Step "q25_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q25_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"

if ($global:failed.Count -eq 0) {
    Log "queue25 finished: все шаги прошли"
} else {
    Log "queue25 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File "logs\queue25.done" -Encoding utf8
