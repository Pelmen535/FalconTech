# Queue 24 (ночь 22-23.09): релиз на входе 392 -> публикация v1.1 -> поиск следующего прироста.
#     запускать ОТВЯЗАННО от сессии (иначе умрёт вместе с ней, так уже было дважды):
#     Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
#         CommandLine='powershell.exe -ExecutionPolicy Bypass -NoProfile -File E:\Хакатон\journal\run_queue24_night.ps1';
#         CurrentDirectory='E:\Хакатон'}
# ASCII only in code. Около 8 часов.
#
# ЧАСТЬ A - РЕЛИЗ. Двойник на 392 уже обучен и провалидирован: суп трёх сидов даёт mAP@10
# 77.60 с k-reciprocal 3/2/0.2 против 74.21 у опубликованного v1.0 по тому же безрисковому
# пути (без каскада). Боевой суп недоучен: сид 0 готов, сид 2 оборвался на 10-й эпохе вместе
# с сессией (возобновления в vreid/train.py нет - частичный best.pt удаляется и учится
# заново), сида 3 нет. Дальше - пересборка релиза, все проверки, замер на новом входе, и
# публикация, если ВСЕ шаги прошли и двойник лучше опубликованного.
#
# ЧАСТЬ B - ПОИСК. Двойник на 392 почти догнал учителя, а вход поднят до разумного предела
# (448 отвергнут: запас по пропускной 11%). Проверяется, недоиспользован ли учитель:
# вес дистилляции 20 -> 40. На 336 переход 1 -> 20 дал +1.76, насыщение не проверяли.
# ДВА сида, а не один: разброс между сидами на 336 доходил до 3.6 пункта, и одиночный прогон
# не отличает эффект в пункт от шума. Сравнение парное - с сидами 0 и 2 того же рецепта на
# весе 20, которые уже обучены.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"
Set-Location "E:\Хакатон"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue24 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

# ================================ ЧАСТЬ A: РЕЛИЗ ================================
$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"
$COMMON = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 392 --epochs 18 " +
          "--P auto --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet " +
          "--distill $TEACHER --workers 4"

# Оборванный на 10-й эпохе сид 2 - не модель, а мусор с правдоподобным именем
if (Test-Path "weights\all18_r392_s2") { Remove-Item -Recurse -Force "weights\all18_r392_s2"; Log "удалён частичный all18_r392_s2" }

Step "q24_all_s2" "python -m vreid.train $COMMON --distill-w 20 --val-frac 0 --seed 2 --out weights/all18_r392_s2"
Step "q24_all_s3" "python -m vreid.train $COMMON --distill-w 20 --val-frac 0 --seed 3 --out weights/all18_r392_s3"
Step "q24_soup_all" ("python scripts/model_soup.py weights/all18_r392_s0/best.pt " +
    "weights/all18_r392_s2/best.pt weights/all18_r392_s3/best.pt --out weights/soup_r392_all/best.pt")

# Разброс между сидами на 392: без него не понять, какой эффект вообще различим (часть B)
Step "q24_val_fit_s2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_s2/best.pt --workers 4 --rerank"
Step "q24_val_fit_s3" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_s3/best.pt --workers 4 --rerank"

# Постобработка и отказ заново на НОВОМ двойнике (в рецепт автоматически не идут - только отчёт)
Step "q24_grid" "python scripts/rerank_grid.py --runs runs/hack/ft_soup_r392_fit runs/hack/ft_fit18_r392_s0 --out results/rerank_grid_r392.json"
Step "q24_stress" "python scripts/refusal_stress.py --run runs/hack/ft_soup_r392_fit --out results/refusal_stress_r392.json"

$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"

Step "q24_tests" "python -m pytest tests -q"
# Релиз: модель на 392, без каскада (решение владельца - без рисков по производительности).
# Порог 0.55 остаётся ниже оптимума валидации 392 (0.584): та же страховка, проверка - q24_audit.
Step "q24_export" ("python scripts/export_release.py --weights weights/soup_r392_all/best.pt " +
    "--val results/hack_ft_soup_r392_fit_val.json --confidence top1 --threshold 0.55 " +
    "--kr --k1 3 --k2 2 --lam 0.2 --crop-pad 0.05 --out release")
Step "q24_predict" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "q24_form" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "q24_indep" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "q24_ties" "python scripts/check_tie_independence.py"
Step "q24_replay" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

EnsureDocker
Step "q24_build_release" "docker build -t vreid-release ."
Step "q24_build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "q24_build_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "q24_build_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
Step "q24_docker_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q24_a:/out vreid-dev"
Step "q24_docker_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q24_b:/out vreid-dev"
Step "q24_cmp_det" "python scripts/compare_outputs.py out_q24_a out_q24_b"
Step "q24_docker_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q24_offline:/out vreid-dev"
Step "q24_cmp_offline" "python scripts/compare_outputs.py out_q24_a out_q24_offline"
Step "q24_bench" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"
Step "q24_official" "python scripts/score_validation_with_official.py"
Step "q24_audit" "python scripts/refusal_audit.py --run runs/hack/ft_soup_r392_fit --recipe release/recipe.json --out results/refusal_audit_r392.json"
Step "q24_errors" "python scripts/error_analysis.py"
Step "q24_scorecard" "python scripts/scorecard.py --release release --submission submission"
Step "q24_manifest" "python scripts/make_manifest.py --release release --submission submission --run runs/hack/ft_soup_r392_fit --out reports/MANIFEST.json"
Step "q24_logs" "python scripts/normalize_logs.py --apply"

# Публикация - ТОЛЬКО если не упал ни один шаг. Скрипт сам откажется, если двойник не лучше
# опубликованного (v1.0: 74.21 по безрисковому пути).
if ($global:failed.Count -eq 0) {
    Step "q24_publish" "python scripts/publish_release.py --tag v1.1 --val results/hack_ft_soup_r392_fit_val.json --img-size 392"
} else {
    Log "PUBLISH SKIPPED: провалились $($global:failed -join ', ') - v1.1 не публикую"
}

Step "q24_compose" "docker compose up -d"
Step "q24_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q24_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"
Log "PART A done, failed so far: $($global:failed -join ', ')"
docker compose down | Out-Null

# ================================ ЧАСТЬ B: ПОИСК ================================
$FIT = "$COMMON --no-select"
Step "q24_dw40_s0" "python -m vreid.train $FIT --distill-w 40 --seed 0 --out weights/fit18_r392_dw40_s0"
Step "q24_val_dw40_s0" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw40_s0/best.pt --workers 4 --rerank"
Step "q24_dw40_s2" "python -m vreid.train $FIT --distill-w 40 --seed 2 --out weights/fit18_r392_dw40_s2"
Step "q24_val_dw40_s2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw40_s2/best.pt --workers 4 --rerank"
# compare_training.py возвращает 1, когда гипотеза ОТВЕРГНУТА. Отрицательный результат - это
# результат, а не сломанный шаг: вердикт читается из лога, код возврата не значит ошибку.
Step "q24_compare_dw40" ("python scripts/compare_training.py --baseline weights/fit18_r392_s0 weights/fit18_r392_s2 " +
    "--candidate weights/fit18_r392_dw40_s0 weights/fit18_r392_dw40_s2 --out results/training_dw40_r392.json & exit /b 0")

if ($global:failed.Count -eq 0) {
    Log "queue24 finished: все шаги прошли"
} else {
    Log "queue24 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File "logs\queue24.done" -Encoding utf8
