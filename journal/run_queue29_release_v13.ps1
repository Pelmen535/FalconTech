# Queue 29: релиз v1.3 - ученик с фокусом на трудных парах в дистилляции (--distill-focus 4).
#     Запускать ОТВЯЗАННО от сессии (через WMI, см. queue24). ASCII only in code. ~8.5 часа.
#
# ОТКУДА. Проба queue28 на сиде 0: 79.21 без постобработки против 75.17 у того же сида v1.2
# (+4.04 при разбросе между сидами 0.23), с ре-ранжированием 81.51. Рецепт v1.2 без изменений,
# кроме --distill-focus 4: пары той же машины с другой камеры и 4 ближайшие по учителю чужие
# машины весят в лоссе дистилляции в 5 раз больше.
#
# ШАГИ.
#   1. Сиды 2 и 3 двойника (сид 0 уже есть из queue28), суп трёх, валидация.
#   2. Ворота: суп лучше двойника v1.2 (78.42 с ре-ранжированием) хотя бы на 0.3?
#      Нет - релиз не собирается.
#   3. Порог отказа заново: у нового ученика уверенность выше, 0.55 для него низок. Правило
#      записано в scripts/choose_threshold.py (вера 0.85 в почти дубликаты с той же камеры).
#   4. Три модели на всех машинах, суп, все проверки релиза, Docker, замер, эталонный scorer,
#      публикация v1.3 - только если все шаги прошли.
# Ре-ранжирование 3/2/0.2 и фокус 4 не подбираются заново: это была бы подгонка под валидацию.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"
Set-Location "E:\Хакатон"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue29 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

docker compose down | Out-Null
Log "прототип остановлен: ученикам нужна видеопамять"

$STUDENT = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 392 --epochs 18 " +
           "--P 8 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet " +
           "--distill weights/hack_dinov2_l_336_cam/best.pt --distill-w 40 --distill-focus 4 --workers 4"
$TWIN = "runs/hack/ft_soup_r392dw40f4_fit"

# ================================ 1. ДВОЙНИК ================================
if (-not (Test-Path "weights\fit18_r392_dw40f4_s0\best.pt")) {
    $global:failed += "no_probe_s0"; Log "нет сида 0 из queue28 - двойник не собрать"
}
foreach ($seed in 2, 3) {
    Step "q29_fit_s$seed" "python -m vreid.train $STUDENT --no-select --seed $seed --out weights/fit18_r392_dw40f4_s$seed"
}
Step "q29_soup_fit" ("python scripts/model_soup.py weights/fit18_r392_dw40f4_s0/best.pt " +
    "weights/fit18_r392_dw40f4_s2/best.pt weights/fit18_r392_dw40f4_s3/best.pt " +
    "--out weights/soup_r392dw40f4_fit/best.pt")
Step "q29_val_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_r392dw40f4_fit/best.pt --workers 4 --rerank"

# ================================ 2. ВОРОТА ================================
cmd /c "python scripts/gate_better.py --candidate $TWIN --baseline runs/hack/ft_soup_r392dw40_fit --mode kr --min-gain 0.3 --out results/gate_students_f4.json 2>&1" | Tee-Object -FilePath "logs\q29_gate_students.log"
$studentsOk = ($LASTEXITCODE -eq 0) -and ($global:failed.Count -eq 0)
cmd /c "python scripts/gate_better.py --candidate $TWIN --baseline runs/hack/ft_soup_r392dw40_fit --mode raw --min-gain 0 --out results/gate_students_f4_raw.json 2>&1" | Tee-Object -FilePath "logs\q29_gate_students_raw.log"
Log "ворота двойника: $(if ($studentsOk) { 'пройдены - собираю v1.3' } else { 'НЕ пройдены - релиз не собираю' })"

if ($studentsOk) {
    # ================================ 3. ПОРОГ ОТКАЗА ================================
    Step "q29_grid" "python scripts/rerank_grid.py --runs $TWIN runs/hack/ft_fit18_r392_dw40f4_s0 --out results/rerank_grid_r392dw40f4.json"
    Step "q29_stress" "python scripts/refusal_stress.py --run $TWIN --out results/refusal_stress_r392dw40f4.json"
    Step "q29_threshold" "python scripts/choose_threshold.py --stress results/refusal_stress_r392dw40f4.json --belief 0.85 --out results/threshold_choice_r392dw40f4.json"
    $THR = ""
    if (Test-Path "results\threshold_choice_r392dw40f4.value.txt") {
        $THR = (Get-Content "results\threshold_choice_r392dw40f4.value.txt" -Raw).Trim()
    }
    if (-not $THR) { $global:failed += "no_threshold"; Log "порог не выбран" } else { Log "порог отказа v1.3: $THR" }

    # ================================ 4. РЕЛИЗ v1.3 ================================
    foreach ($seed in 0, 2, 3) {
        $out = "weights\all18_r392_dw40f4_s$seed"
        if (Test-Path $out) { Remove-Item -Recurse -Force $out; Log "удалён прежний $out" }
        Step "q29_all_s$seed" "python -m vreid.train $STUDENT --val-frac 0 --seed $seed --out weights/all18_r392_dw40f4_s$seed"
    }
    Step "q29_soup_all" ("python scripts/model_soup.py weights/all18_r392_dw40f4_s0/best.pt " +
        "weights/all18_r392_dw40f4_s2/best.pt weights/all18_r392_dw40f4_s3/best.pt " +
        "--out weights/soup_r392dw40f4_all/best.pt")

    $env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
    Step "q29_tests" "python -m pytest tests -q"
    Step "q29_export" ("python scripts/export_release.py --weights weights/soup_r392dw40f4_all/best.pt " +
        "--val results/hack_ft_soup_r392dw40f4_fit_val.json --confidence top1 --threshold $THR --threshold-choice results/threshold_choice_r392dw40f4.json " +
        "--kr --k1 3 --k2 2 --lam 0.2 --crop-pad 0.05 --out release")
    Step "q29_predict" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
    Step "q29_form" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
    Step "q29_indep" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
    Step "q29_ties" "python scripts/check_tie_independence.py"
    Step "q29_replay" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

    EnsureDocker
    Step "q29_build_release" "docker build -t vreid-release ."
    Step "q29_build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
    Step "q29_build_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
    Step "q29_build_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
    Step "q29_docker_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q29_a:/out vreid-dev"
    Step "q29_docker_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q29_b:/out vreid-dev"
    Step "q29_cmp_det" "python scripts/compare_outputs.py out_q29_a out_q29_b"
    Step "q29_docker_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q29_offline:/out vreid-dev"
    Step "q29_cmp_offline" "python scripts/compare_outputs.py out_q29_a out_q29_offline"
    # Сразу после прогонов контейнера: кадры в кеше ОС (механический диск, см. queue25)
    Step "q29_bench" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"
    Step "q29_official" "python scripts/score_validation_with_official.py --run $TWIN"
    Step "q29_audit" "python scripts/refusal_audit.py --run $TWIN --recipe release/recipe.json --out results/refusal_audit_r392dw40f4.json"
    Step "q29_errors" "python scripts/error_analysis.py --run $TWIN"
    Step "q29_scorecard" "python scripts/scorecard.py --release release --submission submission"
    Step "q29_manifest" "python scripts/make_manifest.py --release release --submission submission --run $TWIN --out reports/MANIFEST.json"
    Step "q29_logs" "python scripts/normalize_logs.py --apply"
    if ($global:failed.Count -eq 0) {
        Step "q29_publish" "python scripts/publish_release.py --tag v1.3 --val results/hack_ft_soup_r392dw40f4_fit_val.json --img-size 392 --distill-w 40 --distill-focus 4"
    } else {
        Log "PUBLISH SKIPPED: провалились $($global:failed -join ', ') - v1.3 не публикую"
    }
}



# Прототип обратно. down -v: если релиз сменился, в томе галерея прежней модели
docker compose down -v | Out-Null
EnsureDocker
Step "q29_compose" "docker compose up -d"
Step "q29_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q29_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"

if ($global:failed.Count -eq 0) {
    Log "queue29 finished: все шаги прошли (двойник $(if ($studentsOk) {'лучше - v1.3 собран'} else {'НЕ лучше - релиз не собирался'}))"
} else {
    Log "queue29 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File "logs\queue29.done" -Encoding utf8
