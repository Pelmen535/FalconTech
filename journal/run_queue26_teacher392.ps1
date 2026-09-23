# Queue 26: учитель посильнее - ViT-L на входе 392 - и, если он лучше, ученики и релиз v1.3.
#     запускать ОТВЯЗАННО от сессии (через WMI, см. queue24).
# ASCII only in code. До ~15 часов, если пройдут обе проверки; ~4.5 часа, если учитель не лучше.
#
# ЗАЧЕМ. v1.2 с ре-ранжированием (78.42) догнал своего учителя ViT-L на 336 по всей галерее
# (сырым косинусом 77.95 у того, от которого учились, 78.32 у честного двойника). Дистилляция
# упирается в учителя. Учитель на 336 при этом видит кроп хуже ученика: тот учится на 392, и
# батч для учителя сжимается до 336 (vreid/train.py, teacher_input). Учитель на 392 видит ту же
# детализацию, что ученик. На 448 смысла нет: ученик подаёт 392-е кропы, растянуть их до 448 -
# ноль новой информации.
#
# ДВОЕ ВОРОТ, чтобы не жечь часы впустую (scripts/gate_better.py):
#   1. после учителя: лучше ли он прежнего сырым косинусом хотя бы на 0.3? Нет - ученики не учатся.
#   2. после супа учеников-двойников: лучше ли он двойника v1.2 (78.42) хотя бы на 0.3? Нет -
#      релиз не собирается.
# Публикация v1.3 - только если прошли все шаги, и её собственная проверка (ещё +0.3 над
# опубликованным) тоже пройдена.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"
Set-Location "E:\Хакатон"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue26 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
Log "прототип остановлен: учителю ViT-L нужна вся видеопамять"

# ================================ 1. УЧИТЕЛЬ ================================
# Рецепт прежнего учителя (weights/hack_dinov2_l_336_cam/log.json) без изменений, кроме:
#   --img-size 392  - то, ради чего всё: учитель видит ту же детализацию, что ученик;
#   --no-select     - эпоха не выбирается по валидации, на которой потом меряем учеников;
#   --grad-ckpt     - тот же градиент, иначе ViT-L на 392 с батчем P=5 не влезает в 16 ГиБ
#                     (замерено: 14.5 ГиБ на шаг без него, 4.9 с ним, плюс состояние Adam).
$TEACHER_NEW = "weights/l392cam_fit/best.pt"
Step "q26_teacher" ("python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_l " +
    "--img-size 392 --epochs 20 --P 5 --K 4 --freeze-blocks 8 --cam-aware --cross-cam-triplet " +
    "--grad-ckpt --no-select --seed 0 --workers 4 --out weights/l392cam_fit")
Step "q26_val_teacher" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:$TEACHER_NEW --workers 4 --rerank"

# Ворота 1: учитель лучше того, от которого учились v1.1 и v1.2 (hack_dinov2_l_336_cam, 77.95
# сырым косинусом)? Иначе пять часов учеников впустую.
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_l392cam_fit --baseline runs/hack/ft_hack_dinov2_l_336_cam --mode raw --min-gain 0.3 --out results/gate_teacher_l392.json 2>&1" | Tee-Object -FilePath "logs\q26_gate_teacher.log"
$teacherOk = ($LASTEXITCODE -eq 0) -and ($global:failed.Count -eq 0)
Log "ворота учителя: $(if ($teacherOk) { 'пройдены' } else { 'НЕ пройдены - ученики не учатся' })"

$studentsOk = $false
if ($teacherOk) {
    # ================================ 2. УЧЕНИКИ-ДВОЙНИКИ ================================
    # Рецепт v1.2 без изменений, кроме учителя. Три сида и СУП - урок прошлой ночи: сравнивать
    # надо супы, а не сиды, если изменение влияет на разнообразие между ними.
    $STUDENT = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 392 --epochs 18 " +
               "--P auto --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet " +
               "--distill $TEACHER_NEW --distill-w 40 --workers 4"
    foreach ($seed in 0, 2, 3) {
        Step "q26_fit_s$seed" "python -m vreid.train $STUDENT --no-select --seed $seed --out weights/fit18_r392_dw40t392_s$seed"
    }
    Step "q26_soup_fit" ("python scripts/model_soup.py weights/fit18_r392_dw40t392_s0/best.pt " +
        "weights/fit18_r392_dw40t392_s2/best.pt weights/fit18_r392_dw40t392_s3/best.pt " +
        "--out weights/soup_r392dw40t392_fit/best.pt")
    Step "q26_val_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_r392dw40t392_fit/best.pt --workers 4 --rerank"

    # Ворота 2: суп учеников лучше двойника опубликованного v1.2 (78.42)?
    cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_soup_r392dw40t392_fit --baseline runs/hack/ft_soup_r392dw40_fit --mode kr --min-gain 0.3 --out results/gate_students_t392.json 2>&1" | Tee-Object -FilePath "logs\q26_gate_students.log"
    $studentsOk = ($LASTEXITCODE -eq 0) -and ($global:failed.Count -eq 0)
    Log "ворота учеников: $(if ($studentsOk) { 'пройдены - собираю v1.3' } else { 'НЕ пройдены - релиз не собираю' })"
}

if ($studentsOk) {
    # ================================ 3. РЕЛИЗ v1.3 ================================
    $TWIN = "runs/hack/ft_soup_r392dw40t392_fit"
    foreach ($seed in 0, 2, 3) {
        $out = "weights\all18_r392_dw40t392_s$seed"
        if (Test-Path $out) { Remove-Item -Recurse -Force $out; Log "удалён прежний $out" }
        Step "q26_all_s$seed" "python -m vreid.train $STUDENT --val-frac 0 --seed $seed --out weights/all18_r392_dw40t392_s$seed"
    }
    Step "q26_soup_all" ("python scripts/model_soup.py weights/all18_r392_dw40t392_s0/best.pt " +
        "weights/all18_r392_dw40t392_s2/best.pt weights/all18_r392_dw40t392_s3/best.pt " +
        "--out weights/soup_r392dw40t392_all/best.pt")
    Step "q26_grid" "python scripts/rerank_grid.py --runs $TWIN runs/hack/ft_fit18_r392_dw40t392_s0 --out results/rerank_grid_r392dw40t392.json"
    Step "q26_stress" "python scripts/refusal_stress.py --run $TWIN --out results/refusal_stress_r392dw40t392.json"

    $env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
    Step "q26_tests" "python -m pytest tests -q"
    Step "q26_export" ("python scripts/export_release.py --weights weights/soup_r392dw40t392_all/best.pt " +
        "--val results/hack_ft_soup_r392dw40t392_fit_val.json --confidence top1 --threshold 0.55 " +
        "--kr --k1 3 --k2 2 --lam 0.2 --crop-pad 0.05 --out release")
    Step "q26_predict" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
    Step "q26_form" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
    Step "q26_indep" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
    Step "q26_ties" "python scripts/check_tie_independence.py"
    Step "q26_replay" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

    EnsureDocker
    Step "q26_build_release" "docker build -t vreid-release ."
    Step "q26_build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
    Step "q26_build_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
    Step "q26_build_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
    Step "q26_docker_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q26_a:/out vreid-dev"
    Step "q26_docker_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q26_b:/out vreid-dev"
    Step "q26_cmp_det" "python scripts/compare_outputs.py out_q26_a out_q26_b"
    Step "q26_docker_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q26_offline:/out vreid-dev"
    Step "q26_cmp_offline" "python scripts/compare_outputs.py out_q26_a out_q26_offline"
    # Сразу после прогонов контейнера: кадры в кеше ОС (механический диск, см. queue25)
    Step "q26_bench" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"
    Step "q26_official" "python scripts/score_validation_with_official.py --run $TWIN"
    Step "q26_audit" "python scripts/refusal_audit.py --run $TWIN --recipe release/recipe.json --out results/refusal_audit_r392dw40t392.json"
    Step "q26_errors" "python scripts/error_analysis.py --run $TWIN"
    Step "q26_scorecard" "python scripts/scorecard.py --release release --submission submission"
    Step "q26_manifest" "python scripts/make_manifest.py --release release --submission submission --run $TWIN --out reports/MANIFEST.json"
    Step "q26_logs" "python scripts/normalize_logs.py --apply"
    if ($global:failed.Count -eq 0) {
        Step "q26_publish" "python scripts/publish_release.py --tag v1.3 --val results/hack_ft_soup_r392dw40t392_fit_val.json --img-size 392 --distill-w 40"
    } else {
        Log "PUBLISH SKIPPED: провалились $($global:failed -join ', ') - v1.3 не публикую"
    }
}

# Прототип обратно. down -v: если релиз сменился, в томе галерея прежней модели
docker compose down -v | Out-Null
EnsureDocker
Step "q26_compose" "docker compose up -d"
Step "q26_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q26_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"

if ($global:failed.Count -eq 0) {
    Log "queue26 finished: все шаги прошли (учитель $(if ($teacherOk) {'лучше'} else {'НЕ лучше'}), ученики $(if ($studentsOk) {'лучше'} else {'не собирались или НЕ лучше'}))"
} else {
    Log "queue26 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File "logs\queue26.done" -Encoding utf8
