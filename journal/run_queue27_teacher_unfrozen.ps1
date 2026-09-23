# Queue 27: учитель без заморозки блоков - ViT-L на 336, учатся все 24 блока вместо 16 - и,
#     если он лучше, ученики и релиз v1.3. Запускать ОТВЯЗАННО от сессии (через WMI, см. queue24).
# ASCII only in code. До ~14 часов, если пройдут обе проверки; ~3.5 часа, если учитель не лучше.
#
# ЗАЧЕМ. Дистилляция упирается в учителя: v1.2 (78.42 с ре-ранжированием) догнал своего учителя.
# Вход 392 учителю не помог (queue26: 77.62 против 78.32 на 336). Вторая причина потолка -
# заморозка: у всех учителей первые 8 блоков из 24 и patch_embed не учились (--freeze-blocks 8),
# это решение было ради памяти, а не ради точности. С пересчётом активаций (--grad-ckpt) ViT-L
# учится целиком в 5.7 ГиБ на шаг (замер: без него 14.8 ГиБ из 16 - впритык).
#
# ДВОЕ ВОРОТ (scripts/gate_better.py), как в queue26:
#   1. после учителя: лучше ли он честного учителя на 336 с заморозкой (l336cam_fit, 78.32 сырым
#      косинусом, тот же протокол: без отбора эпохи) хотя бы на 0.3? Сравнение с ним, а не с
#      учителем v1.2 (77.95): так проверяется именно разморозка, а не разница протоколов.
#   2. после супа учеников-двойников: лучше ли он двойника v1.2 (78.42) хотя бы на 0.3?
# Публикация v1.3 - только если прошли все шаги и собственная проверка публикатора.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"
Set-Location "E:\Хакатон"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue27 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

docker compose down | Out-Null
Log "прототип остановлен: учителю ViT-L нужна видеопамять"

# ================================ 1. УЧИТЕЛЬ ================================
# Рецепт честного учителя l336cam_fit (weights/l336cam_fit/log.json) без изменений, кроме:
#   --freeze-blocks 0 - то, ради чего всё: учатся patch_embed и все 24 блока;
#   --grad-ckpt       - тот же градиент, иначе шаг занимает 14.8 ГиБ из 16.
$TEACHER_NEW = "weights/l336f0_fit/best.pt"
Step "q27_teacher" ("python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_l " +
    "--img-size 336 --epochs 20 --P 5 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet " +
    "--grad-ckpt --no-select --seed 0 --workers 4 --out weights/l336f0_fit")
Step "q27_val_teacher" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:$TEACHER_NEW --workers 4 --rerank"

# Ворота 1: разморозка лучше заморозки при прочих равных (l336cam_fit, 78.32 сырым косинусом)?
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_l336f0_fit --baseline runs/hack/ft_l336cam_fit --mode raw --min-gain 0.3 --out results/gate_teacher_l336f0.json 2>&1" | Tee-Object -FilePath "logs\q27_gate_teacher.log"
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
        Step "q27_fit_s$seed" "python -m vreid.train $STUDENT --no-select --seed $seed --out weights/fit18_r392_dw40tf0_s$seed"
    }
    Step "q27_soup_fit" ("python scripts/model_soup.py weights/fit18_r392_dw40tf0_s0/best.pt " +
        "weights/fit18_r392_dw40tf0_s2/best.pt weights/fit18_r392_dw40tf0_s3/best.pt " +
        "--out weights/soup_r392dw40tf0_fit/best.pt")
    Step "q27_val_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_r392dw40tf0_fit/best.pt --workers 4 --rerank"

    # Ворота 2: суп учеников лучше двойника опубликованного v1.2 (78.42)?
    cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_soup_r392dw40tf0_fit --baseline runs/hack/ft_soup_r392dw40_fit --mode kr --min-gain 0.3 --out results/gate_students_tf0.json 2>&1" | Tee-Object -FilePath "logs\q27_gate_students.log"
    $studentsOk = ($LASTEXITCODE -eq 0) -and ($global:failed.Count -eq 0)
    Log "ворота учеников: $(if ($studentsOk) { 'пройдены - собираю v1.3' } else { 'НЕ пройдены - релиз не собираю' })"
}

if ($studentsOk) {
    # ================================ 3. РЕЛИЗ v1.3 ================================
    $TWIN = "runs/hack/ft_soup_r392dw40tf0_fit"
    foreach ($seed in 0, 2, 3) {
        $out = "weights\all18_r392_dw40tf0_s$seed"
        if (Test-Path $out) { Remove-Item -Recurse -Force $out; Log "удалён прежний $out" }
        Step "q27_all_s$seed" "python -m vreid.train $STUDENT --val-frac 0 --seed $seed --out weights/all18_r392_dw40tf0_s$seed"
    }
    Step "q27_soup_all" ("python scripts/model_soup.py weights/all18_r392_dw40tf0_s0/best.pt " +
        "weights/all18_r392_dw40tf0_s2/best.pt weights/all18_r392_dw40tf0_s3/best.pt " +
        "--out weights/soup_r392dw40tf0_all/best.pt")
    Step "q27_grid" "python scripts/rerank_grid.py --runs $TWIN runs/hack/ft_fit18_r392_dw40tf0_s0 --out results/rerank_grid_r392dw40tf0.json"
    Step "q27_stress" "python scripts/refusal_stress.py --run $TWIN --out results/refusal_stress_r392dw40tf0.json"

    $env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
    Step "q27_tests" "python -m pytest tests -q"
    Step "q27_export" ("python scripts/export_release.py --weights weights/soup_r392dw40tf0_all/best.pt " +
        "--val results/hack_ft_soup_r392dw40tf0_fit_val.json --confidence top1 --threshold 0.55 " +
        "--kr --k1 3 --k2 2 --lam 0.2 --crop-pad 0.05 --out release")
    Step "q27_predict" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
    Step "q27_form" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
    Step "q27_indep" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
    Step "q27_ties" "python scripts/check_tie_independence.py"
    Step "q27_replay" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

    EnsureDocker
    Step "q27_build_release" "docker build -t vreid-release ."
    Step "q27_build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
    Step "q27_build_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
    Step "q27_build_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
    Step "q27_docker_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q27_a:/out vreid-dev"
    Step "q27_docker_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q27_b:/out vreid-dev"
    Step "q27_cmp_det" "python scripts/compare_outputs.py out_q27_a out_q27_b"
    Step "q27_docker_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_q27_offline:/out vreid-dev"
    Step "q27_cmp_offline" "python scripts/compare_outputs.py out_q27_a out_q27_offline"
    # Сразу после прогонов контейнера: кадры в кеше ОС (механический диск, см. queue25)
    Step "q27_bench" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"
    Step "q27_official" "python scripts/score_validation_with_official.py --run $TWIN"
    Step "q27_audit" "python scripts/refusal_audit.py --run $TWIN --recipe release/recipe.json --out results/refusal_audit_r392dw40tf0.json"
    Step "q27_errors" "python scripts/error_analysis.py --run $TWIN"
    Step "q27_scorecard" "python scripts/scorecard.py --release release --submission submission"
    Step "q27_manifest" "python scripts/make_manifest.py --release release --submission submission --run $TWIN --out reports/MANIFEST.json"
    Step "q27_logs" "python scripts/normalize_logs.py --apply"
    if ($global:failed.Count -eq 0) {
        Step "q27_publish" "python scripts/publish_release.py --tag v1.3 --val results/hack_ft_soup_r392dw40tf0_fit_val.json --img-size 392 --distill-w 40"
    } else {
        Log "PUBLISH SKIPPED: провалились $($global:failed -join ', ') - v1.3 не публикую"
    }
}

# Прототип обратно. down -v: если релиз сменился, в томе галерея прежней модели
docker compose down -v | Out-Null
EnsureDocker
Step "q27_compose" "docker compose up -d"
Step "q27_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q27_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"

if ($global:failed.Count -eq 0) {
    Log "queue27 finished: все шаги прошли (учитель $(if ($teacherOk) {'лучше'} else {'НЕ лучше'}), ученики $(if ($studentsOk) {'лучше'} else {'не собирались или НЕ лучше'}))"
} else {
    Log "queue27 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File "logs\queue27.done" -Encoding utf8
