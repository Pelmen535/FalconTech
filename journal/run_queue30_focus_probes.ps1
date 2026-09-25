# Queue 30: четыре пробы вокруг фокуса дистилляции (v1.3 = --distill-focus 4, 79.21 на сиде 0).
#     Запускать ОТВЯЗАННО от сессии (через WMI, см. queue24). ASCII only in code. ~6.5 часа.
#
#   D. фокус 4 + второй учитель - двойник v1.3 (soup_r392dw40f4_fit): он сильнее ViT-L и
#      ошибается иначе; цель дистилляции - среднее двух матриц сходств. Двойник v1.3 учился
#      на тех же fit-личностях, валидации не видел. Первой: новая связка, упадёт - будет видно сразу.
#   A. фокус 8 вместо 4 (вес трудных пар x9).
#   B. фокус 4, трудными считаются 8 ближайших чужих машин вместо 4.
#   C. фокус 4 + вес дистилляции 80 (по отдельности +4.04 и +1.41).
# Всё остальное - рецепт двойника v1.3, сид 0, P=8 явно. Сравнение сырым косинусом парно с
# сидом 0 v1.3 (79.21), порог +0.5; с ре-ранжированием - для справки. Релиз НЕ собирается:
# сначала результаты - пользователю.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"
Set-Location "E:\Хакатон"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue30 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

$BASE = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 392 --epochs 18 " +
        "--P 8 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --no-select --seed 0 --workers 4"
$REF = "runs/hack/ft_fit18_r392_dw40f4_s0"
$passed = @()

# ---- проба D ----
Step "q30_d" "python -m vreid.train $BASE --distill weights/hack_dinov2_l_336_cam/best.pt,weights/soup_r392dw40f4_fit/best.pt --distill-w 40 --distill-focus 4 --out weights/fit18_r392_dw40f4t2_s0"
Step "q30_val_d" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw40f4t2_s0/best.pt --workers 4 --rerank"
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f4t2_s0 --baseline $REF --mode raw --min-gain 0.5 --out results/gate_probe30_d.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_d.log"
if ($LASTEXITCODE -eq 0) { $passed += "D" }
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f4t2_s0 --baseline $REF --mode kr --min-gain 0.5 --out results/gate_probe30_d_kr.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_d_kr.log"

# ---- проба A ----
Step "q30_a" "python -m vreid.train $BASE --distill weights/hack_dinov2_l_336_cam/best.pt --distill-w 40 --distill-focus 8 --out weights/fit18_r392_dw40f8_s0"
Step "q30_val_a" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw40f8_s0/best.pt --workers 4 --rerank"
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f8_s0 --baseline $REF --mode raw --min-gain 0.5 --out results/gate_probe30_a.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_a.log"
if ($LASTEXITCODE -eq 0) { $passed += "A" }
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f8_s0 --baseline $REF --mode kr --min-gain 0.5 --out results/gate_probe30_a_kr.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_a_kr.log"

# ---- проба B ----
Step "q30_b" "python -m vreid.train $BASE --distill weights/hack_dinov2_l_336_cam/best.pt --distill-w 40 --distill-focus 4 --distill-hard-k 8 --out weights/fit18_r392_dw40f4k8_s0"
Step "q30_val_b" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw40f4k8_s0/best.pt --workers 4 --rerank"
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f4k8_s0 --baseline $REF --mode raw --min-gain 0.5 --out results/gate_probe30_b.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_b.log"
if ($LASTEXITCODE -eq 0) { $passed += "B" }
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f4k8_s0 --baseline $REF --mode kr --min-gain 0.5 --out results/gate_probe30_b_kr.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_b_kr.log"

# ---- проба C ----
Step "q30_c" "python -m vreid.train $BASE --distill weights/hack_dinov2_l_336_cam/best.pt --distill-w 80 --distill-focus 4 --out weights/fit18_r392_dw80f4_s0"
Step "q30_val_c" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw80f4_s0/best.pt --workers 4 --rerank"
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw80f4_s0 --baseline $REF --mode raw --min-gain 0.5 --out results/gate_probe30_c.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_c.log"
if ($LASTEXITCODE -eq 0) { $passed += "C" }
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw80f4_s0 --baseline $REF --mode kr --min-gain 0.5 --out results/gate_probe30_c_kr.json 2>&1" | Tee-Object -FilePath "logs\q30_gate_c_kr.log"

Log "пробы q30: прошли $(if ($passed.Count) { $passed -join ', ' } else { 'ни одна' })"

docker compose down -v | Out-Null
EnsureDocker
Step "q30_compose" "docker compose up -d"
Step "q30_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q30_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"

if ($global:failed.Count -eq 0) {
    Log "queue30 finished: все шаги прошли"
} else {
    Log "queue30 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File "logs\queue30.done" -Encoding utf8
