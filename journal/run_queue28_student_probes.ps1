# Queue 28: две пробы ученика ViT-B/392 - сильнее перенять учителя, не трогая инференс.
#     Запускать ОТВЯЗАННО от сессии (через WMI, см. queue24). ASCII only in code. ~3.5 часа.
#
# ЗАЧЕМ. В одинаковых условиях ученик v1.2 отстаёт от учителя ViT-L на 3.1 без постобработки
# (75.23 против 78.32) и на 2.3 с ре-ранжированием. Учитель недоиспользован. Две пробы:
#   A. вес дистилляции 80 вместо 40. Переход 20 -> 40 дал +2.0 без постобработки на двух сидах,
#      дальше 40 не пробовали;
#   B. вес 40, но в лоссе дистилляции трудные пары весят в 5 раз больше: та же машина с другой
#      камеры и 4 ближайшие по учителю чужие машины (--distill-focus 4, vreid/train.py).
# Всё остальное - рецепт двойника v1.2 (weights/fit18_r392_dw40_s0/log.json), тот же сид 0 и
# P=8 явно, чтобы авто-подбор батча не стал вторым отличием.
#
# ЧЕМ МЕРИТЬ. Сырым косинусом, парно с тем же сидом v1.2 (75.17). С ре-ранжированием два сида
# v1.2 расходятся на 1.7 пункта при разнице в сыром косинусе 0.23 - по одному сиду там ничего
# не решить. Проба проходит при +0.5 и больше. Ученики на трёх сидах и релиз НЕ собираются:
# сначала результаты - пользователю.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
$env:COMPOSE_DOCKER_CLI_BUILD = "0"
Set-Location "E:\Хакатон"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue28 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
        "--P 8 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet " +
        "--distill weights/hack_dinov2_l_336_cam/best.pt --no-select --seed 0 --workers 4"

# B первой: новый код, если он упадёт, это будет видно в первые минуты, а не через полтора часа
Step "q28_focus" "python -m vreid.train $BASE --distill-w 40 --distill-focus 4 --out weights/fit18_r392_dw40f4_s0"
Step "q28_val_focus" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw40f4_s0/best.pt --workers 4 --rerank"
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f4_s0 --baseline runs/hack/ft_fit18_r392_dw40_s0 --mode raw --min-gain 0.5 --out results/gate_probe_focus4.json 2>&1" | Tee-Object -FilePath "logs\q28_gate_focus.log"
$focusOk = ($LASTEXITCODE -eq 0)

Step "q28_dw80" "python -m vreid.train $BASE --distill-w 80 --out weights/fit18_r392_dw80_s0"
Step "q28_val_dw80" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_dw80_s0/best.pt --workers 4 --rerank"
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw80_s0 --baseline runs/hack/ft_fit18_r392_dw40_s0 --mode raw --min-gain 0.5 --out results/gate_probe_dw80.json 2>&1" | Tee-Object -FilePath "logs\q28_gate_dw80.log"
$dw80Ok = ($LASTEXITCODE -eq 0)

# Для отчёта: те же пробы с ре-ранжированием релиза (шумно на одном сиде, решение не по ним)
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw40f4_s0 --baseline runs/hack/ft_fit18_r392_dw40_s0 --mode kr --min-gain 0.5 --out results/gate_probe_focus4_kr.json 2>&1" | Tee-Object -FilePath "logs\q28_gate_focus_kr.log"
cmd /c "python scripts/gate_better.py --candidate runs/hack/ft_fit18_r392_dw80_s0 --baseline runs/hack/ft_fit18_r392_dw40_s0 --mode kr --min-gain 0.5 --out results/gate_probe_dw80_kr.json 2>&1" | Tee-Object -FilePath "logs\q28_gate_dw80_kr.log"
Log "пробы: фокус $(if ($focusOk) {'ПРОШЁЛ'} else {'не прошёл'}), вес 80 $(if ($dw80Ok) {'ПРОШЁЛ'} else {'не прошёл'})"

docker compose down -v | Out-Null
EnsureDocker
Step "q28_compose" "docker compose up -d"
Step "q28_waitapi" "python scripts/wait_http.py http://localhost:8080/health --timeout 300"
Step "q28_smoke" "python scripts/smoke_backend.py --url http://localhost:8080 --output var/smoke_compose.json"

if ($global:failed.Count -eq 0) {
    Log "queue28 finished: все шаги прошли"
} else {
    Log "queue28 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File "logs\queue28.done" -Encoding utf8
