# Queue 21: разведка по разрешению входа. Один сид на 392 и один на 448.
#     powershell -ExecutionPolicy Bypass -File journal\run_queue21_resolution.ps1
# ASCII only in code. Около 6 часов, два обучения.
#
# ЗАЧЕМ. Шкала производительности опубликована и однозначна (ответ 34): до 40 мс латентности
# и от 100 FPS - полный балл. Релиз стоит на 28.5 мс и 195.7 FPS, то есть часть бюджета не
# потрачена. Тратить её надо ВНУТРИ замеряемого пути: там выигрыш не зависит ни от одного
# допущения о том, как жюри посчитает ре-ранжирование.
#
# Сколько можно потратить - измерено, а не прикинуто (scripts/resolution_budget.py):
#   вход 392: латентность 32.5 мс, 167.4 FPS -> 20/20, запас по пропускной 67%
#   вход 448: латентность 34.9 мс, 117.7 FPS -> 20/20, запас по пропускной 18%
# Запас важен: стенд жюри - A5000, а мерили мы на RTX 5060 Ti, и переносимость не проверена.
#
# Есть ли что извлекать из большего входа - тоже измерено: медиана короткой стороны кропа
# 846 px, ниже 448 только 0.8% кропов. На входе 336 мы выбрасываем реальную детализацию.
#
# Задаром разрешение не даётся: уже обученная на 336 модель на входах 392 и 448 дала
# 73.95 и 73.88 против 74.02 - то есть ничего. Нужно переобучение, и вот оно.
#
# ЧЕСТНАЯ ОГОВОРКА. Один сид на разброс между сидами 2.90 пункта не даёт разрешить прирост
# меньше трёх пунктов. Поэтому это РАЗВЕДКА: если одно из разрешений даст заметный скачок,
# он подтверждается вторым сидом, и только потом собирается релиз.
#
# P не задан явно: на большем входе батч в память не влезает прежним, и --P auto подберёт
# максимальный. Это делает сравнение с fit18_s0 (P=11) не вполне чистым - размер батча
# влияет на отбор троек. Другого выхода нет, и умалчивать об этом не будем.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue21 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 30
}
docker compose down | Out-Null
Log "WAIT  done - GPU free"

# Рецепт fit18 без единого изменения, кроме входа. Учитель остаётся на 336: кроп для него
# масштабируется до его входа (vreid/train.py, teacher_input) - переобучать ViT-L на 448
# это ещё часы GPU, а дистиллируется геометрия сходств, ей разрешение учителя не критично.
$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"
$BASE = "--config configs/hackathon.yaml --backbone dinov2_b --epochs 18 --P auto --K 4 " +
        "--freeze-blocks 0 --cam-aware --cross-cam-triplet --distill $TEACHER --distill-w 20 " +
        "--no-select --seed 0 --workers 4"

Step "q21_train_392" "python -m vreid.train $BASE --img-size 392 --out weights/fit18_r392_s0"
Step "q21_val_392" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r392_s0/best.pt --workers 4 --rerank"

Step "q21_train_448" "python -m vreid.train $BASE --img-size 448 --out weights/fit18_r448_s0"
Step "q21_val_448" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_r448_s0/best.pt --workers 4 --rerank"

# Сравнение с тремя сидами базового рецепта: важен не факт "больше", а больше ли разброса.
Step "q21_compare" ("python scripts/compare_training.py --baseline weights/fit18_s0 weights/fit18_s2 " +
    "weights/fit18_s3 --candidate weights/fit18_r392_s0 weights/fit18_r448_s0 " +
    "--out results/training_resolution.json")

if ($global:failed.Count -eq 0) {
    Log "queue21 finished: все шаги прошли"
} else {
    Log "queue21 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
