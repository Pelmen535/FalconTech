# Queue 22: релиз на входе 392 - супы двойника и боевой модели, переподбор постобработки и порога.
#     powershell -ExecutionPolicy Bypass -File journal\run_queue22_release392.ps1
# ASCII only in code. Около 10 часов, пять обучений.
#
# ЗАЧЕМ. Разведка (queue21) показала, что разрешение входа - настоящий рычаг, и он ничего не
# стоит по баллу, потому что шкала производительности опубликована и запас у нас был:
#
#   одинаковой постобработкой (k-reciprocal 3/2/0.2) по кешам:
#     три сида 336: 75.09 / 75.37 / 71.78  (разброс 3.59), суп-двойник 74.21
#     один сид 392: 76.84
#     один сид 448: 78.17
#
# Почему 392, а не 448. Замерено на НАСТОЯЩЕМ конвейере (vreid.bench_full на временных
# релизах из разведочных весов), а не на чистом форварде:
#     336: 28.5 мс, 195.7 FPS   запас по пропускной 96%
#     392: 31.3 мс, 148.7 FPS   запас 49%
#     448: 33.8 мс, 111.3 FPS   запас 11%
# У 448 точность на разведке выше на 1.33, но это внутри разброса между сидами 3.59 и одним
# сидом не доказано, а запас в 11% не переживёт полуторакратного замедления на чужом железе.
# Стенд жюри - A5000, нам он недоступен. Берём разрешение с запасом.
#
# torch.compile проверен и отвергнут: на чистом форварде даёт 37%, на боевом пути - ноль
# (195.9 против 195.7 FPS) и портит латентность (40.2 против 28.5 мс).
#
# ЧТО ПЕРЕСЧИТЫВАЕТСЯ ОБЯЗАТЕЛЬНО. Порог отказа и параметры ре-ранжирования калибровались на
# модели с входом 336. У модели с другим входом косинусы распределены иначе - у сида 448
# оптимальный порог вышел 0.596 против 0.572 у сида 336, у сида 392 - 0.554. Поэтому после супов идут заново:
# сетка ре-ранжирования, стресс отказа и аудит переноса порога. Брать старые числа нельзя.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue22 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

# Рецепт тот же, что у релиза на 336, изменён ровно один параметр - вход.
$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"
$COMMON = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 392 --epochs 18 " +
          "--P auto --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet " +
          "--distill $TEACHER --distill-w 20 --workers 4"

# 1. Двойник релиза: сид 0 уже обучен разведкой, добираем два и усредняем.
Step "q22_fit_s2" "python -m vreid.train $COMMON --no-select --seed 2 --out weights/fit18_r392_s2"
Step "q22_fit_s3" "python -m vreid.train $COMMON --no-select --seed 3 --out weights/fit18_r392_s3"
Step "q22_soup_fit" ("python scripts/model_soup.py weights/fit18_r392_s0/best.pt " +
    "weights/fit18_r392_s2/best.pt weights/fit18_r392_s3/best.pt --out weights/soup_r392_fit/best.pt")
Step "q22_val_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_r392_fit/best.pt --workers 4 --rerank"

# 2. Боевая модель: все личности, последняя эпоха, три сида.
Step "q22_all_s0" "python -m vreid.train $COMMON --val-frac 0 --seed 0 --out weights/all18_r392_s0"
Step "q22_all_s2" "python -m vreid.train $COMMON --val-frac 0 --seed 2 --out weights/all18_r392_s2"
Step "q22_all_s3" "python -m vreid.train $COMMON --val-frac 0 --seed 3 --out weights/all18_r392_s3"
Step "q22_soup_all" ("python scripts/model_soup.py weights/all18_r392_s0/best.pt " +
    "weights/all18_r392_s2/best.pt weights/all18_r392_s3/best.pt --out weights/soup_r392_all/best.pt")

# 3. Постобработка и порог - заново на НОВОМ двойнике, старые числа не переносятся.
Step "q22_grid" ("python scripts/rerank_grid.py --runs runs/hack/ft_soup_r392_fit " +
    "runs/hack/ft_fit18_r392_s0 --out results/rerank_grid_r392.json")
Step "q22_stress" "python scripts/refusal_stress.py --run runs/hack/ft_soup_r392_fit --out results/refusal_stress_r392.json"
Step "q22_cascade" ("python scripts/cascade_verify.py --light runs/hack/ft_soup_r392_fit " +
    "--heavy runs/hack/ft_l336cam_fit --topk 30 --out results/cascade_verify_r392.json")
Step "q22_compare" ("python scripts/compare_training.py --baseline weights/fit18_s0 weights/fit18_s2 " +
    "weights/fit18_s3 --candidate weights/fit18_r392_s0 weights/fit18_r392_s2 weights/fit18_r392_s3 " +
    "--out results/training_r392.json")

if ($global:failed.Count -eq 0) {
    Log "queue22 finished: все шаги прошли"
} else {
    Log "queue22 finished: ПРОВАЛИЛИСЬ $($global:failed.Count) шаг(ов): $($global:failed -join ', ')"
}
