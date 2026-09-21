# Queue 17: попытка добрать точность обучением. Запускать, когда машина СВОБОДНА.
#     powershell -ExecutionPolicy Bypass -File journal\run_queue17_train.ps1
# ASCII only in code. Около 8 часов, из них 5 - обучение релиза.
#
# ЗАЧЕМ. Точность - единственный блок, где мы далеко от потолка: 33.4 из 45. Рычаг один -
# лучше обучить ту же архитектуру, потому что более тяжёлая модель отнимает баллы за
# производительность (учитель ViT-L даёт +4 mAP, но втрое дороже на форварде: это минус
# около 14 баллов производительности против плюс 2.9 за точность).
#
# ЧТО МЕНЯЕМ и почему именно это:
#
# 1. epochs 18 -> 20. Кривая валидации у всех трёх сидов двойника (fit18_s0/s2/s3) РАСТЁТ
#    до последней эпохи: у s0 это 70.82 -> 71.00 -> 70.99 -> 71.39 -> 71.21 -> 71.63.
#    Мы останавливаемся не на плато, а на подъёме. На соседнем сплите 20-эпошные прогоны
#    выходят на полку к 17-20 эпохе, поэтому 20 - консервативная оценка, а не наугад.
#
# 2. arc-ls 0.0 -> 0.1 (label smoothing в ArcFace). Единственное прямое измерение этого
#    флага в проекте: hack_dinov2_l_336 (0.7419) против hack_dinov2_l_336_ls (0.7623) -
#    те же 20 эпох, тот же сид, та же конфигурация, отличается только arc_ls. +2.0 пункта.
#    Взаимодействие с дистилляцией и cam-aware сэмплером не измерялось - это и проверяем.
#
# Всё остальное - дословно рецепт двойника релиза (fit18_s0): dinov2_b, вход 336, P=11,
# K=4, lr 2e-5, дистилляция от hack_dinov2_l_336_cam с весом 20, camera-aware сэмплер,
# кросс-камерный triplet, --no-select.
#
# КАК СРАВНИВАТЬ. Число, которое печатает train.py в колонке mAP, - это УЖЕ метрика жюри
# (mAP@10, junk-фильтр по vid+cam, см. validate() в vreid/train.py). Сравнивать можно
# напрямую, но ТОЛЬКО внутри одного сплита: у fit18_* эпоха 0 даёт 7.69, у более старых
# прогонов 9.94 - это разные разбиения, и их числа несопоставимы. Новый прогон обязан
# показать на эпохе 0 те же 7.7.
#
#     baseline (fit18, 18 эпох, arc_ls 0): s0 71.63, s2 71.24, s3 68.73, среднее 70.53
#     суп трёх сидов на этом же сплите: 70.66 без постобработки
#
# РЕШАЮЩЕЕ ПРАВИЛО. Разброс между сидами одного рецепта - 2.9 пункта. Одного прогона
# недостаточно, чтобы менять релиз: нужен средний по трём сидам выше 70.53 хотя бы на
# 1.5 пункта. Поэтому очередь сначала считает ТРИ сида на fit, и только если среднее
# выросло - переобучает релиз на всех id. Если не выросло, релиз остаётся прежним, а
# результат всё равно полезен: это измеренный ответ, а не гипотеза.
#
# ПАМЯТЬ. Прогон занимает ~15.6 ГиБ VRAM и несколько ГиБ RAM. На занятой машине он падает:
# 21.09 три попытки подряд умерли на "Couldn't open shared file mapping" и
# "Unable to allocate 331 KiB" при запущенном UnrealEditor (19 ГиБ commit) и двух VM
# Docker. Поэтому: закрыть тяжёлые приложения, остановить docker compose, и только потом
# запускать. workers 2 быстрее, workers 0 надёжнее - выбор в переменной ниже.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"

# 2 - быстрее (воркеры перекрывают загрузку с counting), 0 - без разделяемой памяти,
# переживает нехватку commit на занятой машине.
$WORKERS = 2
$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"
$EPOCHS = 20

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue17 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }
function Step($name, $cmd) {
    if (Test-Path "logs\$name.done") { Log "SKIP  $name (уже сделан)"; return }
    Log "START $name"; $t0 = Get-Date
    $global:LASTEXITCODE = 0
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log"
    $code = $LASTEXITCODE
    $log = "logs\$name.log"
    $empty = -not (Test-Path $log) -or ((Get-Item $log).Length -eq 0)
    if ($code -eq 0 -and -not $empty) { New-Item -ItemType File -Force -Path "logs\$name.done" | Out-Null }
    $verdict = if ($code -ne 0) { "FAILED exit=$code" } elseif ($empty) { "FAILED no output" } else { "ok" }
    Log "END   $name  $verdict  $([int]((Get-Date) - $t0).TotalMinutes) min"
}

# Свободная машина - это условие корректности замера, а не пожелание: обучение падает
# при нехватке commit, а замеры времени рядом с UnrealEditor ничего не значат.
docker compose down 2>&1 | Out-Null
Log "docker compose остановлен"
$busy = Get-Process UnrealEditor -ErrorAction SilentlyContinue
if ($busy) { Log "ВНИМАНИЕ: запущен UnrealEditor - обучение, скорее всего, упадёт по памяти" }

# --- 1. Три сида нового рецепта на fit-сплите: это честный двойник, его и сравниваем ---
foreach ($seed in 0, 2, 3) {
    Step "train_ls20_s$seed" ("python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b " +
        "--img-size 336 --epochs $EPOCHS --P 11 --K 4 --arc-ls 0.1 --no-select " +
        "--distill $TEACHER --distill-w 20 --cam-aware --cross-cam-triplet " +
        "--seed $seed --workers $WORKERS --out weights/fit_ls20_s$seed")
}

# --- 2. Сводка: выросло ли среднее по трём сидам ---
Step "compare17" "python scripts/compare_training.py --baseline weights/fit18_s0 weights/fit18_s2 weights/fit18_s3 --candidate weights/fit_ls20_s0 weights/fit_ls20_s2 weights/fit_ls20_s3 --out results/training_upgrade.json"

# Дальше - только если средний прирост >= 1.5 пункта. Скрипт возвращает 0, если да.
if ($global:LASTEXITCODE -ne 0) {
    Log "queue17: прирост не подтверждён, релиз остаётся прежним. Смотри results/training_upgrade.json"
    exit 0
}

# --- 3. Суп двойника: сравнение с 70.66 на той же валидации ---
Step "soup17_fit" "python scripts/model_soup.py weights/fit_ls20_s0/best.pt weights/fit_ls20_s2/best.pt weights/fit_ls20_s3/best.pt --out weights/soup_ls20_fit/best.pt"
Step "val17_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_ls20_fit/best.pt --workers 4 --rerank"

# --- 4. Релиз: тот же рецепт на ВСЕХ id, три сида, суп ---
foreach ($seed in 0, 2, 3) {
    Step "train_all_ls20_s$seed" ("python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b " +
        "--img-size 336 --epochs $EPOCHS --P 11 --K 4 --arc-ls 0.1 --val-frac 0 " +
        "--distill $TEACHER --distill-w 20 --cam-aware --cross-cam-triplet " +
        "--seed $seed --workers $WORKERS --out weights/all_ls20_s$seed")
}
Step "soup17_all" "python scripts/model_soup.py weights/all_ls20_s0/best.pt weights/all_ls20_s2/best.pt weights/all_ls20_s3/best.pt --out weights/soup_ls20_all/best.pt"

Log "queue17 finished: веса собраны. Релиз НЕ заменён автоматически - это делает"
Log "         scripts/export_release.py после того, как числа проверены глазами."
