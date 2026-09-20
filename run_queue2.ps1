# Fix-up queue, night 17->18.09. Run from E:\Hakaton in ONE PowerShell window:
#     powershell -ExecutionPolicy Bypass -File run_queue2.ps1
#
# ASCII only: PowerShell 5.1 reads .ps1 as cp1251 and a Cyrillic byte 0x94 breaks parsing.
# It waits for the running queue to finish, so you can start it right away.
#
# Why this exists: opencv is not installed in this python, so the plate detector silently
# returned nothing. The cache got built with 11416 empty entries and the whole ablation
# (det / up / rand) came out identical to base - that was missing opencv, not a result.
# The code now fails loudly instead of returning empty. Steps here redo only what was void.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue2 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8

function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }

# Run through cmd.exe, not Invoke-Expression. PowerShell 5.1 turns every stderr line of a
# native command into an ErrorRecord and can kill the script on a harmless warning - that is
# exactly what happened to abl2_det: the log got "NativeCommandError" with no text, and the
# real Python output was lost. cmd merges stderr into stdout BEFORE PowerShell sees it, so the
# log holds the actual traceback and the queue survives a failing step.
function Step($name, $cmd) {
    Log "START $name"
    $t0 = Get-Date
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log"
    $global:code = $LASTEXITCODE
    $dt = [int]((Get-Date) - $t0).TotalMinutes
    Log "END   $name  exit=$global:code  $dt min"
}

# 0. Wait for the first queue to finish (train_b336_seed2 is still running)
Log "WAIT  for running vreid processes"
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 60
}
Log "WAIT  done - GPU free"

$BEST = "ft:weights/hack_dinov2_b_336_distill20/best.pt"
$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"

# The data folder has a Cyrillic name, and a literal Cyrillic byte in a .ps1 breaks the
# PowerShell 5.1 parser (cp1251 reads 0x94 as a quote). So find it by content instead.
$DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } |
         Select-Object -First 1).FullName
if (-not $DATA) { Log "ABORT data folder with test_query.csv not found"; exit 1 }
Log "data folder: $DATA"

# 1. The cache was already rebuilt from the other machine (11416 entries, zone found on 94.0%),
#    and masking only reads that JSON - opencv is needed to BUILD it, not to use it. So this
#    only checks the cache is non-empty and rebuilds it if something is off.
Step "check_cache" "python scripts/check_plate_cache.py"
if ($global:code -ne 0) {
    Log "cache empty or missing - installing opencv and rebuilding"
    Step "pip_opencv" "python -m pip install --quiet opencv-python-headless"
    Step "plate_cache2" "python scripts/build_plate_cache.py --config configs/hackathon.yaml --workers 8"
    Step "check_cache2" "python scripts/check_plate_cache.py"
    if ($global:code -ne 0) { Log "ABORT no plate boxes - ablation would be void again"; exit 1 }
}

# 3. Redo the ablation that was void
Step "abl2_det"  "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --mask-plate det"
Step "abl2_up"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --mask-plate up"
Step "abl2_rand" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --mask-plate rand"

# 4. Report. Exit code 1 means the model really leans on the plate zone.
Step "abl2_report" "python scripts/ablation_report.py --model ft_hack_dinov2_b_336_distill20"
$leans = ($global:code -eq 1)
Log "ablation verdict: leans_on_plate=$leans"

# 5. Train with the plate hidden ONLY if the ablation says it matters (~92 min)
if ($leans) {
    Step "train_b336_maskp2" "python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b --img-size 336 --epochs 20 --P 11 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --distill $TEACHER --distill-w 20 --mask-p 0.5 --workers 4 --out weights/hack_dinov2_b_336_maskp2"
    Step "val_b336_maskp2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_maskp2/best.pt --workers 4 --rerank"
    Step "val_b336_maskp2_det" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_maskp2/best.pt --workers 4 --mask-plate det"
} else {
    Log "SKIP  train_b336_maskp2 - no plate reliance to fix"
}

# 6. Release from the all-ids model. Recipe and refusal point are frozen from the validated
#    distill20 run; the threshold travels as a refusal RATE (quantile), which is what transfers
#    between sets.
#    Postprocessing is k-recip 10/3, NOT 15/4. Three runs of the same recipe gave base mAP
#    73.8 / 73.9 / 74.0, but 15/4 gave 77.1 / 75.9 / 76.8 (spread 1.2) while 10/3 gave
#    76.8 / 76.6 / 77.5 (mean 77.0). 15/4 only looked best because it was picked as the max
#    over 13 configurations on a single run - that selection inflates by about 0.6 points.
Step "export_all" "python scripts/export_release.py --weights weights/hack_dinov2_b_336_all/best.pt --val results/hack_ft_hack_dinov2_b_336_distill20_val.json --kr --k1 10 --k2 3 --out release_all"
Step "predict_all" "python -m vreid.predict --data $DATA --release release_all --out submission_all --workers 4"
Step "bench_all" "python -m vreid.bench_full --data $DATA --release release_all --n 200 --workers 8 --fast-decode"

Log "queue2 finished"
