# Night queue 17->18.09. Run from E:\Hakaton in ONE PowerShell window:
#     powershell -ExecutionPolicy Bypass -File run_queue.ps1
#
# ASCII only on purpose: PowerShell 5.1 reads .ps1 as cp1251 and a Cyrillic byte 0x94
# is treated as a string delimiter, which breaks parsing.
#
# Each step writes logs\<name>.log; a failing step does not stop the rest.
# Progress goes to logs\queue_status.txt (Claude reads it on schedule).
#
# What this night answers:
#   1. plate cache - find the anonymization zone on every crop, once (~50 ms/crop)
#   2. ablation    - does the model lean on the plate zone? det vs up/rand controls of
#                    EQUAL AREA elsewhere, plus band = the old crude rectangle
#   3. mask-p 0.5  - train with the plate zone hidden half the time; if quality holds,
#                    we can claim plate-independence by construction, not by measurement
#   4. final       - winning recipe on ALL ids (--val-frac 0), 18 epochs = the best epoch
#   5. seed 2      - same recipe, other seed: tells us how big a difference is real

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"          # set "0.18" if you will be using the computer
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Encoding utf8

function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }

function Step($name, $cmd) {
    Log "START $name"
    $t0 = Get-Date
    Invoke-Expression "$cmd 2>&1 | Tee-Object -FilePath logs\$name.log"
    $code = $LASTEXITCODE
    $dt = [int]((Get-Date) - $t0).TotalMinutes
    Log "END   $name  exit=$code  $dt min"
}

# 0. Wait for any running vreid process (do not fight for the GPU)
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

# 1. Plate-zone cache over train + test (~2 min on 8 workers)
Step "plate_cache" "python scripts/build_plate_cache.py --config configs/hackathon.yaml --workers 8"

# 2. Ablation. Base number comes from the cached embeddings, the rest re-extract.
Step "abl_base" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --reuse --rerank"
Step "abl_det"  "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --mask-plate det"
Step "abl_up"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --mask-plate up"
Step "abl_rand" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --mask-plate rand"
Step "abl_band" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $BEST --workers 4 --mask-plate band"

# 3. Train with the plate zone hidden half the time (same winning recipe otherwise)
Step "train_b336_maskp" "python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b --img-size 336 --epochs 20 --P 11 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --distill $TEACHER --distill-w 20 --mask-p 0.5 --workers 4 --out weights/hack_dinov2_b_336_maskp"
Step "val_b336_maskp" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_maskp/best.pt --workers 4 --rerank"
Step "val_b336_maskp_det" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_maskp/best.pt --workers 4 --mask-plate det"

# 4. Final model on ALL ids, no validation split, stop at the epoch that was best (18)
Step "train_b336_all" "python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b --img-size 336 --epochs 18 --P 11 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --distill $TEACHER --distill-w 20 --val-frac 0 --workers 4 --out weights/hack_dinov2_b_336_all"

# 5. Same recipe, seed 2: run-to-run spread, so we know which gaps are real
Step "train_b336_seed2" "python -m vreid.train --config configs/hackathon.yaml --backbone dinov2_b --img-size 336 --epochs 20 --P 11 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --distill $TEACHER --distill-w 20 --seed 2 --workers 4 --out weights/hack_dinov2_b_336_seed2"
Step "val_b336_seed2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_seed2/best.pt --workers 4 --rerank"

Log "queue finished"
