# Queue 4, 18.09: verify the release image, re-measure performance on an idle machine,
# then model soup. Run from E:\Hakaton in ONE PowerShell window:
#     powershell -ExecutionPolicy Bypass -File run_queue4.ps1
# ASCII only (PowerShell 5.1 reads .ps1 as cp1251). Paths with Cyrillic are expanded by cmd
# itself through %cd%, so no non-ASCII byte ever appears in this file.
#
# Order is deliberate:
#  1-2. The release image (CUDA 12.1, what the jury will run) has never been executed - only
#       built. It cannot use this Blackwell card, so it runs on CPU over a 40-crop mini set,
#       and the dev image runs the same set on GPU. Then the two outputs are compared. This is
#       the last unverified thing in the whole submission.
#  3.   Performance re-measure: the last container run reported 9.2 ms of network per crop
#       against 5.5 earlier, right after a long image build. An idle machine gives the honest
#       number, and that number is what the 20% criterion is scored on.
#  4.   Soup of the three existing runs on the 80% split - the CHECK that souping helps at all.
#  5-6. Two more all-ids runs and the soup of all three: the release candidate.
#
# Why soup and not an ensemble: runs of one recipe start from the same pretrained weights and
# differ only in data order, so averaging the weights behaves like an ensemble while staying ONE
# model. An ensemble doubles network time and costs about 2.4 of the 20 performance points.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs, out_mini_release, out_mini_dev | Out-Null
$status = "logs\queue_status.txt"
"queue4 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }
function Step($name, $cmd) {
    Log "START $name"; $t0 = Get-Date
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log"
    $global:code = $LASTEXITCODE
    Log "END   $name  exit=$global:code  $([int]((Get-Date) - $t0).TotalMinutes) min"
}
Log "WAIT  for running vreid processes"
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 60
}
Log "WAIT  done - GPU free"

# 1. Release image on CPU over the mini set (the only way to run it on this machine)
Step "mini_release_cpu" "docker run --rm -v %cd%\data\mini:/data:ro -v %cd%\out_mini_release:/out vreid-release --device cpu --batch-size 4 --workers 2"

# 2. Dev image on GPU over the same mini set, then compare
Step "mini_dev_gpu" "docker run --rm --gpus all --shm-size=2g -v %cd%\data\mini:/data:ro -v %cd%\out_mini_dev:/out vreid-dev --batch-size 4 --workers 2"
Step "mini_compare" "python scripts/compare_outputs.py out_mini_release out_mini_dev"
if ($global:code -ne 0) { Log "WARN  release and dev images disagree - see logs\mini_compare.log" }

# 3. Honest performance number on an idle machine, measured inside the container
# FIXED: this used to measure over the 40-crop mini set. With 8 workers and a full warmup
# pass there is nothing to measure in 40 items - the best FPS came out at batch=1, i.e. the
# batched mode never spun up, and the score read 14.9 instead of 20. Measure on the real test,
# at least 200 vehicles. The data folder has a Cyrillic name, so it is handed to cmd through an
# environment variable: no non-ASCII byte may appear in this file.
$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Step "bench_container" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 200 --workers 8 --fast-decode"

$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"
$RECIPE = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 336 --P 11 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --distill $TEACHER --distill-w 20 --workers 4"

# 4. Does souping help at all? Three existing runs on the 80% split.
Step "soup3" "python scripts/model_soup.py --out weights/soup_b336_val/best.pt weights/hack_dinov2_b_336_distill20/best.pt weights/hack_dinov2_b_336_maskp/best.pt weights/hack_dinov2_b_336_seed2/best.pt"
Step "val_soup3" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_b336_val/best.pt --workers 4 --rerank"

# 5. Two more all-ids runs with other seeds (~80 min each)
Step "train_all_s2" "python -m vreid.train $RECIPE --epochs 18 --val-frac 0 --seed 2 --out weights/hack_dinov2_b_336_all_s2"
Step "train_all_s3" "python -m vreid.train $RECIPE --epochs 18 --val-frac 0 --seed 3 --out weights/hack_dinov2_b_336_all_s3"

# 6. Release candidate: soup of the three all-ids runs
Step "soup_all" "python scripts/model_soup.py --out weights/soup_b336_all/best.pt weights/hack_dinov2_b_336_all/best.pt weights/hack_dinov2_b_336_all_s2/best.pt weights/hack_dinov2_b_336_all_s3/best.pt"

Log "queue4 finished"
