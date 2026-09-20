# Queue 5: final assembly from the souped model. Run from E:\Hakaton in ONE window:
#     powershell -ExecutionPolicy Bypass -File run_queue5.ps1
# ASCII only. The data folder has a Cyrillic name and is handed to cmd through an env variable.
#
# The threshold is taken from the SOUP's own validation, not from distill20's: souping shifts
# the confidence scale (its operating point is 0.6011 against 0.5605), and a threshold from a
# different model would not transfer. Post-processing is k-recip 6/2, which won on the
# query-isolated measurement (77.7 for the soup against 76.9 for the best single model).

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs, submission_soup, out_mini_release2, out_mini_dev2 | Out-Null
$status = "logs\queue_status.txt"
"queue5 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"

$SOUP = "weights/soup_b336_all/best.pt"
$SOUPVAL = "results/hack_ft_soup_b336_val_val.json"

# 1. Release from the soup, and the submission
Step "export_soup" "python scripts/export_release.py --weights $SOUP --val $SOUPVAL --kr --k1 6 --k2 2 --out release_soup"
Step "predict_soup" "python -m vreid.predict --data %VREID_DATA% --release release_soup --out submission_soup --workers 8"

# 2. How much did the submission actually move relative to the single all-ids model
Step "diff_vs_single" "python scripts/compare_outputs.py submission_soup submission_all"

# 3. Images from the new release: dev for this card, release for the stand
Step "copy_release" "xcopy /Y /E /I release_soup release"
Step "build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime -t vreid-dev ."
Step "build_release" "docker build -t vreid-release ."

# 4. Honest performance number: real test, 200 vehicles, inside the container
Step "bench_container" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 200 --workers 8 --fast-decode"

# 5. Prove the release image still reproduces the dev image after the model swap
Step "mini_release_cpu" "docker run --rm -v %cd%\data\mini:/data:ro -v %cd%\out_mini_release2:/out vreid-release --device cpu --batch-size 4 --workers 2"
Step "mini_dev_gpu" "docker run --rm --gpus all --shm-size=2g -v %cd%\data\mini:/data:ro -v %cd%\out_mini_dev2:/out vreid-dev --batch-size 4 --workers 2"
Step "mini_compare" "python scripts/compare_outputs.py out_mini_release2 out_mini_dev2"

# 6. The full run in the container, which is what the jury executes
Step "docker_full" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\submission_docker:/out vreid-dev"
Step "docker_vs_native" "python scripts/compare_outputs.py submission_soup submission_docker"

Log "queue5 finished"
