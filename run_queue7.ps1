# Queue 7: produce the fully audited release the partner asked for.
#     powershell -ExecutionPolicy Bypass -File run_queue7.ps1
# ASCII only. Cyrillic data folder is passed to cmd through an environment variable.
#
# What changed and why this has to run:
#  * Query isolation was INCOMPLETE. Forbidding queries to be neighbours of each other is not
#    enough: k-reciprocal also builds neighbourhoods for gallery items, and those could take
#    queries as neighbours, so the normalisation of gallery rows depended on which queries were
#    present. The independence test caught it - only 156 of 666 queries kept their top-10 when
#    other queries were removed. Now the whole query column is masked and it is 666 of 666.
#    Every post-processing number therefore has to be measured again.
#  * The jury rule (remove only vehicle_id AND camera_id matches) is now the headline metric.
#
# Everything here ends in artifacts a third party can check: reports, hashes, split id lists.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs, handoff, handoff/reports | Out-Null
$status = "logs\queue_status.txt"
"queue7 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

$SOUPVAL = "ft:weights/soup_b336_val/best.pt"

# 1. Post-processing, measured again with real query isolation (embeddings are cached)
Step "iso_soup"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $SOUPVAL --workers 4 --reuse --rerank"
Step "iso_single" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_distill20/best.pt --workers 4 --reuse --rerank"

# 2. Plate ablation on the SOUP as well - one model was not enough to call the reliance zero
Step "abl_soup_det"  "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $SOUPVAL --workers 4 --mask-plate det"
Step "abl_soup_up"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $SOUPVAL --workers 4 --mask-plate up"
Step "abl_soup_rand" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $SOUPVAL --workers 4 --mask-plate rand"
Step "abl_soup_report" "python scripts/ablation_report.py --model ft_soup_b336_val"

# 3. Refusal: threshold calibrated on one half of the identities, measured on the other,
#    plus how often re-ranking moves the top-1 away from the one the threshold was set for
Step "refusal_audit" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_val --k1 6 --k2 2"

# 4. Release, submission, images
Step "export_soup" "python scripts/export_release.py --weights weights/soup_b336_all/best.pt --val results/hack_ft_soup_b336_val_val.json --kr --k1 6 --k2 2 --out release_soup"
Step "predict_soup" "python -m vreid.predict --data %VREID_DATA% --release release_soup --out submission_soup --workers 8"
Step "copy_release" "xcopy /Y /E /I release_soup release"
Step "build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime -t vreid-dev ."
Step "build_release" "docker build -t vreid-release ."

# 5. Proof that a query's ranking does not depend on the other queries - on the FINAL submission
Step "independence" "python scripts/check_query_independence.py --submission submission_soup --data %VREID_DATA% --frac 0.6"
Step "independence2" "python scripts/check_query_independence.py --submission submission_soup --data %VREID_DATA% --frac 0.3 --seed 7"

# 6. Performance on a larger sample, inside the container, with the machine idle
Step "bench_container" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 600 --workers 8 --fast-decode"

# 7. Container reproduces the native run, and the release image reproduces the dev image
Step "docker_full" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\submission_docker:/out vreid-dev"
Step "docker_vs_native" "python scripts/compare_outputs.py submission_soup submission_docker"
Step "mini_release_cpu" "docker run --rm -v %cd%\data\mini:/data:ro -v %cd%\out_mini_release2:/out vreid-release --device cpu --batch-size 4 --workers 2"
Step "mini_dev_gpu" "docker run --rm --gpus all --shm-size=2g -v %cd%\data\mini:/data:ro -v %cd%\out_mini_dev2:/out vreid-dev --batch-size 4 --workers 2"
Step "mini_compare" "python scripts/compare_outputs.py out_mini_release2 out_mini_dev2"

# 8. Manifest: hashes, split identity lists, environment, exact commands
Step "manifest" "python scripts/make_manifest.py --release release_soup --submission submission_soup --run runs/hack/ft_soup_b336_val --out handoff/MANIFEST.json"

Log "queue7 finished"
