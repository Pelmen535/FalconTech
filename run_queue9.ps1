# Queue 9: close every gap the architecture review found, then leave a shippable release.
#     powershell -ExecutionPolicy Bypass -File run_queue9.ps1
# ASCII only (PowerShell 5.1 reads .ps1 as cp1251; a Cyrillic byte breaks parsing).
#
# What the review found and what this queue proves:
#  1. submit.py now decides refusal by the cosine of the candidate that actually goes to the
#     file, sorts stably, and never writes a short row. The submission must be rebuilt.
#  2. The threshold 0.6011 was calibrated on soup_b336_val but shipped with soup_b336_all.
#     threshold_transfer.py measures both models on the OPEN test - out of sample for both,
#     no labels touched - and says whether the scale moved.
#  3. bench_full now follows the stated protocol: 300 timed runs after 50 warm-ups, CUDA sync
#     before AND after each run, >=10 s sustained per batch size, plus peak VRAM, weight load
#     time, total weight size and a two-run determinism check.
#  4. The Docker base is pinned by sha256 digest, not by a tag the owner may re-push.
#  5. The container is run twice (determinism) and once WITHOUT --shm-size, because the jury
#     is not obliged to pass it.
#  6. The partner's packager no longer swallows runs/ and weights/ (3.1 GB zip).

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs, handoff, handoff/reports | Out-Null
$status = "logs\queue_status.txt"
"queue9 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

# --- 1. Our own test suite on Windows (the device VM has no pytest, so it never ran here) ---
Step "tests" "python -m pytest tests -q || (pip install -q pytest && python -m pytest tests -q)"

# --- 1b. The release decodes JPEGs at a reduced scale (fast_decode) - that is where the 33.6 ms
#         latency comes from. The threshold, however, was calibrated on FULL-decode embeddings:
#         nobody had checked that the two pipelines give the same confidence scale. Validate
#         under production conditions and export a second candidate release from it. Separate
#         file names, so neither run overwrites the other's threshold. ---
Step "val_fast" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_b336_val/best.pt --workers 4 --fast-decode --rerank"
Step "export_fast" "python scripts/export_release.py --weights weights/soup_b336_all/best.pt --val results/hack_ft_soup_b336_val_fast_val.json --kr --k1 6 --k2 2 --out release_soup_fast"

# --- 2. Rebuild the release: same weights and threshold, but the recipe now records the JURY
#        metric (72.0) instead of the strict cross-camera one (73.9), plus split sizes. ---
Step "export_soup" "python scripts/export_release.py --weights weights/soup_b336_all/best.pt --val results/hack_ft_soup_b336_val_val.json --kr --k1 6 --k2 2 --out release_soup"
Step "copy_release" "xcopy /Y /E /I release_soup release"

# --- 3. THE measurement: does the threshold survive the move from the calibration model to
#        the released one? Both are out of sample on the open test, no labels are read. ---
Step "transfer" "python scripts/threshold_transfer.py --data %VREID_DATA% --cal weights/soup_b336_val/best.pt --rel weights/soup_b336_all/best.pt --release release_soup --val results/hack_ft_soup_b336_val_val.json --workers 8 --out results/threshold_transfer.json"

# --- 4. Submission rebuilt with the corrected submit.py, then proven query-independent ---
Step "predict_soup" "python -m vreid.predict --data %VREID_DATA% --release release_soup --out submission_soup --workers 8"
# Pre-flight on the form of all three files. The official evaluate.py is not in our hands,
# so this is the only thing between us and "the file did not parse".
Step "check_form" "python scripts/check_submission.py --submission submission_soup --data %VREID_DATA%"
Step "independence"  "python scripts/check_query_independence.py --submission submission_soup --data %VREID_DATA% --frac 0.6"
Step "independence2" "python scripts/check_query_independence.py --submission submission_soup --data %VREID_DATA% --frac 0.3 --seed 7"

# --- 5. Pin the base image by digest, then rebuild both images from the pinned base ---
Step "pull_base" "docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
Step "pin_base"  "python scripts/pin_base_image.py --dockerfile Dockerfile"
Step "build_release" "docker build -t vreid-release ."
Step "build_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime -t vreid-dev ."

# --- 6. Performance by the stated protocol, native and in the container ---
Step "bench_native"    "python -m vreid.bench_full --data %VREID_DATA% --release release_soup --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"
Step "bench_container" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# --- 7. Determinism: the same container command twice must give byte-identical artifacts ---
Step "docker_run_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_det_a:/out vreid-dev"
Step "docker_run_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_det_b:/out vreid-dev"
Step "cmp_det" "python scripts/compare_outputs.py out_det_a out_det_b"

# --- 8. The jury is not obliged to pass --shm-size. Without it /dev/shm is 64 MB and a
#        DataLoader with workers dies; safe_workers() is supposed to notice and drop to 0. ---
Step "docker_noshm" "docker run --rm --gpus all -v %VREID_DATA%:/data:ro -v %cd%\out_noshm:/out vreid-dev"
Step "cmp_noshm" "python scripts/compare_outputs.py out_det_a out_noshm"
Step "cmp_native" "python scripts/compare_outputs.py submission_soup out_det_a"
Step "check_form_docker" "python scripts/check_submission.py --submission out_det_a --data %VREID_DATA%"

# --- 9. Reports rebuilt on the jury rule ---
Step "abl_report" "python scripts/ablation_report.py --model ft_soup_b336_val"
# Why the partner's numbers are higher than ours: his gallery is 313 entries, the closed
# test has 750, ours 688. Same model, same data, only the gallery size varies - and the
# share of open-set queries is held at 20% so the comparison is about size and nothing else.
Step "gallery_size" "python scripts/gallery_size_effect.py --run runs/hack/ft_soup_b336_val --repeats 7"
Step "refusal_audit" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_val --k1 6 --k2 2"
Step "manifest" "python scripts/make_manifest.py --release release_soup --submission submission_soup --run runs/hack/ft_soup_b336_val --out handoff/MANIFEST.json"

Step "scorecard" "python scripts/scorecard.py --release release_soup --submission submission_soup"

# --- 10. The partner's bundle, now without the 2.2 GB of training scratch ---
Step "package" "cd competition && python scripts/package_handoff.py --out ..\handoff_competition_v2.zip"

Log "queue9 finished"
