# Queue 12: merged partner patch + the soup question, re-issued end to end.
#     powershell -ExecutionPolicy Bypass -File run_queue12.ps1
# ASCII only. Waits for queue 11's trainings to finish, then runs ~40 minutes.
#
# WHAT WAS MERGED from the partner's 19b patch (his code, our comments kept):
#   metrics.py   stable argsort inside the jury metric itself - we had fixed every tie-break
#                except the one in the metric.
#   predict.py   fail-fast recipe guards (dba, refuse_rate, confidence, topk, mask_plate),
#                f_bavail instead of f_blocks for /dev/shm, sha256 of model/recipe/input CSVs
#                into run_info, fixed-shape GEMV so the cosine matrix cannot depend on Q.
#   rerank.py    gallery-gallery block computed once at fixed shape, sorted() over the
#                expanded neighbour set, chunk-local masks built from keys.
#   submit.py    embeddings validated finite and non-degenerate; no duplicate padding.
#   plus eval_common.py, artifacts.py, paired.py and his rewritten refusal_audit and
#   gallery_size_effect, three new test files, .dockerignore, docs/WORKFLOW.md.
#
# WHAT WAS NOT TAKEN, and why:
#   rerank_chunk_size raised MemoryError with "no cosine fallback". A crash on the jury stand
#   is zero for every block, including the ones already earned. The fallback he removed is
#   keyed on GALLERY size alone, which is identical for every query, so it is not a dependence
#   on other queries. Restored, and the mode actually used is recorded in run_info.
#   The hard bbox raise is kept for the competition path only (strict=True); on training a
#   single bad row would throw away hours of GPU time.
#   typing-extensions pinned to 4.16.0, not 4.12.1: his own pip check fails on 4.12.1
#   (anyio 4.15.1 requires >=4.16.0). That defect was real and was in our image.
#
# Verified before this queue, on cached embeddings: the merged code reproduces submission_v2
# exactly - 0 of 11100 cells.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue12 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }
function Step($name, $cmd) {
    Log "START $name"; $t0 = Get-Date
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log"
    $global:code = $LASTEXITCODE
    Log "END   $name  exit=$global:code  $([int]((Get-Date) - $t0).TotalMinutes) min"
}
Log "WAIT  for queue 11 trainings"
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 60
}
Log "WAIT  done - GPU free"
$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"
$REC = "release_soup/recipe.json"

# 1. Tests, including his three new files and our tie regressions
Step "tests12" "python -m pytest tests -q"
Step "ties12" "python scripts/check_tie_independence.py"

# 2. THE SOUP QUESTION. Queue 11 lost this to my own bug: --out took a directory but the
#    script wrote a file without the .pt suffix, so the soup was computed and then not found.
#    Same recipe, two seeds: does averaging land at or above the mean of its members (72.89)?
Step "soup2" "python scripts/model_soup.py weights/hack_dinov2_b_336_distill20/best.pt weights/hack_dinov2_b_336_seed2/best.pt --out weights/soup2_same_recipe/best.pt"
Step "val_soup2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup2_same_recipe/best.pt --workers 4 --rerank"

# 3. The honest twin of the release: three fit-split runs with exactly the release recipe.
Step "soup_fit" "python scripts/model_soup.py weights/fit18_s0/best.pt weights/fit18_s2/best.pt weights/fit18_s3/best.pt --out weights/soup_b336_fit/best.pt"
Step "val_fit_s0"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_s0/best.pt --workers 4 --rerank"
Step "val_fit_s2"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_s2/best.pt --workers 4 --rerank"
Step "val_fit_s3"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_s3/best.pt --workers 4 --rerank"
Step "val_soup_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_b336_fit/best.pt --workers 4 --rerank"

# 4. Refusal on the twin, by his stricter procedure: grid from A only, frozen .55 scored on the
#    emitted CSV, second half labelled post-selection rather than pretend-holdout.
Step "audit_fit"  "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe %REC% --out results/refusal_audit_fit.json"
Step "audit_val"  "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_val --recipe %REC% --out results/refusal_audit_val.json"
Step "stress_fit" "python scripts/refusal_stress.py --run runs/hack/ft_soup_b336_fit --release release_soup --out results/refusal_stress_fit.json"
# Distractor-only design. On our track protocol only 5 of 688 gallery rows are not a positive
# for some query, so this is expected to report not_estimable - and that is the point: it is
# why the earlier -7.3 points figure is withdrawn rather than defended.
Step "gallery_fit" "python scripts/gallery_size_effect.py --run runs/hack/ft_soup_b336_val --recipe %REC% --sizes 683 685 688 --out results/gallery_size_effect_fixed.json"

# 5. Image rebuilt from the merged Dockerfile; pip check must now pass inside the build
Step "build12" "docker build -t vreid-release ."
Step "build12dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime -t vreid-dev ."
Step "freeze12" "docker run --rm --entrypoint python vreid-release -c \"print(open('/app/installed-packages.txt').read())\""

# 6. Submission re-issued and re-proven
Step "predict12" "python -m vreid.predict --data %VREID_DATA% --release release_soup --out submission_v3 --workers 8"
Step "cmp_v3" "python scripts/compare_outputs.py submission_v2 submission_v3"
Step "form_v3" "python scripts/check_submission.py --submission submission_v3 --data %VREID_DATA%"
Step "indep_v3" "python scripts/check_query_independence.py --submission submission_v3 --data %VREID_DATA% --frac 0.6"
Step "docker_v3a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v3_a:/out vreid-dev"
Step "docker_v3b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v3_b:/out vreid-dev"
Step "cmp_v3_det" "python scripts/compare_outputs.py out_v3_a out_v3_b"
Step "docker_v3_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v3_offline:/out vreid-dev"
Step "cmp_v3_offline" "python scripts/compare_outputs.py out_v3_a out_v3_offline"
Step "docker_v3_noshm" "docker run --rm --gpus all -v %VREID_DATA%:/data:ro -v %cd%\out_v3_noshm:/out vreid-dev"
Step "cmp_v3_noshm" "python scripts/compare_outputs.py out_v3_a out_v3_noshm"

# 7. His own verification tools, run against our output
Step "replay" "python scripts/check_cached_replay.py --submission submission_v3 --data %VREID_DATA% --release release_soup --out results/cached_replay.json"

# 8. Reports and manifest
Step "scorecard12" "python scripts/scorecard.py --release release_soup --submission submission_v3"
Step "manifest12" "python scripts/make_manifest.py --release release_soup --submission submission_v3 --run runs/hack/ft_soup_b336_fit --out handoff/MANIFEST.json"

Log "queue12 finished"
