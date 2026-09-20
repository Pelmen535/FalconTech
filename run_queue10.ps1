# Queue 10: close the three P0 defects the external audit found, re-issue the submission.
#     powershell -ExecutionPolicy Bypass -File run_queue10.ps1
# ASCII only.
#
# What changed in the code since queue 9:
#  R01 k-reciprocal sorted neighbours with an unstable sort. On exactly equal distances the
#      order was decided by introsort, which depends on the row layout - and that shifts when
#      other queries sit next to ours in the matrix. Real data has no exact ties, so the
#      666-query test never saw it. Now stable, masked pairs are pushed strictly past the
#      maximum real distance, and k1 is clamped by the GALLERY size, not by Q+G.
#  R02 the memory gate added in queue 9 turned re-ranking off entirely above a limit computed
#      from (Q+G) - so adding other queries changed the algorithm for ours. That is exactly the
#      dependence answer 38 forbids, and I introduced it. Replaced by chunking: memory now
#      bounds the chunk size, never the choice of algorithm, and the chunked result is provably
#      identical because query columns are excluded from every neighbourhood.
#  R15 NaN < threshold is false in IEEE, so a numeric failure was written out as a CONFIDENT
#      answer with confidence=nan, and check_submission accepted it. Both fixed.
#
# And one decision: the refusal threshold moves 0.6011 -> 0.55. All 970 matched validation
# queries have a same-camera frame of the same vehicle (cosine ~0.91); the closed test only
# guarantees a cross-camera positive. Without those duplicates the block scores 52.9 at 0.6011
# and 61.8 at 0.55, while in the likely case it drops only 97.2 -> 96.4. Details in
# results/refusal_stress.json.
#
# Verified before this queue, on the cached embeddings: the sort fix changes 0 of 11100 cells
# of the real submission. So this re-issue is expected to differ ONLY by the threshold.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue10 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

# 1. Tests, now including the regressions for R01/R02/R15
Step "tests10" "python -m pytest tests -q"
Step "ties" "python scripts/check_tie_independence.py"

# 2. The threshold decision, on the record
Step "stress" "python scripts/refusal_stress.py --run runs/hack/ft_soup_b336_val --release release_soup"

# 3. Re-issue the submission with the fixed reranker and the new threshold
Step "predict10" "python -m vreid.predict --data %VREID_DATA% --release release_soup --out submission_v2 --workers 8"
Step "form10" "python scripts/check_submission.py --submission submission_v2 --data %VREID_DATA%"
Step "indep10"  "python scripts/check_query_independence.py --submission submission_v2 --data %VREID_DATA% --frac 0.6"
Step "indep10b" "python scripts/check_query_independence.py --submission submission_v2 --data %VREID_DATA% --frac 0.3 --seed 7"
# ranking must be untouched by the threshold; only candidates.csv may differ from queue 9
Step "cmp_v2" "python scripts/compare_outputs.py submission_soup submission_v2"

# 4. The image carries release/, which now has the new recipe - rebuild and re-verify
Step "build10" "docker build -t vreid-release ."
Step "build10dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime -t vreid-dev ."
Step "docker10a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v2_a:/out vreid-dev"
Step "docker10b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v2_b:/out vreid-dev"
Step "cmp10_det" "python scripts/compare_outputs.py out_v2_a out_v2_b"
Step "docker10_noshm" "docker run --rm --gpus all -v %VREID_DATA%:/data:ro -v %cd%\out_v2_noshm:/out vreid-dev"
Step "cmp10_noshm" "python scripts/compare_outputs.py out_v2_a out_v2_noshm"
Step "form10_docker" "python scripts/check_submission.py --submission out_v2_a --data %VREID_DATA%"

# 5. Reports and the bundle
Step "scorecard10" "python scripts/scorecard.py --release release_soup --submission submission_v2"
Step "manifest10" "python scripts/make_manifest.py --release release_soup --submission submission_v2 --run runs/hack/ft_soup_b336_val --out handoff/MANIFEST.json"

Log "queue10 finished"
