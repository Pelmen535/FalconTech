# Queue 8: run the partner's control experiment on HIS protocol, then package everything.
#     powershell -ExecutionPolicy Bypass -File run_queue8.ps1
# ASCII only. Runs inside competition\ - a working copy of his 1809 handoff with our extra
# scripts added and dataset.root pointed at the real data folder.
#
# Why his tree and not ours: his train.py refuses distillation and full-fit training whenever
# an independent protocol is configured, and his k_reciprocal processes one query at a time,
# so independence holds by construction rather than by a patch. That is the point of the run:
# an auditable baseline with provenance, not our best model. Expect it to score BELOW our
# current release - it has neither distillation nor training on all identities.
#
# Our own tree stays untouched; the submission still comes from release_soup.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue8 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
# The data folder has a Cyrillic name; hand it to cmd through an environment variable so that
# no non-ASCII byte appears in this file (PowerShell 5.1 reads .ps1 as cp1251 and breaks).
$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"

# 1. His tests first: 15 competition tests plus the older suite. If they fail, nothing below counts.
Step "his_tests" "cd competition && python -m unittest discover -s tests -v"

# 2. The control training he asked for. No distillation, no full-fit - his code forbids both
#    while an independent protocol is configured. About 90 minutes.
Step "baseline_seed0" "cd competition && python -m vreid.train --config configs/independent.yaml --backbone dinov2_b --img-size 336 --epochs 20 --P auto --K 4 --cam-aware --cross-cam-triplet --seed 0 --out weights/baseline_seed0"

# 3. Assess on SELECTION only. Holdout stays untouched until calibration.
Step "assess_selection" "cd competition && python scripts/assess_checkpoint.py --checkpoint weights/baseline_seed0/best.pt --protocol protocol --data %VREID_DATA% --recipe release/recipe.json --out results/selection_baseline --workers 4"

# 4. Calibrate the threshold on CALIBRATION, then score HOLDOUT once, and export.
Step "calibrate_export" "cd competition && python scripts/export_release.py --checkpoint weights/baseline_seed0/best.pt --protocol protocol --data %VREID_DATA% --recipe release/recipe.json --out release_calibrated_v1 --batch-size 32 --workers 4"

# 5. Our independence check against his per-query reranking, on his own output
Step "independence_his" "cd competition && python scripts/check_query_independence.py --submission release_calibrated_v1 --data %VREID_DATA% --frac 0.6"

# 6. Bundle with checksums, by his own packager
Step "package" "cd competition && python scripts/package_handoff.py --out ..\handoff_competition_2026-09-18.zip"

Log "queue8 finished"
