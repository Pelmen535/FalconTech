# Queue 3, 18.09: bring the release in line with the organizers' answers.
# Run from E:\Hakaton in ONE PowerShell window:
#     powershell -ExecutionPolicy Bypass -File run_queue3.ps1
# ASCII only (PowerShell 5.1 reads .ps1 as cp1251).
#
# What changed and why:
#  * answer 14 - clustering over the whole test_query is forbidden, k-reciprocal is allowed only
#    within one query's top-K. Our k-reciprocal built neighbourhoods over [queries; gallery], so a
#    query's ranking depended on other queries. Queries are now isolated from each other, which
#    changes the post-processing numbers - step 1 re-measures them.
#  * answer 6 - 20% of test queries have no pair. Our own estimate said ~3%, but it compared the
#    all-ids model on test against the 80%-model on val, so it was biased. The organizers' figure
#    is primary evidence, so the refusal point goes back to transferring the RATE (step 2).
#  * step 4 re-measures the distributions with ONE model on both sets, which settles 3% vs 20%.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue3 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }
function Step($name, $cmd) {
    Log "START $name"; $t0 = Get-Date
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log"
    $global:code = $LASTEXITCODE
    Log "END   $name  exit=$global:code  $([int]((Get-Date) - $t0).TotalMinutes) min"
}
$DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } |
         Select-Object -First 1).FullName
Log "data folder: $DATA"

$D20 = "ft:weights/hack_dinov2_b_336_distill20/best.pt"

# 1. Post-processing numbers with query-isolated k-reciprocal (compliant version)
Step "val_iso" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone $D20 --workers 4 --reuse --rerank"

# 2. Release: rate-based refusal (20% open-set per answer 6)
Step "export_rate" "python scripts/export_release.py --weights weights/hack_dinov2_b_336_all/best.pt --val results/hack_ft_hack_dinov2_b_336_distill20_val.json --kr --k1 10 --k2 3 --rate-threshold --out release_all"
Step "predict_rate" "python -m vreid.predict --data $DATA --release release_all --out submission_all --workers 8"

# 3. Same model on both sets, so the open-set share can be checked without bias
Step "export_d20" "python scripts/export_release.py --weights weights/hack_dinov2_b_336_distill20/best.pt --val results/hack_ft_hack_dinov2_b_336_distill20_val.json --kr --k1 10 --k2 3 --rate-threshold --out release_d20"
Step "predict_d20" "python -m vreid.predict --data $DATA --release release_d20 --out submission_d20 --workers 8"

Log "queue3 finished"
