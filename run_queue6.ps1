# Queue 6: re-measure everything under the JURY rule. Run from E:\Hakaton in ONE window:
#     powershell -ExecutionPolicy Bypass -File run_queue6.ps1
# ASCII only.
#
# Why: our headline metric removed ALL gallery items from the query's camera, including OTHER
# vehicles. The organizers (answers 5 and 38) remove only pairs where vehicle_id AND camera_id
# both match. The strict mode drops the hardest negatives - other cars from the same viewpoint,
# same light - and inflates mAP by 1 to 3 points. Fixed in metrics/hack_cli/train.
#
# Consequences to re-measure: the post-processing choice (k-recip 6/2) was picked on the
# inflated metric, and so was the refusal operating point. Embeddings are cached, so --reuse
# makes each run about two minutes - no re-extraction.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue6 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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

# 1. Re-measure the three candidates under the jury rule (cached embeddings, --reuse)
Step "jury_soup"    "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_b336_val/best.pt --workers 4 --reuse --rerank"
Step "jury_single"  "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_distill20/best.pt --workers 4 --reuse --rerank"
Step "jury_teacher" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_l_336_cam/best.pt --workers 4 --reuse --rerank"

# 2. Re-measure the plate ablation under the jury rule as well (masked embeddings are cached too)
Step "jury_abl_det"  "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_distill20/best.pt --workers 4 --reuse --mask-plate det"
Step "jury_abl_up"   "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_distill20/best.pt --workers 4 --reuse --mask-plate up"
Step "jury_abl_rand" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/hack_dinov2_b_336_distill20/best.pt --workers 4 --reuse --mask-plate rand"
Step "jury_abl_report" "python scripts/ablation_report.py --model ft_hack_dinov2_b_336_distill20"

Log "queue6 finished"
