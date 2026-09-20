# Queue 11: answer the one question the whole release rests on, and close the provenance gaps.
#     powershell -ExecutionPolicy Bypass -File run_queue11.ps1
# ASCII only. Part 1 is ~20 minutes, part 2 runs about four hours - start it before bed.
#
# THE QUESTION. Every validation number we publish (mAP 72.0 / 74.9, the refusal threshold, the
# plate ablation, the gallery-size study) was measured on weights/soup_b336_val. The release is
# weights/soup_b336_all. These two are NOT built the same way:
#
#     soup_b336_val  = distill20 (seed 0, 20 epochs, best epoch picked on val)
#                    + maskp     (seed 0, 20 epochs, TRAINED WITH THE PLATE MASKED)
#                    + seed2     (seed 2, 20 epochs, best epoch picked on val)
#     soup_b336_all  = all (seed 0) + all_s2 (seed 2) + all_s3 (seed 3),
#                      all 18 epochs, --val-frac 0, no epoch selection at all
#
# Three differences at once: mixed recipes vs one recipe, epoch selection vs none, 20 vs 18
# epochs. So the proxy is not the release. And the proxy looks bad on its own terms: the soup
# scores 71.96 while the mean of its three members is 73.23 - averaging LOST 1.27 points, where
# a soup should land at or above the mean.
#
# Part 1 asks whether weight averaging works here at all, using weights we already have.
# Part 2 builds the honest twin: three fit-split runs with EXACTLY the release recipe.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue11 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
$TEACHER = "weights/hack_dinov2_l_336_cam/best.pt"

# ---------- PART 1: does weight averaging work here at all? (~20 min) ----------
# Same recipe, two seeds, nothing else. If this soup lands at or above the mean of its two
# members (71.82 and 73.95 -> 72.89), averaging is sound and maskp was the bad ingredient.
# If it lands below, soups do not help on this task and the release should be a single model.
Step "soup2" "python scripts/model_soup.py weights/hack_dinov2_b_336_distill20/best.pt weights/hack_dinov2_b_336_seed2/best.pt --out weights/soup2_same_recipe"
Step "val_soup2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup2_same_recipe/best.pt --workers 4 --rerank"

# ---------- Provenance and offline, the two requirements still open (~10 min) ----------
# Answer 46: unambiguous source for external weights. Hash what actually downloaded.
Step "hash_pretrained" "python scripts/hash_pretrained.py"
# Answer 39: exact versions, including transitive ones, taken from inside the built image.
Step "freeze" "docker run --rm --entrypoint python vreid-release -m pip freeze"
Step "pipcheck" "docker run --rm --entrypoint python vreid-release -m pip check"
# Answer 39: inference offline. Runtime env vars are not proof - cut the network and run.
Step "offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_offline:/out vreid-dev"
Step "cmp_offline" "python scripts/compare_outputs.py out_v2_a out_offline"
Step "form_offline" "python scripts/check_submission.py --submission out_offline --data %VREID_DATA%"

# ---------- PART 2: the honest validation twin of the release (~4 hours) ----------
# Exactly the release recipe, but on the fit split so the 307 validation identities stay unseen,
# and --no-select so the epoch is fixed at 18 like the release instead of being chosen on val.
$R = "--config configs/hackathon.yaml --backbone dinov2_b --img-size 336 --epochs 18 --P 11 --K 4 --freeze-blocks 0 --cam-aware --cross-cam-triplet --distill $TEACHER --distill-w 20 --no-select --workers 4"
Step "fit_s0" "python -m vreid.train $R --seed 0 --out weights/fit18_s0"
Step "fit_s2" "python -m vreid.train $R --seed 2 --out weights/fit18_s2"
Step "fit_s3" "python -m vreid.train $R --seed 3 --out weights/fit18_s3"
Step "soup_fit" "python scripts/model_soup.py weights/fit18_s0/best.pt weights/fit18_s2/best.pt weights/fit18_s3/best.pt --out weights/soup_b336_fit"

# Each member and the soup on the same protocol: now the comparison is apples to apples.
Step "val_fit_s0" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_s0/best.pt --workers 4 --rerank"
Step "val_fit_s2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_s2/best.pt --workers 4 --rerank"
Step "val_fit_s3" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/fit18_s3/best.pt --workers 4 --rerank"
Step "val_soup_fit" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup_b336_fit/best.pt --workers 4 --rerank"

# The threshold belongs to the twin, not to a differently built model.
Step "stress_fit" "python scripts/refusal_stress.py --run runs/hack/ft_soup_b336_fit --release release_soup"
Step "audit_fit" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --k1 6 --k2 2"

Log "queue11 finished"
