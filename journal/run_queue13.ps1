# Queue 13: fix what queue 12 got wrong, rebuild the image for real, re-issue on the new image.
#     powershell -ExecutionPolicy Bypass -File run_queue13.ps1
# ASCII only. About 25 minutes, no training.
#
# What broke in queue 12 and why:
#   build12       pip check failed inside the build: anyio 4.15.1 wants typing_extensions>=4.16.0.
#                 Pinning 4.16.0 did not help - pip downloaded 4.16.0 and reported installing
#                 4.12.1, twice, in queues 9 and 12. Fixed at the root instead: anyio, httpx,
#                 httpcore and h11 are the HTTP layer of huggingface_hub, and we never touch the
#                 network at runtime (HF_HUB_OFFLINE=1, pretrained=False, proven with
#                 --network none). Dropped from the closure, so nothing requires >=4.16.0.
#                 Because the build failed, every docker step in queue 12 silently ran the OLD
#                 image - those green results say nothing about the merge and are redone here.
#   soup2         weights/soup2_same_recipe already existed as a FILE, left over from queue 11's
#                 broken --out. Moved to _to_delete.
#   audit/gallery I wrote %REC% in a PowerShell string. $REC is a PowerShell variable, invisible
#                 to cmd, so the scripts got the literal text and looked for a recipe in the
#                 current directory. Uses $REC now.
#   3 tests       They encoded the partner's two decisions that we deliberately reversed
#                 (crash on out-of-frame bbox, crash instead of cosine fallback). Rewritten to
#                 assert our behaviour WITH the reasoning, so nobody silently flips it back.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue13 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
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
$env:VREID_RECIPE = "release_soup/recipe.json"
Log "data folder: $env:VREID_DATA"

Step "tests13" "python -m pytest tests -q"

# 1. The soup question, second angle: same recipe, two seeds, from the older 20-epoch runs.
Step "soup2" "python scripts/model_soup.py weights/hack_dinov2_b_336_distill20/best.pt weights/hack_dinov2_b_336_seed2/best.pt --out weights/soup2_same_recipe/best.pt"
Step "val_soup2" "python -m vreid.hack_cli val --config configs/hackathon.yaml --backbone ft:weights/soup2_same_recipe/best.pt --workers 4 --rerank"

# 2. The diagnostics that never ran, now with a recipe path cmd can actually see
Step "audit_fit"  "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe %VREID_RECIPE% --out results/refusal_audit_fit.json"
Step "audit_val"  "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_val --recipe %VREID_RECIPE% --out results/refusal_audit_val.json"
Step "gallery_fit" "python scripts/gallery_size_effect.py --run runs/hack/ft_soup_b336_val --recipe %VREID_RECIPE% --sizes 683 685 688 --out results/gallery_size_effect_fixed.json"

# 3. Recipe provenance now points at the structurally matching twin. Threshold stays .55 -
#    it is a deliberate hedge, not a fit, and the twin confirms it (A 96.40, B 63.16).
Step "export_twin" "python scripts/export_release.py --weights weights/soup_b336_all/best.pt --val results/hack_ft_soup_b336_fit_val.json --threshold 0.55 --kr --k1 6 --k2 2 --out release_soup"
Step "copy_release" "xcopy /Y /E /I release_soup release"

# 4. The image, for real this time. pip check runs inside the build and must pass.
Step "build13" "docker build -t vreid-release ."
Step "build13dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime -t vreid-dev ."
Step "packages13" "docker run --rm --entrypoint python vreid-release /app/scripts/image_report.py"

# 5. Submission re-issued on the merged code, then every proof redone on the NEW image
Step "predict13" "python -m vreid.predict --data %VREID_DATA% --release release_soup --out submission_v4 --workers 8"
Step "cmp_v4" "python scripts/compare_outputs.py submission_v3 submission_v4"
Step "form_v4" "python scripts/check_submission.py --submission submission_v4 --data %VREID_DATA%"
Step "indep_v4" "python scripts/check_query_independence.py --submission submission_v4 --data %VREID_DATA% --frac 0.6"
Step "ties13" "python scripts/check_tie_independence.py"
Step "docker_v4a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v4_a:/out vreid-dev"
Step "docker_v4b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v4_b:/out vreid-dev"
Step "cmp_v4_det" "python scripts/compare_outputs.py out_v4_a out_v4_b"
Step "docker_v4_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v4_offline:/out vreid-dev"
Step "cmp_v4_offline" "python scripts/compare_outputs.py out_v4_a out_v4_offline"
Step "docker_v4_noshm" "docker run --rm --gpus all -v %VREID_DATA%:/data:ro -v %cd%\out_v4_noshm:/out vreid-dev"
Step "cmp_v4_noshm" "python scripts/compare_outputs.py out_v4_a out_v4_noshm"
Step "form_v4_docker" "python scripts/check_submission.py --submission out_v4_a --data %VREID_DATA%"
Step "replay13" "python scripts/check_cached_replay.py --submission submission_v4 --data %VREID_DATA% --release release_soup --out results/cached_replay.json"

# 6. Performance on the rebuilt image, by the organisers' own method
Step "bench13" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

Step "stress13" "python scripts/refusal_stress.py --run runs/hack/ft_soup_b336_fit --release release_soup --out results/refusal_stress_fit.json"
Step "scorecard13" "python scripts/scorecard.py --release release_soup --submission submission_v4"
Step "manifest13" "python scripts/make_manifest.py --release release_soup --submission submission_v4 --run runs/hack/ft_soup_b336_fit --out handoff/MANIFEST.json"

Log "queue13 finished"
