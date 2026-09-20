# Queue 14: собрать продукт целиком и переснять ВСЕ доказательства на пересобранных образах.
#     powershell -ExecutionPolicy Bypass -File run_queue14.ps1
# ASCII only in code, about 30 minutes, no training.
#
# Why this queue exists. Queue 13 left three things broken or unproven:
#   1. Both docker builds failed. The release image died on `pip check` (huggingface_hub 1.32
#      needs httpx, which had been dropped from the closure) and then on load_recipe(str) after
#      the merge; the dev image died on PEP 668. Every "green" docker step in queue 13 therefore
#      ran the OLD image and proved nothing about the merged code.
#   2. The benchmark was not attributed to the release: scorecard refused it because the image
#      predated the merge. Correct behaviour of the tool, wrong state of the world.
#   3. The product side of the task (TZ section 6: microservice split, database, OpenAPI,
#      browser client, one-command compose) existed only as a separate archive from the partner.
#
# What changed since queue 13, and is verified here:
#   - Dockerfile fixed and parameterised (PIP_FLAGS/REQUIREMENTS for the local sm_120 base).
#   - service/ merged into the repo, plus explain (exact cosine decomposition), ANN shortlist,
#     demo endpoints, PostgreSQL/pgvector storage, offline Swagger UI.
#   - release/ is now the single canonical release directory; recipe.json carries
#     threshold_choice, so the threshold argument travels with the artefact.
#   - submission/ is the canonical competition output; submission_v* are history.

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$env:VREID_THROTTLE = "0"
# Compose on this machine cannot use bake: the project path is not ASCII and buildkit
# rejects the session header it derives from it. Per-service build works.
$env:COMPOSE_DOCKER_CLI_BUILD = "0"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$status = "logs\queue_status.txt"
"queue14 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $status -Append -Encoding utf8
function Log($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $status -Append -Encoding utf8 }
function Step($name, $cmd) {
    Log "START $name"; $t0 = Get-Date
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log" -Encoding utf8
    $global:code = $LASTEXITCODE
    Log "END   $name  exit=$global:code  $([int]((Get-Date) - $t0).TotalMinutes) min"
}

# The GPU must be free: a loaded model in the service container would sit in VRAM during
# the benchmark and the numbers would describe the machine, not the release.
Log "WAIT  for running vreid processes"
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "vreid\.(train|hack_cli|predict)|run_service" }
    if (-not $busy) { break }
    Start-Sleep -Seconds 30
}
docker compose down | Out-Null
Log "WAIT  done - GPU free, service stack down"

$env:VREID_DATA = (Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName "test_query.csv") } | Select-Object -First 1).FullName
Log "data folder: $env:VREID_DATA"

# 1. Everything the repo asserts about itself
Step "tests14" "python -m pytest tests -q"

# 2. Images, for real. A failing build must stop being a silent no-op.
Step "build14_release" "docker build -t vreid-release ."
Step "build14_dev" "docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-dev ."
Step "build14_api" "docker build -f Dockerfile.service --build-arg BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --build-arg PIP_FLAGS=--break-system-packages --build-arg REQUIREMENTS=requirements-infer-dev.txt -t vreid-api:1.1 ."
Step "build14_web" "docker build -f web/Dockerfile -t vreid-web:1.1 ."
Step "packages14" "docker run --rm --entrypoint python vreid-release /app/scripts/image_report.py"

# 3. The submission itself, on the merged code
Step "predict14" "python -m vreid.predict --data %VREID_DATA% --release release --out submission --workers 8"
Step "cmp14_v4" "python scripts/compare_outputs.py submission_v4 submission"
Step "form14" "python scripts/check_submission.py --submission submission --data %VREID_DATA%"
Step "indep14" "python scripts/check_query_independence.py --submission submission --data %VREID_DATA% --frac 0.6"
Step "ties14" "python scripts/check_tie_independence.py"
Step "replay14" "python scripts/check_cached_replay.py --submission submission --data %VREID_DATA% --release release --out results/cached_replay.json"

# 4. The same thing in the container, twice, offline, and without shared memory
Step "docker14_a" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v5_a:/out vreid-dev"
Step "docker14_b" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v5_b:/out vreid-dev"
Step "cmp14_det" "python scripts/compare_outputs.py out_v5_a out_v5_b"
Step "docker14_offline" "docker run --rm --network none --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\out_v5_offline:/out vreid-dev"
Step "cmp14_offline" "python scripts/compare_outputs.py out_v5_a out_v5_offline"
Step "docker14_noshm" "docker run --rm --gpus all -v %VREID_DATA%:/data:ro -v %cd%\out_v5_noshm:/out vreid-dev"
Step "cmp14_noshm" "python scripts/compare_outputs.py out_v5_a out_v5_noshm"
Step "cmp14_native" "python scripts/compare_outputs.py submission out_v5_a"
Step "form14_docker" "python scripts/check_submission.py --submission out_v5_a --data %VREID_DATA%"

# 5. Performance by the organisers' own method, on the image that exists now
Step "bench14" "docker run --rm --gpus all --shm-size=2g -v %VREID_DATA%:/data:ro -v %cd%\results:/app/results --entrypoint python vreid-dev -m vreid.bench_full --data /data --release /app/release --n 300 --warmup 50 --min-seconds 10 --workers 8 --fast-decode"

# 6. Refusal, error analysis, and the card that says what all of it is worth
Step "stress14" "python scripts/refusal_stress.py --run runs/hack/ft_soup_b336_fit --release release --out results/refusal_stress_fit.json"
Step "audit14" "python scripts/refusal_audit.py --run runs/hack/ft_soup_b336_fit --recipe release/recipe.json --out results/refusal_audit_fit.json"
Step "errors14" "python scripts/error_analysis.py --run runs/hack/ft_soup_b336_fit --release release --data %VREID_DATA%"
Step "scorecard14" "python scripts/scorecard.py --release release --submission submission"
Step "manifest14" "python scripts/make_manifest.py --release release --submission submission --run runs/hack/ft_soup_b336_fit --out handoff/MANIFEST.json"

# 7. The product back up, so the prototype link works right after the queue
Step "compose14" "docker compose up -d"

Log "queue14 finished"
