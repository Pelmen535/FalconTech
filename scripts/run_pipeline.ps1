param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$Output = "outputs/metrics.json",
    [ValidateSet("color-grid", "onnx")]
    [string]$Backend = "color-grid",
    [string]$Model = "",
    [string]$ModelSpec = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
if (-not $Python) {
    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $Python = (Resolve-Path -LiteralPath $venvPython).Path
    } else {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($command) { $Python = $command.Source }
    }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
    throw "Python was not found. Pass its path with -Python."
}

& $Python -m pip install -e . --no-build-isolation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m vehicle_reid audit --manifest $Manifest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$arguments = @("-m", "vehicle_reid", "evaluate", "--manifest", $Manifest, "--out", $Output, "--backend", $Backend)
if ($Backend -eq "onnx") {
    if (-not $Model -or -not $ModelSpec) {
        throw "ONNX backend requires -Model and -ModelSpec"
    }
    $arguments += @("--model", $Model, "--model-spec", $ModelSpec)
}

& $Python @arguments
exit $LASTEXITCODE
