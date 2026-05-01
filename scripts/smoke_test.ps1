#Requires -Version 5.1
<#
.SYNOPSIS
    Local smoke test for the Multi-Cloud AI Workload Manager.
.DESCRIPTION
    Starts the backend API, runs CLI deploy/list/overview against it, then tears down.
    Exits 0 on success, 1 on any failure.
.PARAMETER Port
    Port to run the backend on. Default: 8000.
.PARAMETER ApiUrl
    Override the API URL entirely (skips server start/stop).
#>

param(
    [int]$Port = 8000,
    [string]$ApiUrl = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ManagedServer = $false
$ServerProcess = $null

function Write-Step([string]$Msg) {
    Write-Host "`n==> $Msg" -ForegroundColor Cyan
}

function Fail([string]$Msg) {
    Write-Host "`nFAIL: $Msg" -ForegroundColor Red
    exit 1
}

# ── Resolve API URL ──────────────────────────────────────────────────────────
if (-not $ApiUrl) {
    $ApiUrl = "http://127.0.0.1:$Port"
    $ManagedServer = $true
}

# ── Start backend (if managed) ───────────────────────────────────────────────
if ($ManagedServer) {
    Write-Step "Starting backend server on $ApiUrl"

    $backendDir = Join-Path $ProjectRoot "backend"
    $ServerProcess = Start-Process `
        -FilePath "py" `
        -ArgumentList "-3.11 -m uvicorn app.main:app --host 127.0.0.1 --port $Port" `
        -WorkingDirectory $backendDir `
        -PassThru `
        -WindowStyle Hidden

    Write-Host "  PID $($ServerProcess.Id)"

    # Wait for readiness (up to 20 s)
    Write-Step "Waiting for server readiness"
    $ready = $false
    for ($i = 1; $i -le 20; $i++) {
        try {
            $null = Invoke-WebRequest -Uri "$ApiUrl/api/overview" -UseBasicParsing -TimeoutSec 2
            $ready = $true
            Write-Host "  Ready after ${i}s"
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) {
        if ($ServerProcess -and -not $ServerProcess.HasExited) { Stop-Process -Id $ServerProcess.Id -Force }
        Fail "Backend did not become ready within 20 seconds"
    }
}

# ── Helper: run CLI command ──────────────────────────────────────────────────
function Invoke-CLI([string[]]$CliArgs) {
    $joined = $CliArgs -join " "
    Write-Host "  $ py -3.11 -m flexai_sdk.main $joined"
    $result = & py -3.11 -m flexai_sdk.main @CliArgs
    if ($LASTEXITCODE -ne 0) {
        Fail "CLI command failed (exit $LASTEXITCODE): $joined"
    }
    return $result
}

try {
    # ── Deploy ───────────────────────────────────────────────────────────────
    Write-Step "CLI: deploy smoke-test model"
    $modelPath = Join-Path $ProjectRoot "examples\vision-service\model.ckpt"
    Invoke-CLI @(
        "deploy", $modelPath,
        "--model-name", "smoke-test-model",
        "--gpu", "A100",
        "--region", "us-east",
        "--api-url", $ApiUrl
    )

    # ── List ─────────────────────────────────────────────────────────────────
    Write-Step "CLI: list deployments"
    Invoke-CLI @("list", "--api-url", $ApiUrl)

    # ── Overview ─────────────────────────────────────────────────────────────
    Write-Step "CLI: fleet overview"
    Invoke-CLI @("overview", "--api-url", $ApiUrl)

    # ── REST health check ────────────────────────────────────────────────────
    Write-Step "REST: GET /api/overview"
    $resp = Invoke-WebRequest -Uri "$ApiUrl/api/overview" -UseBasicParsing
    if ($resp.StatusCode -ne 200) {
        Fail "/api/overview returned HTTP $($resp.StatusCode)"
    }
    Write-Host "  HTTP 200 OK"

    Write-Host "`nAll smoke tests passed." -ForegroundColor Green

} finally {
    if ($ManagedServer -and $ServerProcess -and -not $ServerProcess.HasExited) {
        Write-Step "Stopping backend server (PID $($ServerProcess.Id))"
        Stop-Process -Id $ServerProcess.Id -Force
    }
}

exit 0
