# Loads .env into this PowerShell session's environment variables, then runs
# the app. Local development convenience only - never used by the real
# Docker container, which reads env vars injected by the judging harness.
#
# Usage:
#   1. Copy .env.example to .env and fill in your real values (one time).
#   2. Whenever you want to run the app:  .\run_local.ps1
#   3. Or just run a different command with the env loaded:  .\run_local.ps1 -Command "python validate_submission.py ..."

param(
    [string]$Command = "python -m app.main"
)

$envFile = Join-Path $PSScriptRoot ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "No .env file found at $envFile" -ForegroundColor Yellow
    Write-Host "Copy .env.example to .env and fill in your real values first:" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.example .env" -ForegroundColor Cyan
    exit 1
}

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split "=", 2
        if ($parts.Length -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim()
            Set-Item -Path "Env:$name" -Value $value
            if ($name -eq "FIREWORKS_API_KEY") {
                Write-Host "Set $name = ****$($value.Substring([Math]::Max(0, $value.Length - 4)))" -ForegroundColor Green
            } else {
                Write-Host "Set $name = $value" -ForegroundColor Green
            }
        }
    }
}

Write-Host ""
Write-Host "Running: $Command" -ForegroundColor Cyan
Invoke-Expression $Command
