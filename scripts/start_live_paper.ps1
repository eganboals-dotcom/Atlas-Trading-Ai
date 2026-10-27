# Project Atlas - live paper trading (simulated orders only; prices from Alpaca IEX).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\start_live_paper.ps1
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Python environment not found. See README 'Setup'."; exit 1 }
Write-Host "=== Preflight ==="
& $py -m atlas.cli --log-level ERROR paper-check --config configs\paper_live.yaml
if ($LASTEXITCODE -ne 0) { Write-Host "Preflight failed - fix the FAIL lines above."; exit 1 }
Write-Host "=== Live paper session (Ctrl+C to stop cleanly) ==="
& $py -m atlas.cli --log-level WARNING paper --config configs\paper_live.yaml --auto-resume
