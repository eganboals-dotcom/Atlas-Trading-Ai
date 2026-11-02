# Project Atlas - opens the read-only leaderboard for today's paper session on weekdays.
# The morning task runs the session with no screen, so this shows it on the signed-in desktop. It runs at
# sign-in (a Startup folder shortcut) and at 5:40 am ("Atlas leaderboard" task) for a PC already signed in.
# A window left open points at its own day, so an older one is closed first.
if ((Get-Date).DayOfWeek -in "Saturday", "Sunday") { exit 0 }
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
    Where-Object { $_.CommandLine -match "atlas\.cli.*\spaper-window\s" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
Start-Process -FilePath ".venv\Scripts\pythonw.exe" -ArgumentList "-m atlas.cli paper-window --config configs\paper_live.yaml"
