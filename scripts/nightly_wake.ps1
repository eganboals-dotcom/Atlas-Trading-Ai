# Project Atlas - unattended nightly research (research only; places no orders).
# For the "Atlas daily research" scheduled task. It waits for the internet, starts the local AI reader
# (Ollama) if it isn't running, keeps the PC awake while scripts\swing_track.cmd auto runs, then shuts the
# PC down (-After shutdown), puts it back to sleep (-After sleep) or leaves it on (-After none).
# It never powers off a PC someone is using: it stays on if the mouse or keyboard was used in the last
# -IdleMinutes, if anyone is signed in while the task runs in the background (no one signed in is the
# normal case after the motherboard's alarm boots the PC), or if a live paper session is running.
# Every step is appended to artifacts\reports\swing_track\runs.log.
param(
    [ValidateSet("shutdown", "sleep", "none")] [string] $After = "shutdown",
    [int] $IdleMinutes = 10
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root "artifacts\reports\swing_track\runs.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Write-Log([string] $msg) { Add-Content -Path $log -Value "[nightly_wake $(Get-Date -Format s)] $msg" }

Add-Type -Namespace Atlas -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
[StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
[DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO info);
public static double IdleMinutes() {
    LASTINPUTINFO i = new LASTINPUTINFO(); i.cbSize = (uint)Marshal.SizeOf(i);
    if (!GetLastInputInfo(ref i)) return 0;
    return unchecked((uint)Environment.TickCount - i.dwTime) / 60000.0;
}
"@

# Session 0 means the task runs in the background ("whether user is logged on or not"); mouse and keyboard
# idle time is only visible from the signed-in user's own session.
$background = (Get-Process -Id $PID).SessionId -eq 0
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: without it Windows sleeps again ~2 minutes after a timed wake.
[Atlas.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Write-Log "start (after: $After, background: $background)"
$ollama = $null
try {
    $deadline = (Get-Date).AddMinutes(10)  # right after boot the network can take a minute or two
    while (-not (Get-NetConnectionProfile -ErrorAction SilentlyContinue | Where-Object IPv4Connectivity -eq "Internet")) {
        if ((Get-Date) -gt $deadline) { Write-Log "no internet after 10 minutes; running anyway"; break }
        Start-Sleep -Seconds 10
    }
    $exe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (-not (Get-Process -Name ollama -ErrorAction SilentlyContinue) -and (Test-Path $exe)) {
        $ollama = Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden -PassThru
        Write-Log "started Ollama (process $($ollama.Id))"
        Start-Sleep -Seconds 15
    }
    & cmd.exe /c "`"$root\scripts\swing_track.cmd`" auto"
    Write-Log "research finished (exit $LASTEXITCODE)"
} finally {
    if ($ollama) { Stop-Process -Id $ollama.Id -ErrorAction SilentlyContinue }
    [Atlas.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
}

if ($After -eq "none") { exit 0 }
$paper = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "atlas\.cli.*\spaper\s" }
$signedIn = Get-CimInstance Win32_Process -Filter "Name = 'explorer.exe'"
$idle = [Atlas.Power]::IdleMinutes()
if ($background -and $signedIn) {
    Write-Log "staying on: someone is signed in"
} elseif (-not $background -and $idle -lt $IdleMinutes) {
    Write-Log "staying on: someone used the PC $([math]::Round($idle, 1)) min ago"
} elseif ($paper) {
    Write-Log "staying on: a live paper session is running (process $($paper[0].ProcessId))"
} elseif ($After -eq "shutdown") {
    Write-Log "shutting down in 60 s (cancel with: shutdown /a)"
    & shutdown.exe /s /t 60 /c "Atlas nightly research finished; shutting down. Run 'shutdown /a' to cancel."
} else {
    Write-Log "going back to sleep"
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.Application]::SetSuspendState([System.Windows.Forms.PowerState]::Suspend, $false, $false) | Out-Null
}
