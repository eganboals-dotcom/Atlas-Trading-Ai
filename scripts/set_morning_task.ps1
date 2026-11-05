# Project Atlas - moves the "Atlas daily research" task to 5:35 am and has it start the weekday paper session.
# Windows only lets an administrator change this task. Run it once from an administrator PowerShell.
# 5:35 leaves the motherboard's 5:30 alarm time to boot; no time zone offset, so it stays 5:35 through clock changes.
$task = Get-ScheduledTask -TaskName "Atlas daily research"
$task.Triggers[0].StartBoundary = "2026-11-02T05:35:00"
$task.Settings.ExecutionTimeLimit = "PT12H"  # the task now lasts until the paper session ends
$task.Settings.WakeToRun = $true
if ($task.Actions[0].Arguments -notmatch "-Trade") { $task.Actions[0].Arguments += " -Trade" }
# Sleep, not shutdown: the 5:35 trigger can wake a sleeping PC, but only the motherboard's alarm can start one that's off.
$task.Actions[0].Arguments = $task.Actions[0].Arguments -replace "-After shutdown", "-After sleep"
Set-ScheduledTask -InputObject $task | Out-Null
$next = ($task | Get-ScheduledTaskInfo).NextRunTime
Write-Host "Done. Next run: $next"
Read-Host "Press Enter to close"
