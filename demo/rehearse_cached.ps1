$ErrorActionPreference = 'Stop'
# Cold-boot cached rehearsal: zero GPU. Spec row E.
$demo = $PSScriptRoot
$sat = Split-Path -Parent $demo
$parent = Split-Path -Parent $sat
$py = $null
foreach ($c in @(
  (Join-Path $sat '.venv\Scripts\python.exe'),
  (Join-Path $parent '.venv\Scripts\python.exe'),
  'python'
)) {
  if ($c -eq 'python') { $py = 'python'; break }
  if (Test-Path $c) { $py = $c; break }
}
Set-Location $demo
Write-Host "rehearse_cached start $(Get-Date -Format o)"
& $py cache.py --mode cached
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "launching app --mode cached on 127.0.0.1:7860 (Ctrl+C to stop)"
& $py app.py --mode cached --port 7860
