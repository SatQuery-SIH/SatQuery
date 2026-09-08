$ErrorActionPreference = 'Continue'
# SatQuery demo launcher (ASCII only). Starts llama-server then Gradio.
# Paths are relative to this script so clones work. Weights are local, not in git.
$demo = $PSScriptRoot
$sat = Split-Path -Parent $demo
$qdir = Join-Path $sat 'gates\qwen3vl'
$parent = Split-Path -Parent $sat
$mode = 'live'
if ($args.Count -ge 1) { $mode = $args[0] }

$py = $null
foreach ($c in @(
  (Join-Path $sat '.venv\Scripts\python.exe'),
  (Join-Path $parent '.venv\Scripts\python.exe'),
  'python'
)) {
  if ($c -eq 'python') { $py = 'python'; break }
  if (Test-Path $c) { $py = $c; break }
}

$log = Join-Path $demo 'serve.log'
"=== SatQuery serve mode=$mode start=$(Get-Date -Format o) ===" | Out-File $log -Encoding ascii

$server = Join-Path $qdir 'llama_cpp\llama-server.exe'
$model = Join-Path $qdir 'Qwen3VL-8B-Instruct-Q4_K_M.gguf'
$mmproj = Join-Path $qdir 'mmproj-Qwen3VL-8B-Instruct-F16.gguf'
$srvLog = Join-Path $demo 'llama_server.log'
$srvErr = Join-Path $demo 'llama_server_err.log'

$needServer = $mode -ne 'cached'
$srvProc = $null
if ($needServer) {
  if (-not (Test-Path $server)) { "MISSING llama-server.exe" | Out-File $log -Append -Encoding ascii; exit 1 }
  if (-not (Test-Path $model)) { "MISSING GGUF" | Out-File $log -Append -Encoding ascii; exit 1 }
  $argStr = "-m `"$model`" --mmproj `"$mmproj`" -ngl 99 -c 4096 --port 8080 --host 127.0.0.1"
  "args: $argStr" | Out-File $log -Append -Encoding ascii
  $srvProc = Start-Process -FilePath $server -ArgumentList $argStr -WindowStyle Hidden -RedirectStandardOutput $srvLog -RedirectStandardError $srvErr -PassThru
  "server PID: $($srvProc.Id)" | Out-File $log -Append -Encoding ascii
  $healthy = $false
  foreach ($i in 1..80) {
    Start-Sleep -Seconds 3
    if ($srvProc.HasExited) { "SERVER EXITED EARLY" | Out-File $log -Append -Encoding ascii; break }
    try {
      $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -UseBasicParsing -TimeoutSec 3
      if ($h.StatusCode -eq 200) { $healthy = $true; "healthy after ~$($i*3)s" | Out-File $log -Append -Encoding ascii; break }
    } catch { }
  }
  if (-not $healthy) {
    "SERVER NEVER BECAME HEALTHY" | Out-File $log -Append -Encoding ascii
    exit 2
  }
} else {
  "cached mode: llama-server not started (zero GPU)" | Out-File $log -Append -Encoding ascii
}

Set-Location $demo
"starting app.py --mode $mode" | Out-File $log -Append -Encoding ascii
& $py app.py --mode $mode --port 7860
