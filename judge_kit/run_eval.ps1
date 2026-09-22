$ErrorActionPreference = 'Stop'
# Eval-kit launcher. READS demo/serve.ps1 for paths/flags; never edits it.
# Reuse :8080 if healthy frozen-base is already up; else start :8081.

$kit = $PSScriptRoot
$sat = Split-Path -Parent $kit
$demo = Join-Path $sat 'demo'
$parent = Split-Path -Parent $sat
$qdir = Join-Path $sat 'gates\qwen3vl'

$Questions = ''
$Out = ''
$WithTraces = $false
$ScoreIfGold = ''
for ($i = 0; $i -lt $args.Count; $i++) {
  $a = [string]$args[$i]
  if ($a -eq '-Questions' -and ($i + 1) -lt $args.Count) { $Questions = [string]$args[++$i]; continue }
  if ($a -eq '-Out' -and ($i + 1) -lt $args.Count) { $Out = [string]$args[++$i]; continue }
  if ($a -eq '-WithTraces') { $WithTraces = $true; continue }
  if ($a -eq '-ScoreIfGold' -and ($i + 1) -lt $args.Count) { $ScoreIfGold = [string]$args[++$i]; continue }
}

if (-not $Questions) { $Questions = Join-Path $kit 'inbox\questions.jsonl' }
if (-not $Out) { $Out = Split-Path -Parent $Questions }
if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out | Out-Null }

$py = $null
foreach ($c in @(
  (Join-Path $sat '.venv\Scripts\python.exe'),
  (Join-Path $parent '.venv\Scripts\python.exe')
)) {
  if (Test-Path $c) { $py = $c; break }
}
if (-not $py) { Write-Error 'missing .venv Scripts python.exe'; exit 1 }

$serverExe = Join-Path $qdir 'llama_cpp\llama-server.exe'
$model = Join-Path $qdir 'Qwen3VL-8B-Instruct-Q4_K_M.gguf'
$mmproj = Join-Path $qdir 'mmproj-Qwen3VL-8B-Instruct-F16.gguf'
$wantGguf = '67D1659BFE71B89D50B45A4AD1A9E5B997E5BB16CE5DA66A6A6167ABD569E9E2'
$wantMm = 'CA524100EBF825C9A870DB1C580D03879E0DA0AB2541697E2458E64891CF9D38'

if (-not (Test-Path $model)) { Write-Error "MISSING GGUF $model"; exit 1 }
if (-not (Test-Path $mmproj)) { Write-Error "MISSING mmproj $mmproj"; exit 1 }

Write-Host 'hashing frozen GGUF + mmproj (pre-launch)'
$gotGguf = (Get-FileHash -Algorithm SHA256 $model).Hash.ToUpper()
$gotMm = (Get-FileHash -Algorithm SHA256 $mmproj).Hash.ToUpper()
if ($gotGguf -ne $wantGguf) { Write-Error "STOP GGUF sha mismatch $gotGguf"; exit 2 }
if ($gotMm -ne $wantMm) { Write-Error "STOP mmproj sha mismatch $gotMm"; exit 2 }

function Test-Health([int]$port) {
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 3
    return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
  } catch { return $false }
}

$vlmUrl = 'http://127.0.0.1:8080'
$serverTag = 'reused_8080'
$startedHere = $false
$srvProc = $null

if (Test-Health 8080) {
  Write-Host 'reusing healthy frozen-base on :8080'
} elseif (Test-Health 8081) {
  $vlmUrl = 'http://127.0.0.1:8081'
  $serverTag = 'reused_8081'
  Write-Host 'reusing already-up kit server on :8081'
} else {
  if (-not (Test-Path $serverExe)) { Write-Error "MISSING llama-server.exe $serverExe"; exit 1 }
  $vlmUrl = 'http://127.0.0.1:8081'
  $serverTag = 'kit_8081'
  $startedHere = $true
  $srvLog = Join-Path $kit 'llama_8081.log'
  $srvErr = Join-Path $kit 'llama_8081_err.log'
  $argStr = "-m `"$model`" --mmproj `"$mmproj`" -ngl 99 -c 4096 --port 8081 --host 127.0.0.1"
  Write-Host "starting llama-server on :8081"
  $srvProc = Start-Process -FilePath $serverExe -ArgumentList $argStr -WindowStyle Hidden -RedirectStandardOutput $srvLog -RedirectStandardError $srvErr -PassThru
  $healthy = $false
  foreach ($i in 1..80) {
    Start-Sleep -Seconds 3
    if ($srvProc.HasExited) { Write-Error 'SERVER EXITED EARLY'; exit 2 }
    if (Test-Health 8081) { $healthy = $true; break }
  }
  if (-not $healthy) { Write-Error 'SERVER NEVER BECAME HEALTHY on :8081'; exit 2 }
}

$env:PYTHONUTF8 = '1'
$pyArgs = @(
  (Join-Path $kit 'drop_eval.py'),
  '--questions', $Questions,
  '--out', $Out,
  '--vlm-url', $vlmUrl,
  '--server', $serverTag
)
if ($startedHere) { $pyArgs += '--started-here' }
if ($WithTraces) { $pyArgs += '--with-traces' }
if ($ScoreIfGold) { $pyArgs += @('--score-if-gold', $ScoreIfGold) }

Set-Location $sat
& $py @pyArgs
$ec = $LASTEXITCODE
exit $ec
