$ErrorActionPreference = 'Continue'
# DEMO-SPEC-05 row A. Disconnect Wi-Fi, run the 3 live scenes on localhost, reconnect.
# ASCII only. Run from an elevated-enough PowerShell if disconnect is denied.
# Optional: $env:SATQUERY_WIFI_SSID = your network name before reconnect.
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
$ssid = $env:SATQUERY_WIFI_SSID
$out = Join-Path $demo 'traces\airplane.json'

'=== airplane_test start ' + (Get-Date -Format o) + ' ==='
netsh wlan show interfaces
'--- disconnect ---'
netsh wlan disconnect
Start-Sleep -Seconds 3
netsh wlan show interfaces
'wan ping (expect fail):'
try { Test-Connection -ComputerName 8.8.8.8 -Count 1 -TimeoutSeconds 2 | Out-Null; 'WAN REACHABLE' } catch { 'WAN UNREACHABLE (good)' }
'llama health:'
try { (Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -UseBasicParsing -TimeoutSec 3).StatusCode } catch { 'llama fail' }
'ui:'
try { (Invoke-WebRequest -Uri 'http://127.0.0.1:7860' -UseBasicParsing -TimeoutSec 5).StatusCode } catch { 'ui fail' }
Set-Location $demo
& $py drive_ui.py
'--- reconnect ---'
if ($ssid) {
  netsh wlan connect name=$ssid
} else {
  'SATQUERY_WIFI_SSID not set; reconnect Wi-Fi yourself'
}
Start-Sleep -Seconds 5
netsh wlan show interfaces
'=== airplane_test end ==='
