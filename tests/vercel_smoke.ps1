param(
  [string]$BaseUrl = "https://global-agents.vercel.app"
)

Write-Host "Healthz:" -ForegroundColor Cyan
try {
  $h = Invoke-RestMethod -Method Get -Uri "$BaseUrl/healthz" -TimeoutSec 10
  $h | ConvertTo-Json -Depth 4
} catch { Write-Host $_.Exception.Message -ForegroundColor Yellow }

Write-Host "Flow Snapshot:" -ForegroundColor Cyan
try {
  $f = Invoke-RestMethod -Method Get -Uri "$BaseUrl/flow/snapshot" -TimeoutSec 10
  $f | ConvertTo-Json -Depth 6
} catch { Write-Host $_.Exception.Message -ForegroundColor Yellow }

Write-Host "Alpha Hunter:" -ForegroundColor Cyan
$alphaBody = '{"symbol":"ETHUSD","timeframe":"1h"}'
try {
  $a = Invoke-RestMethod -Method Post -Uri "$BaseUrl/alpha/hunter" -ContentType 'application/json' -Body $alphaBody -TimeoutSec 25
  $a | ConvertTo-Json -Depth 6
} catch { Write-Host $_.Exception.Message -ForegroundColor Yellow }