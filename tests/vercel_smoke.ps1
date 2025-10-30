param(
  [string]$BaseUrl = "https://global-agents.vercel.app"
)

Write-Host "Healthz:" -ForegroundColor Cyan
try {
  $h = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/ta_decision/healthz" -TimeoutSec 10
  $h | ConvertTo-Json -Depth 4
} catch { Write-Host $_.Exception.Message -ForegroundColor Yellow }

Write-Host "TA Decision:" -ForegroundColor Cyan
$body = '{"symbol":"ETHUSD","tf":"H1","bars":300}'
try {
  $d = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/ta_decision" -ContentType 'application/json' -Body $body -TimeoutSec 25
  $d | ConvertTo-Json -Depth 6
} catch { Write-Host $_.Exception.Message -ForegroundColor Yellow }

Write-Host "Run Once (GET):" -ForegroundColor Cyan
try {
  $r = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/run_once?symbol=ETHUSD&tf=H1&bars=300" -TimeoutSec 25
  $r | ConvertTo-Json -Depth 6
} catch { Write-Host $_.Exception.Message -ForegroundColor Yellow }