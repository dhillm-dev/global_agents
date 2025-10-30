param(
  [string]$BaseUrl = "https://global-agents.vercel.app"
)

function Try-Endpoints {
  param(
    [string]$Primary,
    [string]$Fallback,
    [string]$Method = 'Get',
    [string]$Body = ''
  )
  try {
    if ($Method -eq 'Post') {
      return Invoke-RestMethod -Method Post -Uri $Primary -ContentType 'application/json' -Body $Body -TimeoutSec 25
    } else {
      return Invoke-RestMethod -Method Get -Uri $Primary -TimeoutSec 15
    }
  } catch {
    try {
      if ($Method -eq 'Post') {
        return Invoke-RestMethod -Method Post -Uri $Fallback -ContentType 'application/json' -Body $Body -TimeoutSec 25
      } else {
        return Invoke-RestMethod -Method Get -Uri $Fallback -TimeoutSec 15
      }
    } catch {
      Write-Host $_.Exception.Message -ForegroundColor Yellow
      return $null
    }
  }
}

Write-Host "Healthz:" -ForegroundColor Cyan
${h} = Try-Endpoints -Primary "$BaseUrl/healthz" -Fallback "$BaseUrl/api/healthz" -Method 'Get'
if ($h) { $h | ConvertTo-Json -Depth 4 }

Write-Host "Flow Snapshot:" -ForegroundColor Cyan
${f} = Try-Endpoints -Primary "$BaseUrl/flow/snapshot" -Fallback "$BaseUrl/api/flow/snapshot" -Method 'Get'
if ($f) { $f | ConvertTo-Json -Depth 6 }

Write-Host "Alpha Hunter:" -ForegroundColor Cyan
$alphaBody = '{"symbol":"ETHUSD","timeframe":"1h"}'
${a} = Try-Endpoints -Primary "$BaseUrl/alpha/hunter" -Fallback "$BaseUrl/api/alpha/hunter" -Method 'Post' -Body $alphaBody
if ($a) { $a | ConvertTo-Json -Depth 6 }