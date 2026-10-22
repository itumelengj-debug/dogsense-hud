# Starts the DogSense server for live use with the glasses.
#   .\start_server.ps1                 # record every session to .\recordings
#   .\start_server.ps1 -NoRecord
#   .\start_server.ps1 -Vlm qwen2.5vl:7b
param(
    [int]$Port = 8765,
    [switch]$NoRecord,
    [string]$Vlm = ""
)
Set-Location $PSScriptRoot

# The glasses can only connect if Windows Firewall lets inbound TCP through on the port.
$rule = Get-NetFirewallRule -DisplayName "DogSense" -ErrorAction SilentlyContinue
if (-not $rule) {
    Write-Host "Firewall: no 'DogSense' rule yet. Run this once in an ADMIN PowerShell:" -ForegroundColor Yellow
    Write-Host "  netsh advfirewall firewall add rule name=`"DogSense`" dir=in action=allow protocol=TCP localport=$Port" -ForegroundColor Yellow
    Write-Host ""
}

$argsList = @("-m", "server.app", "--host", "0.0.0.0", "--port", $Port)
if (-not $NoRecord) { $argsList += @("--record", "recordings") }
if ($Vlm) { $argsList += @("--vlm", $Vlm) }
$env:PYTHONIOENCODING = "utf8"
python @argsList
