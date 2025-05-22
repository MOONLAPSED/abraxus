Start-Sleep -Seconds 20  # allow sandbox to fully initialize

# Path to your actual setup script
$setupScript = "C:\Users\WDAGUtilityAccount\invoke_setup.bat"

if (Test-Path $setupScript) {
    Start-Process -FilePath $setupScript -WindowStyle Normal
} else {
    Write-Host "Setup script not found at $setupScript"
}
