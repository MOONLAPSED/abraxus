Write-Host "Starting Visual Studio minimal install script..."

# Define the desired Visual Studio version and components
$VSVersion = "2022" # Or "2019", etc. - choose the one you prefer
$VSEdition = "Community" # Or "Professional", "Enterprise"
$InstallDir = "C:\Program Files (x86)\Microsoft Visual Studio\$VSVersion\$VSEdition" # Adjust if needed

# The workload ID for Desktop development with C++ (includes native debugging)
$WorkloadId = "Microsoft.VisualStudio.Workload.NativeDesktop"

# Optional individual components - we can refine this if needed
# For basic debugging, the workload might be enough.
# Let's start with just the workload for simplicity.
# $ComponentIds = @(
#     "Microsoft.VisualStudio.Component.Debugger.JustInTime",
#     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" # Common tools
# )

# Construct the command-line arguments for the installer
$InstallerArgs = @(
    "--installPath", "`"$InstallDir`"",
    "--add", "`"$WorkloadId`"",
    "--passive", # No user interaction
    "--norestart", # Don't restart automatically
    "--wait"      # Wait for the installation to complete
    # If you had specific components:
    # "--add", "`$ComponentIds -join ','`"
)

# Construct the full installer command
$InstallerCommand = Join-Path -Path $env:TEMP -ChildPath "vs_installer.exe"
$DownloadUrl = "https://aka.ms/vs/$VSVersion/release/vs_Community.exe" # Adjust for edition

# Check if the installer already exists (unlikely in a fresh sandbox)
if (-not (Test-Path $InstallerCommand)) {
    Write-Host "Downloading Visual Studio Installer..."
    try {
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $InstallerCommand
        if (-not $?) {
            Write-Error "Failed to download Visual Studio Installer."
            exit 1
        }
    } catch {
        Write-Error "Error downloading Visual Studio Installer: $($_.Exception.Message)"
        exit 1
    }
} else {
    Write-Host "Visual Studio Installer already present."
}

# Start the Visual Studio installation
Write-Host "Starting Visual Studio $VSEdition $VSVersion installation (minimal)..."
Start-Process -FilePath $InstallerCommand -ArgumentList $InstallerArgs -Wait -NoNewWindow

if ($LASTEXITCODE -eq 0) {
    Write-Host "Visual Studio minimal installation completed successfully."
} else {
    Write-Error "Visual Studio installation failed with exit code: $($LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Visual Studio minimal install script finished."