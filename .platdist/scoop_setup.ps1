Write-Host "Starting scoop_setup.ps1 script..."

# Function to safely add a bucket
function Add-ScoopBucket {
    param([string]$bucketName, [string]$repoUrl = "")
    
    Write-Host "Checking bucket: $bucketName"
    $buckets = scoop bucket list
    if ($buckets -match $bucketName) {
        Write-Host "Bucket '$bucketName' is already added."
    } else {
        Write-Host "Adding bucket: $bucketName"
        if ($repoUrl) {
            scoop bucket add $bucketName $repoUrl
        } else {
            scoop bucket add $bucketName
        }
    }
}

# Function to safely install a package
function Install-ScoopPackage {
    param([string]$packageName, [string]$bucket = "")
    
    $fullPackageName = if ($bucket) { "$bucket/$packageName" } else { $packageName }
    Write-Host "Checking package: $fullPackageName"
    
    $installed = scoop list | Where-Object { $_.Name -eq $packageName }
    if ($installed) {
        Write-Host "Package '$fullPackageName' is already installed."
        return $true
    } else {
        Write-Host "Installing package: $fullPackageName"
        scoop install $fullPackageName
        return $?  # Return success status of installation
    }
}

# Ensure Scoop is in the PATH for this session
$env:SCOOP = "C:\Users\WDAGUtilityAccount\scoop"
$env:PATH = "$env:SCOOP\shims;$env:PATH"

# Verify Scoop is working
try {
    $scoopVersion = & scoop --version
    Write-Host "Scoop version: $scoopVersion"
} catch {
    Write-Error "Error accessing Scoop. Details: $_"
    exit 1
}

# First, ensure git is installed (required for adding buckets)
Install-ScoopPackage "git"

# Add required buckets
Add-ScoopBucket "versions"
Add-ScoopBucket "extras"
Add-ScoopBucket "nerd-fonts"

Write-Host "Installing essential packages..."
# Install packages with error handling
$packages = @(
    @{name="gh"; bucket="main"},
    @{name="vscode"; bucket="extras"},
    @{name="windows-terminal"; bucket="extras"},
    @{name="7zip"; bucket="main"},
    @{name="fzf"; bucket="main"},
    @{name="yazi"; bucket="main"},
    @{name="uv"; bucket="main"},
    @{name="notepadplusplus"; bucket="extras"}
)

foreach ($package in $packages) {
    Install-ScoopPackage -packageName $package.name -bucket $package.bucket
}

# Update all installed applications
Write-Host "Updating all installed applications..."
scoop update *

# Launch common applications
Write-Host "Launching common applications..."
$apps = @(
    @{path="C:\Windows\System32\notepad.exe"; required=$false},
    @{path="C:\Windows\explorer.exe"; required=$true}
)

foreach ($app in $apps) {
    try {
        if (Test-Path $app.path) {
            Start-Process $app.path -ErrorAction SilentlyContinue
            Write-Host "Started $($app.path)"
        } else {
            $message = if ($app.required) {
                "Error: Required application not found: $($app.path)"
            } else {
                "Warning: Could not find $($app.path)"
            }
            Write-Host $message
        }
    } catch {
        Write-Host ("Error starting {0}: {1}" -f $app.path, $_.Exception.Message)
    }
}

# Try to start Windows Terminal, fallback to PowerShell if not available
try {
    $wtPath = Get-Command "wt.exe" -ErrorAction SilentlyContinue
    if ($wtPath) {
        Write-Host "Starting Windows Terminal..."
        Start-Process "wt.exe"
    } else {
        Write-Host "Windows Terminal not found in PATH, checking Scoop installation..."
        $wtScoop = Join-Path $env:SCOOP "apps\windows-terminal\current\WindowsTerminal.exe"
        if (Test-Path $wtScoop) {
            Write-Host "Starting Windows Terminal from Scoop installation..."
            Start-Process $wtScoop
        } else {
            Write-Host "Windows Terminal not found, starting PowerShell instead..."
            Start-Process "powershell.exe"
        }
    }
} catch {
    Write-Host ("Error launching terminal: {0}" -f $_.Exception.Message)
    Write-Host "Falling back to PowerShell..."
    Start-Process "powershell.exe"
}

Write-Host "scoop_setup.ps1 script completed successfully."