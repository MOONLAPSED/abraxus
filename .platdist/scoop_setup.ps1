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
    @{name="vscode"; bucket="extras"},
    @{name="windows-terminal"; bucket="extras"},
    @{name="uv"; bucket="main"},
    @{name="notepadplusplus"; bucket="extras"},
    @{name="python"; bucket="main"}
)

foreach ($package in $packages) {
    Install-ScoopPackage -packageName $package.name -bucket $package.bucket
}

# Update all installed applications
Write-Host "Updating all installed applications..."
scoop update *

# Configure VSCode with necessary extensions and settings
Write-Host "Configuring VSCode..."

# Get paths
$vscodeExePath = Join-Path $env:SCOOP "shims\code.exe"
$uvPath = Join-Path $env:SCOOP "shims\uv.exe"
$pythonPath = Join-Path $env:SCOOP "shims\python.exe"

# Ensure we use portable python even if it has a different executable path
if (-not (Test-Path $pythonPath)) {
    $pythonPath = Join-Path $env:SCOOP "apps\python-portable\current\python.exe"
}

# Install VSCode extensions
if (Test-Path $vscodeExePath) {
    Write-Host "Installing VSCode extensions..."
    # Install extensions with timeout to prevent hanging
    $timeout = 60 # seconds
    $job = Start-Job -ScriptBlock { 
        & $using:vscodeExePath --install-extension ms-python.python --force 
    }
    if (Wait-Job $job -Timeout $timeout) {
        Receive-Job $job
    } else {
        Write-Host "Warning: Installing Python extension timed out, but VSCode may still work"
        Stop-Job $job
    }
    Remove-Job $job -Force
    
    $job = Start-Job -ScriptBlock { 
        & $using:vscodeExePath --install-extension ms-python.vscode-pylance --force 
    }
    if (Wait-Job $job -Timeout $timeout) {
        Receive-Job $job
    } else {
        Write-Host "Warning: Installing Pylance extension timed out, but VSCode may still work"
        Stop-Job $job
    }
    Remove-Job $job -Force
    
    # Create VSCode settings directory if it doesn't exist
    $vscodeDirPath = Join-Path $env:APPDATA "Code\User"
    if (-not (Test-Path $vscodeDirPath)) {
        New-Item -Path $vscodeDirPath -ItemType Directory -Force | Out-Null
    }
    
    # Create/update settings.json for VSCode
    $settingsPath = Join-Path $vscodeDirPath "settings.json"
    
    # Check if settings file exists and read it, otherwise create new settings object
    if (Test-Path $settingsPath) {
        try {
            $settings = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
        } catch {
            $settings = [PSCustomObject]@{}
        }
    } else {
        $settings = [PSCustomObject]@{}
    }
    
    # Convert PSCustomObject to hashtable for easier manipulation
    $settingsHash = @{}
    if ($settings.PSObject.Properties) {
        foreach ($property in $settings.PSObject.Properties) {
            $settingsHash[$property.Name] = $property.Value
        }
    }
    
    # Update Python settings
    $settingsHash["python.defaultInterpreterPath"] = $pythonPath
    $settingsHash["python.terminal.activateEnvironment"] = $true
    $settingsHash["terminal.integrated.env.windows"] = @{
        "PATH" = "$env:SCOOP\shims;$env:PATH"
    }
    
    # Create an launch.json template in user's home directory to be used as reference
    $launchDirPath = Join-Path $env:USERPROFILE ".vscode"
    if (-not (Test-Path $launchDirPath)) {
        New-Item -Path $launchDirPath -ItemType Directory -Force | Out-Null
    }
    
    $launchPath = Join-Path $launchDirPath "launch.json"
    $launchContent = @{
        "version" = "0.2.0"
        "configurations" = @(
            @{
                "name" = "Python: Current File"
                "type" = "python"
                "request" = "launch"
                "program" = "`${file}"
                "console" = "integratedTerminal"
                "env" = @{
                    "PATH" = "$env:SCOOP\shims;$env:PATH"
                }
            }
        )
    } | ConvertTo-Json -Depth 5
    
    $launchContent | Out-File -FilePath $launchPath -Encoding utf8 -Force
    
    # Convert settings hashtable back to JSON and save
    $settingsHash | ConvertTo-Json -Depth 5 | Out-File -FilePath $settingsPath -Encoding utf8 -Force
    
    Write-Host "VSCode configuration completed."
} else {
    Write-Host "Warning: VSCode executable not found. Skipping configuration."
}

# Setup uv for Python package management
Write-Host "Configuring uv as Python package manager..."
if (Test-Path $uvPath) {
    # Create a batch file that helps configure uv in VSCode terminal
    $uvSetupScript = @"
@echo off
echo Setting up uv environment in PATH...
set PATH=$env:SCOOP\shims;%PATH%
echo uv path: $uvPath
echo Python path: $pythonPath
echo.
echo To use uv in your projects:
echo 1. Use 'uv venv' to create virtual environments
echo 2. Use 'uv pip install package_name' to install packages
echo.
"@
    
    $uvSetupPath = Join-Path $env:USERPROFILE "setup_uv.bat"
    $uvSetupScript | Out-File -FilePath $uvSetupPath -Encoding ascii -Force
    
    Write-Host "Created uv setup helper at $uvSetupPath"
}

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

# Launch VSCode as the last step
if (Test-Path $vscodeExePath) {
    Write-Host "Launching VSCode..."
    Start-Process $vscodeExePath
}

Write-Host "scoop_setup.ps1 script completed successfully."