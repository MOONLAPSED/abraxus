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
    @{name="uv"; bucket="main"}
)

foreach ($package in $packages) {
    Install-ScoopPackage -packageName $package.name -bucket $package.bucket
}

# Update all installed applications
Write-Host "Updating all installed applications..."
scoop update *

# Configure Python environment using uv
Write-Host "Setting up Python environment using uv..."

# Get paths
$vscodeExePath = Join-Path $env:SCOOP "shims\code.exe"
$uvPath = Join-Path $env:SCOOP "shims\uv.exe"
$workspaceDir = Join-Path $env:USERPROFILE "dev-workspace"
# "C:\Users\WDAGUtilityAccount\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Scoop Apps\Visual Studio Code.lnk"

# Create workspace directory if it doesn't exist
if (-not (Test-Path $workspaceDir)) {
    New-Item -Path $workspaceDir -ItemType Directory -Force | Out-Null
    Write-Host "Created workspace directory at $workspaceDir"
}

# Change to workspace directory
Push-Location $workspaceDir

# Create a Python virtual environment using uv
if (Test-Path $uvPath) {
    Write-Host "Creating Python virtual environment using uv..."
    
    # This will create a .venv directory with a Python installation
    & $uvPath venv --seed
    
    # Get the path to the Python executable in the virtual environment
    $pythonVenvPath = Join-Path $workspaceDir ".venv\Scripts\python.exe"
    
    # Install some basic packages using uv
    if (Test-Path $pythonVenvPath) {
        Write-Host "Installing basic Python packages..."
        & $uvPath pip install pytest requests black pylint jupyter
        Write-Host "Python environment setup complete!"
    } else {
        Write-Host "Warning: Python virtual environment not created successfully!"
    }
} else {
    Write-Host "Warning: uv executable not found. Skipping Python setup."
}

# Return to original directory
Pop-Location

# Configure VSCode with necessary extensions and settings
Write-Host "Configuring VSCode..."

# Install VSCode extensions
if (Test-Path $vscodeExePath) {
    Write-Host "Installing VSCode extensions..."
    & $vscodeExePath --install-extension ms-python.python --force
    & $vscodeExePath --install-extension ms-python.vscode-pylance --force
    
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
    
    # Point VSCode to our uv-managed Python in the virtual environment
    $pythonVenvPath = Join-Path $workspaceDir ".venv\Scripts\python.exe"
    
    # Update Python settings
    $settingsHash["python.defaultInterpreterPath"] = $pythonVenvPath
    $settingsHash["python.terminal.activateEnvironment"] = $true
    $settingsHash["terminal.integrated.env.windows"] = @{
        "PATH" = "$env:SCOOP\shims;$env:PATH"
    }
    $settingsHash["python.venvPath"] = $workspaceDir
    
    # Create a launch.json template in the workspace directory
    $launchDirPath = Join-Path $workspaceDir ".vscode"
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
                "justMyCode" = $true
                "python" = $pythonVenvPath
                "env" = @{
                    "PATH" = "$env:SCOOP\shims;$env:PATH"
                }
            }
        )
    } | ConvertTo-Json -Depth 5
    
    $launchContent | Out-File -FilePath $launchPath -Encoding utf8 -Force
    
    # Create a workspace settings directory
    $workspaceSettingsDir = Join-Path $workspaceDir ".vscode"
    if (-not (Test-Path $workspaceSettingsDir)) {
        New-Item -Path $workspaceSettingsDir -ItemType Directory -Force | Out-Null
    }
    
    # Create workspace-specific settings.json
    $workspaceSettingsPath = Join-Path $workspaceSettingsDir "settings.json"
    $workspaceSettings = @{
        "python.defaultInterpreterPath" = $pythonVenvPath
        "python.analysis.extraPaths" = @("${workspaceDir}\.venv\Lib\site-packages")
        "python.terminal.activateEnvironment" = $true
        "python.linting.enabled" = $true
        "python.linting.pylintEnabled" = $true
        "python.formatting.provider" = "black"
    } | ConvertTo-Json -Depth 5
    
    $workspaceSettings | Out-File -FilePath $workspaceSettingsPath -Encoding utf8 -Force
    
    # Convert global settings hashtable back to JSON and save
    $settingsHash | ConvertTo-Json -Depth 5 | Out-File -FilePath $settingsPath -Encoding utf8 -Force
    
    Write-Host "VSCode configuration completed."
} else {
    Write-Host "Warning: VSCode executable not found. Skipping configuration."
}

# Create helper scripts for Python development with uv
Write-Host "Creating helper scripts for Python development..."
if (Test-Path $uvPath) {
    # Create a detailed helper batch file for working with uv
    $uvSetupScript = @"
@echo off
echo ===== Python Development Environment Setup =====
echo.
echo Setting up uv environment in PATH...
set PATH=$env:SCOOP\shims;%PATH%
set WORKSPACE_DIR=$workspaceDir

echo.
echo Python environment information:
echo - Workspace: %WORKSPACE_DIR%
echo - uv path: $uvPath
echo - Python path: %WORKSPACE_DIR%\.venv\Scripts\python.exe
echo.
echo Common uv commands:
echo 1. Create new project:  uv venv --seed my-project
echo 2. Install package:     uv pip install package_name
echo 3. Install from file:   uv pip install -r requirements.txt
echo 4. Run Python:          %WORKSPACE_DIR%\.venv\Scripts\python.exe your_script.py
echo.
echo === VSCode is configured to use the Python environment ===
echo.
"@
    
    $uvSetupPath = Join-Path $env:USERPROFILE "python_dev_setup.bat"
    $uvSetupScript | Out-File -FilePath $uvSetupPath -Encoding ascii -Force
    
    # Create a simple Python test script in the workspace
    $testScriptPath = Join-Path $workspaceDir "hello.py"
    $testScript = @"
# Simple test script to verify Python environment
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {platform.platform()}")

# Try to import installed modules
try:
    import requests
    print("✓ requests module is installed")
except ImportError:
    print("✗ requests module is not installed")

try:
    import pytest
    print("✓ pytest module is installed")
except ImportError:
    print("✗ pytest module is not installed")

print("\nEverything is working! You're ready to start developing with Python.")
"@
    
    $testScript | Out-File -FilePath $testScriptPath -Encoding utf8 -Force
    
    Write-Host "Created Python development helpers:"
    Write-Host "  - Setup guide: $uvSetupPath"
    Write-Host "  - Test script: $testScriptPath"
}

# Launch common applications
Write-Host "Launching common applications..."
$apps = @(
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

# Launch VSCode as the last step with the workspace folder
if (Test-Path $vscodeExePath) {
    Write-Host "Launching VSCode with workspace..."
    Start-Process $vscodeExePath -ArgumentList "$workspaceDir"
}

Write-Host "scoop_setup.ps1 script completed successfully."