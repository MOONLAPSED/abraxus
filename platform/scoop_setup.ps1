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

# CONSOLIDATED PATH DETECTION FOR VSCODE - Only find it once
$vscodeExePath = $null
Write-Host "Locating VSCode installation..."

# Try different potential paths for VSCode
$potentialVSCodePaths = @(
    (Join-Path $env:SCOOP "shims\code.exe"),
    (Join-Path $env:SCOOP "apps\vscode\current\Code.exe"),
    (Join-Path $env:SCOOP "apps\vscode-portable\current\Code.exe")
)

foreach ($path in $potentialVSCodePaths) {
    if (Test-Path $path) {
        $vscodeExePath = $path
        Write-Host "Found VSCode at: $vscodeExePath"
        break
    }
}

# If VSCode wasn't found in the predefined paths, try to find it using Scoop's info
if (-not $vscodeExePath) {
    try {
        $scoopInfo = scoop info vscode
        $scoopInfoStr = $scoopInfo -join "`n"
        if ($scoopInfoStr -match "Installed: (.+)") {
            $vscodePath = Join-Path $env:SCOOP "apps\vscode\$($matches[1])\Code.exe"
            if (Test-Path $vscodePath) {
                $vscodeExePath = $vscodePath
                Write-Host "Found VSCode via scoop info at: $vscodeExePath"
            }
        }
    } catch {
        Write-Host "Could not get VSCode info from scoop: $_"
    }
}

# Last resort - search for it in the Scoop apps directory
if (-not $vscodeExePath) {
    $scoopAppsDir = Join-Path $env:SCOOP "apps"
    if (Test-Path $scoopAppsDir) {
        $vscodeExeFiles = Get-ChildItem -Path $scoopAppsDir -Recurse -Filter "Code.exe" -ErrorAction SilentlyContinue
        if ($vscodeExeFiles -and $vscodeExeFiles.Count -gt 0) {
            $vscodeExePath = $vscodeExeFiles[0].FullName
            Write-Host "Found VSCode by searching Scoop directory: $vscodeExePath"
        }
    }
}

if (-not $vscodeExePath) {
    Write-Warning "VSCode executable not found. VSCode configuration and launch will be skipped."
}

# Set up UV and workspace paths
$uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uvPath) {
    $uvPath = Join-Path $env:SCOOP "shims\uv.exe"
}
$workspaceDir = Join-Path $env:USERPROFILE "dev-workspace"

# Create workspace directory if it doesn't exist
if (-not (Test-Path $workspaceDir)) {
    New-Item -Path $workspaceDir -ItemType Directory -Force | Out-Null
    Write-Host "Created workspace directory at $workspaceDir"
}

# Change to workspace directory
Push-Location $workspaceDir

# Create a Python virtual environment using uv
$pythonVenvPath = Join-Path $workspaceDir ".venv\Scripts\python.exe"

if (Test-Path $uvPath) {
    Write-Host "Creating Python virtual environment using uv..."
    
    # This will create a .venv directory with a Python installation
    & $uvPath venv --seed
    
    # Install some basic packages using uv
    if (Test-Path $pythonVenvPath) {
        Write-Host "Installing basic Python packages..."
        & $uvPath pip install pytest requests black pylint jupyter
        Write-Host "Python environment setup complete!"
    } else {
        Write-Host "Warning: Python virtual environment not created successfully!"
    }
} else {
    Write-Host "Warning: uv executable not found at $uvPath. Trying to find it..."
    $uvExeFiles = Get-ChildItem -Path $env:SCOOP -Recurse -Filter "uv.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($uvExeFiles) {
        $uvPath = $uvExeFiles.FullName
        Write-Host "Found uv.exe at: $uvPath"
        
        # Try again with the found path
        & $uvPath venv --seed
        if (Test-Path $pythonVenvPath) {
            Write-Host "Installing basic Python packages..."
            & $uvPath pip install pytest requests black pylint jupyter
            Write-Host "Python environment setup complete!"
        }
    } else {
        Write-Host "Could not find uv.exe in Scoop installation."
    }
}

# Return to original directory
Pop-Location

# Configure VSCode ONLY if we found the executable
if ($vscodeExePath -and (Test-Path $vscodeExePath)) {
    Write-Host "Configuring VSCode..."
    
    # Install VSCode extensions
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
    
    # Create workspace-specific settings.json
    $workspaceSettingsPath = Join-Path $launchDirPath "settings.json"
    $workspaceSettings = @{
        "python.defaultInterpreterPath" = $pythonVenvPath
        "python.analysis.extraPaths" = @("${workspaceDir}\.venv\Lib\site-packages")
        "python.terminal.activateEnvironment" = $true
        "python.linting.enabled" = $true
        "python.linting.pylintEnabled" = $true
        "python.formatting.provider" = "black"
        "terminal.integrated.env.windows" = @{
            "PATH" = "$env:SCOOP\shims;$env:PATH"
        }
    } | ConvertTo-Json -Depth 5
    
    $workspaceSettings | Out-File -FilePath $workspaceSettingsPath -Encoding utf8 -Force
    
    # Convert global settings hashtable back to JSON and save
    $settingsHash | ConvertTo-Json -Depth 5 | Out-File -FilePath $settingsPath -Encoding utf8 -Force
    
    Write-Host "VSCode configuration completed."
} else {
    Write-Host "Skipping VSCode configuration - executable not found."
}

# Create workspace extensions recommendation
$vscodeWorkspaceVscode = Join-Path $workspaceDir ".vscode"
if (-not (Test-Path $vscodeWorkspaceVscode)) {
    New-Item -ItemType Directory -Path $vscodeWorkspaceVscode | Out-Null
}
$recommendationsJson = @{
    "recommendations" = @(
        "ms-python.python",
        "ms-python.vscode-pylance"
    )
} | ConvertTo-Json -Depth 2
$recommendationsJson | Out-File (Join-Path $vscodeWorkspaceVscode "extensions.json") -Encoding UTF8

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
    
    # Create VSCode workspace file
    $workspaceFilePath = Join-Path $workspaceDir "python-dev.code-workspace"
    $workspaceFileContent = @{
        "folders" = @(
            @{
                "path" = "."
            }
        )
        "settings" = @{
            "python.defaultInterpreterPath" = "${workspaceDir}\.venv\Scripts\python.exe"
            "python.analysis.extraPaths" = @("${workspaceDir}\.venv\Lib\site-packages")
            "python.terminal.activateEnvironment" = $true
            "python.linting.enabled" = $true
            "python.linting.pylintEnabled" = $true
            "python.formatting.provider" = "black"
            "terminal.integrated.env.windows" = @{
                "PATH" = "$env:SCOOP\shims;$env:PATH"
            }
        }
    } | ConvertTo-Json -Depth 5
    
    $workspaceFileContent | Out-File -FilePath $workspaceFilePath -Encoding utf8 -Force
    
    Write-Host "Created Python development helpers:"
    Write-Host "  - Setup guide: $uvSetupPath"
    Write-Host "  - Test script: $testScriptPath" 
    Write-Host "  - VSCode workspace: $workspaceFilePath"
}

# Verify uv/python workspace is in $PATH
$env:PATH = "$($workspaceDir)\.venv\Scripts;$env:PATH"
if (-not (Test-Path $pythonVenvPath)) {
    Write-Error "ERROR: Virtual environment was not correctly created!"
    exit 1
}

# SINGLE VSCode launch at the end - only if VSCode was found and configured
if ($vscodeExePath -and (Test-Path $vscodeExePath)) {
    Write-Host "Launching VSCode with workspace..."
    
    # Wait a moment to ensure all file operations are complete
    Start-Sleep -Seconds 2
    
    # Use the workspace file instead of just the directory
    $workspaceFilePath = Join-Path $workspaceDir "python-dev.code-workspace"
    if (Test-Path $workspaceFilePath) {
        # Use Start-Process with -PassThru to get a handle, but don't wait for it
        $vscodeProcess = Start-Process $vscodeExePath -ArgumentList "--reuse-window `"$workspaceFilePath`"" -PassThru
        Write-Host "VSCode launched with PID: $($vscodeProcess.Id)"
    } else {
        # Fallback to directory launch
        $vscodeProcess = Start-Process $vscodeExePath -ArgumentList "--reuse-window `"$workspaceDir`"" -PassThru  
        Write-Host "VSCode launched with directory, PID: $($vscodeProcess.Id)"
    }
} else {
    Write-Host "VSCode not launched - executable not found or not configured."
}

Write-Host "scoop_setup.ps1 script completed successfully."