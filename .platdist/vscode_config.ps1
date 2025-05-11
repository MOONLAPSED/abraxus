Write-Host "Configuring VSCode for uv and Python..."

# Variables
$scoopRoot = "C:\Users\WDAGUtilityAccount\scoop"
$uvPath = "$scoopRoot\apps\uv\current\uv.exe"
$vscodePath = "$scoopRoot\apps\vscode\current\code.exe"
$vscodeUserDir = "$env:APPDATA\Code\User"
$settingsPath = "$vscodeUserDir\settings.json"

# Ensure the VSCode user directory exists
if (-not (Test-Path $vscodeUserDir)) {
    Write-Host "Creating VSCode user directory..."
    New-Item -Path $vscodeUserDir -ItemType Directory -Force | Out-Null
}

# Create or update VSCode settings.json
Write-Host "Updating VSCode settings.json to include uv path..."
$settings = @{
    "python.defaultInterpreterPath" = $uvPath
    "python.terminal.activateEnvironment" = $true
    "python.terminal.activateEnvInCurrentTerminal" = $true
    "terminal.integrated.profiles.windows" = @{
        "PowerShell" = @{
            "source" = "PowerShell"
            "icon" = "terminal-powershell"
            "env" = @{
                "PATH" = "$scoopRoot\shims;$env:PATH"
            }
        }
    }
    "terminal.integrated.defaultProfile.windows" = "PowerShell"
}

# Check if settings file already exists
if (Test-Path $settingsPath) {
    # Load existing settings
    try {
        $existingSettings = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json
        # Convert to hashtable
        $existingSettingsHash = @{}
        $existingSettings.PSObject.Properties | ForEach-Object {
            $existingSettingsHash[$_.Name] = $_.Value
        }
        # Merge with new settings
        foreach ($key in $settings.Keys) {
            $existingSettingsHash[$key] = $settings[$key]
        }
        $settings = $existingSettingsHash
    } catch {
        Write-Host "Error reading existing settings: $_"
        # Proceed with new settings if there's an error
    }
}

# Save the settings
$settings | ConvertTo-Json -Depth 10 | Set-Content -Path $settingsPath -Force
Write-Host "VSCode settings updated successfully."

# Install Python extension
Write-Host "Installing Python extension for VSCode..."
try {
    # Check if VSCode executable exists
    if (Test-Path $vscodePath) {
        # Install Python extension
        & $vscodePath --install-extension ms-python.python --force

        # Add additional useful extensions
        & $vscodePath --install-extension ms-python.vscode-pylance --force
        & $vscodePath --install-extension visualstudioexptteam.vscodeintellicode --force
        
        Write-Host "VSCode extensions installed successfully."
    } else {
        Write-Error "VSCode executable not found at: $vscodePath"
    }
} catch {
    Write-Error "Failed to install VSCode extensions: $_"
}

# Create a workspace file (optional)
$workspacePath = "C:\Users\WDAGUtilityAccount\Desktop\dev-workspace.code-workspace"
$workspace = @{
    "folders" = @(
        @{
            "path" = "C:\Users\WDAGUtilityAccount"
        }
    )
    "settings" = @{
        "python.defaultInterpreterPath" = $uvPath
    }
}

$workspace | ConvertTo-Json -Depth 5 | Set-Content -Path $workspacePath -Force
Write-Host "VS Code workspace file created at: $workspacePath"

Write-Host "VSCode configuration completed."