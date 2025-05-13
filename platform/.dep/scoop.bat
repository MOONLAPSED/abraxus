:: This batch file sets up the environment for the sandbox init process
:: It adds Scoop to the PATH and launches the rollout script.

@echo off
echo Starting sandbox initialization...

:: Wait for Scoop installation to complete before proceeding
echo Waiting for Scoop installation to complete...
timeout /t 15

:: Add Scoop to PATH for this session 
set PATH=%USERPROFILE%\scoop\shims;%PATH%

:: Display current PATH for debugging
echo Current PATH: %PATH%

:: Check if Scoop is installed and accessible
where scoop >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Scoop is installed and accessible.
    echo Installing git...
    scoop install git
    
    :: Check if the destination script exists before executing
    IF EXIST "C:\Users\WDAGUtilityAccount\Desktop\abraxus\scoop.ps1" (
        echo Running scoop.ps1 script...
        powershell.exe -ExecutionPolicy Bypass -File "C:\Users\WDAGUtilityAccount\Desktop\abraxus\scoop.ps1"
    ) ELSE (
        echo ERROR: scoop.ps1 not found at the expected location.
        dir "C:\Users\WDAGUtilityAccount\Desktop\abraxus"
    )
) ELSE (
    echo Scoop is not installed or not in PATH. Installing Scoop...
    powershell.exe -ExecutionPolicy Bypass -Command "iex (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')"
    
    :: Wait for installation to complete
    timeout /t 10
    
    :: Try again after installation
    goto check_scoop
)

:check_scoop
where scoop >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Scoop is now installed. Running setup...
    scoop install git
    powershell.exe -ExecutionPolicy Bypass -File "C:\Users\WDAGUtilityAccount\Desktop\abraxus\scoop.ps1"
) ELSE (
    echo Failed to install Scoop. Please check manually.
)

echo Batch file execution completed.