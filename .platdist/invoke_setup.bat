@echo off
REM This batch file will install Scoop and run the setup script

echo Setting up Windows Sandbox development environment...

REM Install Scoop if not already installed
if not exist C:\Users\WDAGUtilityAccount\scoop (
    echo Installing Scoop...
    powershell -Command "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')"
    if %ERRORLEVEL% neq 0 (
        echo Failed to install Scoop
        pause
        exit /b 1
    )
) else (
    echo Scoop already installed
)

REM Run the PowerShell setup script
echo Running scoop_setup.ps1...
powershell -ExecutionPolicy Bypass -File "%~dp0scoop_setup.ps1"

echo Setup completed!
exit /b 0