@echo off
:: Setup log file
set LOGFILE=C:\Users\WDAGUtilityAccount\Desktop\sandbox_log.txt
echo [%TIME%] Running provisioner.bat in Windows Sandbox... > "%LOGFILE%" 2>&1

:: Set current directory to script location
cd /d "%~dp0"
echo [%TIME%] Current directory: %CD% >> "%LOGFILE%"

:: Scoop environment variables
set "SANDBOX_USER=WDAGUtilityAccount"
set "SCOOP=C:\Users\%SANDBOX_USER%\scoop"
set "SCOOP_SHIMS=%SCOOP%\shims"
echo [%TIME%] Using Scoop path: %SCOOP% >> "%LOGFILE%"

:: Add Scoop shims to PATH for this session
set "PATH=%PATH%;%SCOOP_SHIMS%"
echo [%TIME%] PATH after adding Scoop: %PATH% >> "%LOGFILE%"

:: Persist scoop shims into the user PATH (so any new process sees it)
echo [%TIME%] Persisting scoop\shims into User PATH... >> "%LOGFILE%"
setx PATH "%PATH%" >> "%LOGFILE%" 2>&1

:: Check if Scoop is installed
where scoop >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo [%TIME%] Scoop is already installed. >> "%LOGFILE%"
) ELSE (
    echo [%TIME%] Scoop is not installed. Installing Scoop... >> "%LOGFILE%"

    powershell.exe -ExecutionPolicy Bypass -NoProfile -Command ^
        "$env:SCOOP='C:\Users\WDAGUtilityAccount\scoop'; [Environment]::SetEnvironmentVariable('SCOOP', $env:SCOOP, 'User'); iwr -useb get.scoop.sh | iex" >> "%LOGFILE%" 2>&1

    timeout /t 5 >nul

    where scoop >nul 2>&1
    IF %ERRORLEVEL% EQU 0 (
        echo [%TIME%] Scoop installed successfully. >> "%LOGFILE%"
    ) ELSE (
        echo [%TIME%] ERROR: Scoop installation failed. >> "%LOGFILE%"
        exit /b 1
    )
)

:: Run the wrapper to safely execute the PowerShell setup script
echo [%TIME%] Invoking invoke_setup.bat to launch scoop_setup.ps1... >> "%LOGFILE%"
call "%~dp0invoke_setup.bat" >> C:\Users\WDAGUtilityAccount\Desktop\scoop_setup_log.txt 2>&1
set PS1_RESULT=%ERRORLEVEL%

IF %PS1_RESULT% NEQ 0 (
    echo [%TIME%] ERROR: scoop_setup.ps1 failed with code %PS1_RESULT% >> "%LOGFILE%"
    exit /b %PS1_RESULT%
)

echo [%TIME%] Success: Provisioning complete. >> "%LOGFILE%"
start notepad.exe "C:\Users\WDAGUtilityAccount\Desktop\sandbox_log.txt"
exit /b 0
