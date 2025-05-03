@echo off
:: Windows-safe way to invoke the PowerShell script with bypass
:: We 'miss the timing' of the sandbox startup-system
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0scoop_setup.ps1"
exit /b %ERRORLEVEL%
