@echo off
:: Windows-safe way to invoke the PowerShell script with bypass
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0scoop_setup.ps1"
exit /b %ERRORLEVEL%
