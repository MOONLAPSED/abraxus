@echo off
REM VSCode Extension Installer - Run this inside the Sandbox if extensions aren't installed automatically
echo VSCode Python Extension Installer
echo ==============================

echo Detecting VSCode installation locations...

set FOUND_VSCODE=0
set "POTENTIAL_PATHS=C:\Users\WDAGUtilityAccount\scoop\shims\code.exe C:\Users\WDAGUtilityAccount\scoop\apps\vscode\current\Code.exe"

for %%p in (%POTENTIAL_PATHS%) do (
    if exist "%%p" (
        echo Found VSCode at: %%p
        set VSCODE_PATH=%%p
        set FOUND_VSCODE=1
        goto :found_vscode
    )
)

:search_scoop
echo Searching in Scoop directory...
for /r "C:\Users\WDAGUtilityAccount\scoop\apps" %%f in (Code.exe) do (
    echo Found VSCode at: %%f
    set VSCODE_PATH=%%f
    set FOUND_VSCODE=1
    goto :found_vscode
)

:found_vscode
if %FOUND_VSCODE% EQU 0 (
    echo ERROR: Could not find VSCode installation.
    pause
    exit /b 1
)

echo Found VSCode at: %VSCODE_PATH%
echo Installing Python extension...

"%VSCODE_PATH%" --install-extension ms-python.python --force
timeout /t 5

echo Installing Pylance extension...
"%VSCODE_PATH%" --install-extension ms-python.vscode-pylance --force
timeout /t 5

echo Verifying installed extensions...
"%VSCODE_PATH%" --list-extensions

echo.
echo Installation completed!
echo If you still don't see the extensions in VSCode, try restarting VSCode.
echo.
pause