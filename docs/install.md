
 - Invoke-WebRequest -Uri https://aka.ms/wslubuntu2204 -OutFile Ubuntu.appx -UseBasicParsing
 - Add-AppxPackage Ubuntu.appx
 - wsl --update
 - wsl --set-default-version 2
 - wsl --install -d Ubuntu-22.04
 - wsl --set-default Ubuntu-22.04