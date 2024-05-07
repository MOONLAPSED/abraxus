powershell:
 - `Invoke-WebRequest -Uri https://aka.ms/wslubuntu2204 -OutFile Ubuntu.appx -UseBasicParsing`
 - `Add-AppxPackage Ubuntu.appx`
 - `wsl --update`
 - `wsl --set-default-version 2`
 - `wsl --install -d Ubuntu-22.04`
 - `wsl --setdefault Ubuntu-22.04`

wsl:
 - `cd ~`
 - `sudo apt update -y && sudo apt full-upgrade -y && sudo apt autoremove -y && sudo apt clean -y && sudo apt autoclean -y`
 - `sudo apt-get install build-essential && sudo apt-get install manpages-dev`
 - `sudo apt install build-essential libglvnd-dev pkg-config`
 - 
