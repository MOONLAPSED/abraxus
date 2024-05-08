powershell:
 - upgrade to release windows from the flashstick-version .iso (use shift f-10 for command prompt): 
 
       `https://www.microsoft.com/en-us/software-download/windows11` and use the .exe while offline to get new service pack instead of .iso-installed version.
 - download explorer-patcher for tolerable explorer UI:

       `https://github.com/valinet/ExplorerPatcher`
 - if it still doesn't work, hack the registry:

       `https://www.wisecleaner.com/think-tank/389-How-to-Enable-Explorer-Tabs-on-Windows-11-22H2.html`
 - install windows terminal, vscode-insiders, curl, hurl, scoop, micromamba, etc.

        included `.bat` is a good starting-point for customization
 - scoop applications

        `scoop.ps1` is a good starting-point for customization
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
 - `sudo apt install --fix-broken -y`
 - `wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.4/install.sh | bash`
 - `sudo apt install --fix-broken -y`
 - `mkdir chrome`
 - `cd chrome`
 - `sudo wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb`
 - `sudo dpkg -i google-chrome-stable_current_amd64.deb`