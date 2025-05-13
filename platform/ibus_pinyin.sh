#!/bin/bash

# ===========================
# Comprehensive IBus Pinyin Setup for Kali Linux
# ===========================

# Ensure the script is run as root or with sudo
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root or with sudo."
    exit 1
fi

# Step 1: Update package lists
echo "Updating package lists..."
apt update

# Step 2: Install required packages
echo "Installing IBus and Pinyin support..."
apt install -y ibus ibus-pinyin ibus-gtk ibus-gtk3 ibus-qt4 ibus-qt5

# Step 3: Configure environment variables
echo "Configuring environment variables for IBus..."

# Add IBus configuration to ~/.xprofile (or create it if it doesn't exist)
cat <<EOF >> /etc/profile.d/ibus.sh
export GTK_IM_MODULE=ibus
export QT_IM_MODULE=ibus
export XMODIFIERS=@im=ibus
EOF

# Make the script executable
chmod +x /etc/profile.d/ibus.sh

# Step 4: Start IBus daemon
echo "Starting IBus daemon..."
ibus-daemon --xim --daemonize --replace

# Step 5: Add Pinyin input method to IBus
echo "Adding Pinyin input method to IBus..."

# Create a default IBus configuration file if it doesn't exist
mkdir -p ~/.config/ibus
cat <<EOF > ~/.config/ibus/bus/default.conf
[General]
UseCustomSymbolTable=False
DefaultInputMethod=pinyin
EOF

# Step 6: Restart the session
echo "Setup complete!"
echo "Please log out and log back in for changes to take effect."