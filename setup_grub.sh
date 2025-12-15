#!/bin/bash
# Set GRUB default boot entry to 6.0-lake kernel and add kernel parameters

set -e  # Exit on error

echo "=== Step 1: Extract advanced menu ID and 6.0-lake kernel ID ==="

# Extract advanced menu ID (extract content from last single quotes in submenu line, usually the ID)
SUBMENU_LINE=$(cat /boot/grub/grub.cfg | grep submenu | head -1)
if [ -z "$SUBMENU_LINE" ]; then
    echo "Error: Unable to find submenu line"
    exit 1
fi
echo "Found submenu line: $SUBMENU_LINE"

# Extract ID (usually content from last single quotes, format like gnulinux-advanced-...)
ADVANCED_MENU_ID=$(echo "$SUBMENU_LINE" | grep -o "'[^']*'" | tail -1 | tr -d "'")
if [ -z "$ADVANCED_MENU_ID" ] || [[ ! "$ADVANCED_MENU_ID" =~ ^gnulinux- ]]; then
    # If the last one is not an ID, try to find one containing gnulinux-advanced
    ADVANCED_MENU_ID=$(echo "$SUBMENU_LINE" | grep -o "gnulinux-advanced-[^' ]*" | head -1)
fi
if [ -z "$ADVANCED_MENU_ID" ]; then
    echo "Error: Unable to extract advanced menu ID from submenu line"
    exit 1
fi
echo "Advanced menu ID: $ADVANCED_MENU_ID"

# Extract 6.0-lake kernel ID (as per user instructions: grep option | grep 6.0.0-lake)
OPTION_LAKE_LINE=$(cat /boot/grub/grub.cfg | grep option | grep "6.0.*lake" | head -1)
if [ -z "$OPTION_LAKE_LINE" ]; then
    echo "Error: Unable to find line containing 'option' and '6.0.*lake'"
    echo "Trying to find menuentry line containing '6.0' and 'lake'..."
    OPTION_LAKE_LINE=$(cat /boot/grub/grub.cfg | grep -i "6.0.*lake" | head -1)
fi
if [ -z "$OPTION_LAKE_LINE" ]; then
    echo "Error: Unable to find 6.0-lake kernel related line"
    exit 1
fi
echo "Found 6.0-lake line: $OPTION_LAKE_LINE"

# Extract ID (find ID containing gnulinux and lake)
LAKE_KERNEL_ID=$(echo "$OPTION_LAKE_LINE" | grep -o "gnulinux-[^' ]*lake[^' ]*" | head -1)
if [ -z "$LAKE_KERNEL_ID" ]; then
    # Try to extract from last single quotes
    LAKE_KERNEL_ID=$(echo "$OPTION_LAKE_LINE" | grep -o "'[^']*'" | tail -1 | tr -d "'")
fi
if [ -z "$LAKE_KERNEL_ID" ]; then
    echo "Error: Unable to extract 6.0-lake kernel ID"
    exit 1
fi
echo "6.0-lake kernel ID: $LAKE_KERNEL_ID"

echo ""
echo "=== Step 2: Combine menu IDs ==="
GRUB_DEFAULT_VALUE="${ADVANCED_MENU_ID}>${LAKE_KERNEL_ID}"
echo "Combined GRUB_DEFAULT value: $GRUB_DEFAULT_VALUE"

echo ""
echo "=== Step 3: Update /etc/default/grub - Add GRUB_DEFAULT ==="

# Backup original file
sudo cp /etc/default/grub /etc/default/grub.backup.$(date +%Y%m%d_%H%M%S)
echo "Backed up /etc/default/grub"

# Check if GRUB_DEFAULT already exists
if grep -q "^GRUB_DEFAULT=" /etc/default/grub; then
    # If exists, update it
    echo "Updating existing GRUB_DEFAULT..."
    sudo sed -i "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"${GRUB_DEFAULT_VALUE}\"|" /etc/default/grub
else
    # If not exists, add it at the top of the file
    echo "Adding GRUB_DEFAULT at the top of the file..."
    sudo sed -i "1i GRUB_DEFAULT=\"${GRUB_DEFAULT_VALUE}\"" /etc/default/grub
fi

echo ""
echo "=== Step 4: Update GRUB_CMDLINE_LINUX_DEFAULT ==="

KERNEL_PARAMS="cma=128M@0-4G log_buf_len=16M"

# Check if GRUB_CMDLINE_LINUX_DEFAULT exists
if grep -q "^GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub; then
    # If exists, check if it already contains these parameters
    CURRENT_CMDLINE=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub | sed 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/\1/')
    
    # Check if parameters need to be added
    if echo "$CURRENT_CMDLINE" | grep -q "cma=128M@0-4G" && echo "$CURRENT_CMDLINE" | grep -q "log_buf_len=16M"; then
        echo "Kernel parameters already exist, no update needed"
    else
        echo "Updating GRUB_CMDLINE_LINUX_DEFAULT, adding new parameters..."
        # Remove quotes, add new parameters, then re-add quotes
        NEW_CMDLINE="${CURRENT_CMDLINE} ${KERNEL_PARAMS}"
        sudo sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${NEW_CMDLINE}\"|" /etc/default/grub
    fi
else
    # If not exists, create it
    echo "Creating GRUB_CMDLINE_LINUX_DEFAULT..."
    # Add after GRUB_DEFAULT
    sudo sed -i "/^GRUB_DEFAULT=/a GRUB_CMDLINE_LINUX_DEFAULT=\"${KERNEL_PARAMS}\"" /etc/default/grub
fi

echo ""
echo "=== Step 5: Update GRUB_TIMEOUT_STYLE and GRUB_TIMEOUT ==="

# Update GRUB_TIMEOUT_STYLE
if grep -q "^GRUB_TIMEOUT_STYLE=" /etc/default/grub; then
    echo "Updating existing GRUB_TIMEOUT_STYLE..."
    sudo sed -i "s|^GRUB_TIMEOUT_STYLE=.*|GRUB_TIMEOUT_STYLE=show|" /etc/default/grub
else
    echo "Creating GRUB_TIMEOUT_STYLE..."
    # Add after GRUB_CMDLINE_LINUX_DEFAULT
    sudo sed -i "/^GRUB_CMDLINE_LINUX_DEFAULT=/a GRUB_TIMEOUT_STYLE=show" /etc/default/grub
fi

# Update GRUB_TIMEOUT
if grep -q "^GRUB_TIMEOUT=" /etc/default/grub; then
    echo "Updating existing GRUB_TIMEOUT..."
    sudo sed -i "s|^GRUB_TIMEOUT=.*|GRUB_TIMEOUT=5|" /etc/default/grub
else
    echo "Creating GRUB_TIMEOUT..."
    # Add after GRUB_TIMEOUT_STYLE
    sudo sed -i "/^GRUB_TIMEOUT_STYLE=/a GRUB_TIMEOUT=5" /etc/default/grub
fi

echo ""
echo "=== Step 6: Update GRUB configuration ==="
echo "Running sudo update-grub..."
sudo update-grub

echo ""
echo "=== Complete ==="
echo ""
echo "Successfully configured GRUB:"
echo "  - GRUB_DEFAULT: $GRUB_DEFAULT_VALUE"
echo "  - GRUB_CMDLINE_LINUX_DEFAULT: Added cma=128M@0-4G log_buf_len=16M"
echo "  - GRUB_TIMEOUT_STYLE: show"
echo "  - GRUB_TIMEOUT: 5"
echo ""
echo "Please reboot the system, then run the following command to verify:"
echo "  uname -r"
echo ""
echo "If the displayed kernel version contains '6.0' and 'lake', the configuration is successful!"

