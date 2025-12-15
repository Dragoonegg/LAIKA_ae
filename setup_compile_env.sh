#!/bin/bash
# Check and configure CUDA environment variables

set -e  # Exit on error

echo "=== CUDA Environment Configuration Script ==="
echo ""

# Step 1: Check if nvcc is available
echo "=== Step 1: Check CUDA Compiler (nvcc) ==="
if command -v nvcc &> /dev/null; then
    echo "✓ Found nvcc, checking version..."
    nvcc --version
    echo ""
    echo "CUDA is correctly configured in PATH!"
    exit 0
else
    echo "✗ nvcc command not found"
    echo "Need to add CUDA path to environment variables"
fi

echo ""
echo "=== Step 2: Find CUDA Installation Path ==="

# Common CUDA installation paths
POSSIBLE_CUDA_PATHS=(
    "/usr/local/cuda-12.8"
    "/usr/local/cuda-12.7"
    "/usr/local/cuda-12.6"
    "/usr/local/cuda-12.5"
    "/usr/local/cuda-12.4"
    "/usr/local/cuda-12.3"
    "/usr/local/cuda-12.2"
    "/usr/local/cuda-12.1"
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-11.8"
    "/usr/local/cuda-11.7"
    "/usr/local/cuda"
)

CUDA_PATH=""
for path in "${POSSIBLE_CUDA_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
        CUDA_PATH="$path"
        echo "✓ Found CUDA installation: $CUDA_PATH"
        break
    fi
done

# If not found, try to find all possible cuda directories
if [ -z "$CUDA_PATH" ]; then
    echo "CUDA not found in common paths, searching /usr/local/cuda* ..."
    FOUND_PATHS=$(ls -d /usr/local/cuda* 2>/dev/null || true)
    if [ -n "$FOUND_PATHS" ]; then
        for path in $FOUND_PATHS; do
            if [ -f "$path/bin/nvcc" ]; then
                CUDA_PATH="$path"
                echo "✓ Found CUDA installation: $CUDA_PATH"
                break
            fi
        done
    fi
fi

if [ -z "$CUDA_PATH" ]; then
    echo "Error: Unable to find CUDA installation path"
    echo "Please manually specify CUDA path, or ensure CUDA is properly installed"
    echo ""
    echo "If your CUDA is installed in a non-standard location, please edit this script and set the CUDA_PATH variable"
    exit 1
fi

echo ""
echo "=== Step 3: Check ~/.bashrc Configuration ==="

BASHRC_FILE="$HOME/.bashrc"
BACKUP_FILE="${BASHRC_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# Backup .bashrc
if [ -f "$BASHRC_FILE" ]; then
    cp "$BASHRC_FILE" "$BACKUP_FILE"
    echo "✓ Backed up ~/.bashrc to $BACKUP_FILE"
fi

# Check if CUDA PATH configuration already exists
PATH_EXPORT="export PATH=${CUDA_PATH}/bin:\$PATH"
LD_LIBRARY_PATH_EXPORT="export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH"

PATH_EXISTS=false
LD_LIBRARY_PATH_EXISTS=false

if grep -q "export PATH=.*cuda.*bin" "$BASHRC_FILE" 2>/dev/null; then
    PATH_EXISTS=true
    echo "✓ PATH configuration already exists"
else
    echo "✗ PATH configuration does not exist"
fi

if grep -q "export LD_LIBRARY_PATH=.*cuda.*lib64" "$BASHRC_FILE" 2>/dev/null; then
    LD_LIBRARY_PATH_EXISTS=true
    echo "✓ LD_LIBRARY_PATH configuration already exists"
else
    echo "✗ LD_LIBRARY_PATH configuration does not exist"
fi

echo ""
echo "=== Step 4: Update ~/.bashrc ==="

# Add configuration (if it doesn't exist)
if [ "$PATH_EXISTS" = false ]; then
    echo "" >> "$BASHRC_FILE"
    echo "# CUDA environment variables configuration" >> "$BASHRC_FILE"
    echo "$PATH_EXPORT" >> "$BASHRC_FILE"
    echo "✓ Added PATH configuration"
fi

if [ "$LD_LIBRARY_PATH_EXISTS" = false ]; then
    if [ "$PATH_EXISTS" = true ]; then
        # If PATH exists but LD_LIBRARY_PATH doesn't, add it after PATH line
        sed -i "/export PATH=.*cuda.*bin/a $LD_LIBRARY_PATH_EXPORT" "$BASHRC_FILE"
    else
        # If PATH also doesn't exist, append directly
        echo "$LD_LIBRARY_PATH_EXPORT" >> "$BASHRC_FILE"
    fi
    echo "✓ Added LD_LIBRARY_PATH configuration"
fi

if [ "$PATH_EXISTS" = true ] && [ "$LD_LIBRARY_PATH_EXISTS" = true ]; then
    echo "✓ All configurations already exist, no update needed"
fi

echo ""
echo "=== Step 5: Verify Configuration ==="
echo "Configuration added to ~/.bashrc:"
echo "  $PATH_EXPORT"
echo "  $LD_LIBRARY_PATH_EXPORT"
echo ""

echo "=== Complete ==="
echo ""
echo "Configuration has been successfully added to ~/.bashrc"
echo ""
echo "Please execute one of the following commands to apply the configuration:"
echo "  1. Run: source ~/.bashrc"
echo "  2. Or reopen the terminal"
echo ""
echo "Then run the following command to verify CUDA is available:"
echo "  nvcc --version"
echo ""

