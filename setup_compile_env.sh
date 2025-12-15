#!/bin/bash
# 检查并配置 CUDA 环境变量

set -e  # 遇到错误时退出

echo "=== CUDA 环境配置脚本 ==="
echo ""

# 步骤1: 检查 nvcc 是否可用
echo "=== 步骤1: 检查 CUDA 编译器 (nvcc) ==="
if command -v nvcc &> /dev/null; then
    echo "✓ 找到 nvcc，正在检查版本..."
    nvcc --version
    echo ""
    echo "CUDA 已正确配置在 PATH 中！"
    exit 0
else
    echo "✗ 未找到 nvcc 命令"
    echo "需要将 CUDA 路径添加到环境变量中"
fi

echo ""
echo "=== 步骤2: 查找 CUDA 安装路径 ==="

# 常见的 CUDA 安装路径
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
        echo "✓ 找到 CUDA 安装: $CUDA_PATH"
        break
    fi
done

# 如果没找到，尝试查找所有可能的 cuda 目录
if [ -z "$CUDA_PATH" ]; then
    echo "在常见路径中未找到 CUDA，正在搜索 /usr/local/cuda* ..."
    FOUND_PATHS=$(ls -d /usr/local/cuda* 2>/dev/null || true)
    if [ -n "$FOUND_PATHS" ]; then
        for path in $FOUND_PATHS; do
            if [ -f "$path/bin/nvcc" ]; then
                CUDA_PATH="$path"
                echo "✓ 找到 CUDA 安装: $CUDA_PATH"
                break
            fi
        done
    fi
fi

if [ -z "$CUDA_PATH" ]; then
    echo "错误: 无法找到 CUDA 安装路径"
    echo "请手动指定 CUDA 路径，或确保 CUDA 已正确安装"
    echo ""
    echo "如果您的 CUDA 安装在非标准位置，请编辑此脚本并设置 CUDA_PATH 变量"
    exit 1
fi

echo ""
echo "=== 步骤3: 检查 ~/.bashrc 配置 ==="

BASHRC_FILE="$HOME/.bashrc"
BACKUP_FILE="${BASHRC_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# 备份 .bashrc
if [ -f "$BASHRC_FILE" ]; then
    cp "$BASHRC_FILE" "$BACKUP_FILE"
    echo "✓ 已备份 ~/.bashrc 到 $BACKUP_FILE"
fi

# 检查是否已存在 CUDA PATH 配置
PATH_EXPORT="export PATH=${CUDA_PATH}/bin:\$PATH"
LD_LIBRARY_PATH_EXPORT="export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH"

PATH_EXISTS=false
LD_LIBRARY_PATH_EXISTS=false

if grep -q "export PATH=.*cuda.*bin" "$BASHRC_FILE" 2>/dev/null; then
    PATH_EXISTS=true
    echo "✓ PATH 配置已存在"
else
    echo "✗ PATH 配置不存在"
fi

if grep -q "export LD_LIBRARY_PATH=.*cuda.*lib64" "$BASHRC_FILE" 2>/dev/null; then
    LD_LIBRARY_PATH_EXISTS=true
    echo "✓ LD_LIBRARY_PATH 配置已存在"
else
    echo "✗ LD_LIBRARY_PATH 配置不存在"
fi

echo ""
echo "=== 步骤4: 更新 ~/.bashrc ==="

# 添加配置（如果不存在）
if [ "$PATH_EXISTS" = false ]; then
    echo "" >> "$BASHRC_FILE"
    echo "# CUDA 环境变量配置" >> "$BASHRC_FILE"
    echo "$PATH_EXPORT" >> "$BASHRC_FILE"
    echo "✓ 已添加 PATH 配置"
fi

if [ "$LD_LIBRARY_PATH_EXISTS" = false ]; then
    if [ "$PATH_EXISTS" = true ]; then
        # 如果 PATH 已存在但 LD_LIBRARY_PATH 不存在，在 PATH 行后添加
        sed -i "/export PATH=.*cuda.*bin/a $LD_LIBRARY_PATH_EXPORT" "$BASHRC_FILE"
    else
        # 如果 PATH 也不存在，直接追加
        echo "$LD_LIBRARY_PATH_EXPORT" >> "$BASHRC_FILE"
    fi
    echo "✓ 已添加 LD_LIBRARY_PATH 配置"
fi

if [ "$PATH_EXISTS" = true ] && [ "$LD_LIBRARY_PATH_EXISTS" = true ]; then
    echo "✓ 所有配置已存在，无需更新"
fi

echo ""
echo "=== 步骤5: 验证配置 ==="
echo "已添加到 ~/.bashrc 的配置："
echo "  $PATH_EXPORT"
echo "  $LD_LIBRARY_PATH_EXPORT"
echo ""

echo "=== 完成 ==="
echo ""
echo "配置已成功添加到 ~/.bashrc"
echo ""
echo "请执行以下命令之一来使配置生效："
echo "  1. 运行: source ~/.bashrc"
echo "  2. 或者重新打开终端"
echo ""
echo "然后运行以下命令验证 CUDA 是否可用："
echo "  nvcc --version"
echo ""

