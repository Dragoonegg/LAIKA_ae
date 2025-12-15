#!/bin/bash
# 设置GRUB默认启动项为6.0-lake内核并添加内核参数

set -e  # 遇到错误时退出

echo "=== 步骤1: 提取高级菜单ID和6.0-lake内核ID ==="

# 提取高级菜单ID（从submenu行中提取最后一个单引号中的内容，通常是ID）
SUBMENU_LINE=$(cat /boot/grub/grub.cfg | grep submenu | head -1)
if [ -z "$SUBMENU_LINE" ]; then
    echo "错误: 无法找到submenu行"
    exit 1
fi
echo "找到submenu行: $SUBMENU_LINE"

# 提取ID（通常是最后一个单引号中的内容，格式如 gnulinux-advanced-...）
ADVANCED_MENU_ID=$(echo "$SUBMENU_LINE" | grep -o "'[^']*'" | tail -1 | tr -d "'")
if [ -z "$ADVANCED_MENU_ID" ] || [[ ! "$ADVANCED_MENU_ID" =~ ^gnulinux- ]]; then
    # 如果最后一个不是ID，尝试查找包含gnulinux-advanced的
    ADVANCED_MENU_ID=$(echo "$SUBMENU_LINE" | grep -o "gnulinux-advanced-[^' ]*" | head -1)
fi
if [ -z "$ADVANCED_MENU_ID" ]; then
    echo "错误: 无法从submenu行中提取高级菜单ID"
    exit 1
fi
echo "高级菜单ID: $ADVANCED_MENU_ID"

# 提取6.0-lake内核ID（按照用户说明：grep option | grep 6.0.0-lake）
OPTION_LAKE_LINE=$(cat /boot/grub/grub.cfg | grep option | grep "6.0.*lake" | head -1)
if [ -z "$OPTION_LAKE_LINE" ]; then
    echo "错误: 无法找到包含'option'和'6.0.*lake'的行"
    echo "尝试查找包含'6.0'和'lake'的menuentry行..."
    OPTION_LAKE_LINE=$(cat /boot/grub/grub.cfg | grep -i "6.0.*lake" | head -1)
fi
if [ -z "$OPTION_LAKE_LINE" ]; then
    echo "错误: 无法找到6.0-lake内核相关行"
    exit 1
fi
echo "找到6.0-lake行: $OPTION_LAKE_LINE"

# 提取ID（查找包含gnulinux和lake的ID）
LAKE_KERNEL_ID=$(echo "$OPTION_LAKE_LINE" | grep -o "gnulinux-[^' ]*lake[^' ]*" | head -1)
if [ -z "$LAKE_KERNEL_ID" ]; then
    # 尝试从最后一个单引号中提取
    LAKE_KERNEL_ID=$(echo "$OPTION_LAKE_LINE" | grep -o "'[^']*'" | tail -1 | tr -d "'")
fi
if [ -z "$LAKE_KERNEL_ID" ]; then
    echo "错误: 无法提取6.0-lake内核ID"
    exit 1
fi
echo "6.0-lake内核ID: $LAKE_KERNEL_ID"

echo ""
echo "=== 步骤2: 组合菜单ID ==="
GRUB_DEFAULT_VALUE="${ADVANCED_MENU_ID}>${LAKE_KERNEL_ID}"
echo "组合后的GRUB_DEFAULT值: $GRUB_DEFAULT_VALUE"

echo ""
echo "=== 步骤3: 更新 /etc/default/grub - 添加GRUB_DEFAULT ==="

# 备份原始文件
sudo cp /etc/default/grub /etc/default/grub.backup.$(date +%Y%m%d_%H%M%S)
echo "已备份 /etc/default/grub"

# 检查是否已存在GRUB_DEFAULT
if grep -q "^GRUB_DEFAULT=" /etc/default/grub; then
    # 如果存在，更新它
    echo "更新现有的GRUB_DEFAULT..."
    sudo sed -i "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"${GRUB_DEFAULT_VALUE}\"|" /etc/default/grub
else
    # 如果不存在，在文件顶部添加
    echo "在文件顶部添加GRUB_DEFAULT..."
    sudo sed -i "1i GRUB_DEFAULT=\"${GRUB_DEFAULT_VALUE}\"" /etc/default/grub
fi

echo ""
echo "=== 步骤4: 更新GRUB_CMDLINE_LINUX_DEFAULT ==="

KERNEL_PARAMS="cma=128M@0-4G log_buf_len=16M"

# 检查GRUB_CMDLINE_LINUX_DEFAULT是否存在
if grep -q "^GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub; then
    # 如果存在，检查是否已包含这些参数
    CURRENT_CMDLINE=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub | sed 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/\1/')
    
    # 检查是否需要添加参数
    if echo "$CURRENT_CMDLINE" | grep -q "cma=128M@0-4G" && echo "$CURRENT_CMDLINE" | grep -q "log_buf_len=16M"; then
        echo "内核参数已存在，无需更新"
    else
        echo "更新GRUB_CMDLINE_LINUX_DEFAULT，添加新参数..."
        # 移除引号，添加新参数，然后重新添加引号
        NEW_CMDLINE="${CURRENT_CMDLINE} ${KERNEL_PARAMS}"
        sudo sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${NEW_CMDLINE}\"|" /etc/default/grub
    fi
else
    # 如果不存在，创建它
    echo "创建GRUB_CMDLINE_LINUX_DEFAULT..."
    # 在GRUB_DEFAULT之后添加
    sudo sed -i "/^GRUB_DEFAULT=/a GRUB_CMDLINE_LINUX_DEFAULT=\"${KERNEL_PARAMS}\"" /etc/default/grub
fi

echo ""
echo "=== 步骤5: 更新GRUB_TIMEOUT_STYLE和GRUB_TIMEOUT ==="

# 更新GRUB_TIMEOUT_STYLE
if grep -q "^GRUB_TIMEOUT_STYLE=" /etc/default/grub; then
    echo "更新现有的GRUB_TIMEOUT_STYLE..."
    sudo sed -i "s|^GRUB_TIMEOUT_STYLE=.*|GRUB_TIMEOUT_STYLE=show|" /etc/default/grub
else
    echo "创建GRUB_TIMEOUT_STYLE..."
    # 在GRUB_CMDLINE_LINUX_DEFAULT之后添加
    sudo sed -i "/^GRUB_CMDLINE_LINUX_DEFAULT=/a GRUB_TIMEOUT_STYLE=show" /etc/default/grub
fi

# 更新GRUB_TIMEOUT
if grep -q "^GRUB_TIMEOUT=" /etc/default/grub; then
    echo "更新现有的GRUB_TIMEOUT..."
    sudo sed -i "s|^GRUB_TIMEOUT=.*|GRUB_TIMEOUT=5|" /etc/default/grub
else
    echo "创建GRUB_TIMEOUT..."
    # 在GRUB_TIMEOUT_STYLE之后添加
    sudo sed -i "/^GRUB_TIMEOUT_STYLE=/a GRUB_TIMEOUT=5" /etc/default/grub
fi

echo ""
echo "=== 步骤6: 更新GRUB配置 ==="
echo "正在运行 sudo update-grub..."
sudo update-grub

echo ""
echo "=== 完成 ==="
echo ""
echo "已成功配置GRUB："
echo "  - GRUB_DEFAULT: $GRUB_DEFAULT_VALUE"
echo "  - GRUB_CMDLINE_LINUX_DEFAULT: 已添加 cma=128M@0-4G log_buf_len=16M"
echo "  - GRUB_TIMEOUT_STYLE: show"
echo "  - GRUB_TIMEOUT: 5"
echo ""
echo "请重启系统，然后运行以下命令验证："
echo "  uname -r"
echo ""
echo "如果显示的内核版本包含'6.0'和'lake'，则配置成功！"

