#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 切换到项目目录
cd "$SCRIPT_DIR"

# 打开终端窗口并执行gensyn.sh
osascript -e 'tell application "Terminal" to do script "cd '"$SCRIPT_DIR"' && ./gensyn.sh"'

# 保持脚本运行一小段时间，确保Terminal启动
sleep 1

echo "项目启动脚本已执行，请查看Terminal窗口"