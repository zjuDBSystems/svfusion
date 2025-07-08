#!/bin/bash

# 指定项目路径
directory=$(pwd)

# 查找指定扩展名文件并统计行数，排除 build/ 目录
total_lines=$(find "$directory" -type d -name "build" -prune -o -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.c" -o -name "*.hpp" -o -name "*.cuh" \) -exec cat {} + | sed '/^\s*$/d; /\/\//d' | wc -l)

echo "Total lines of code: $total_lines"
