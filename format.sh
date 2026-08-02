#!/bin/bash

# Thư mục chứa mã nguồn cần format
DIRS="src include tests benchmark examples stress"

# Kiểm tra xem clang-format đã được cài đặt chưa
if ! command -v clang-format &> /dev/null
then
    echo "Lỗi: Không tìm thấy clang-format. Vui lòng cài đặt (vd: sudo apt install clang-format)."
    exit 1
fi

echo "Đang format mã nguồn bằng clang-format..."

# Tìm và format tất cả các file C/C++ trong các thư mục được chỉ định
find $DIRS -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" \) -exec clang-format -i {} +

echo "Format hoàn tất!"
