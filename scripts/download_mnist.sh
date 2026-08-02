#!/bin/bash

# Thư mục đích
TARGET_DIR="data/mnist"

# Tạo thư mục nếu chưa tồn tại
mkdir -p "$TARGET_DIR"

# URLs
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo "Downloading MNIST dataset into $TARGET_DIR..."

for FILE in "${FILES[@]}"; do
    if [ ! -f "$TARGET_DIR/${FILE%.gz}" ]; then
        echo "Downloading $FILE..."
        wget "$BASE_URL/$FILE" -O "$TARGET_DIR/$FILE"
        echo "Extracting $FILE..."
        gzip -d -f "$TARGET_DIR/$FILE"
    else
        echo "File ${FILE%.gz} already exists. Skipping."
    fi
done

echo "MNIST dataset downloaded and extracted successfully."
