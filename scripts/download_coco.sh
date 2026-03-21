#!/usr/bin/env bash
# Download COCO val2017 dataset for the pipeline.
# Run from project root: bash scripts/download_coco.sh

set -euo pipefail

DATA_DIR="./data/coco"
mkdir -p "$DATA_DIR"

echo "=== Downloading COCO val2017 images ==="
if [ ! -d "$DATA_DIR/val2017" ]; then
    wget -q --show-progress -O "$DATA_DIR/val2017.zip" \
        http://images.cocodataset.org/zips/val2017.zip
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/val2017.zip"
    echo "Images extracted to $DATA_DIR/val2017/"
else
    echo "val2017/ already exists, skipping."
fi

echo "=== Downloading COCO annotations ==="
if [ ! -d "$DATA_DIR/annotations" ]; then
    wget -q --show-progress -O "$DATA_DIR/annotations.zip" \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q "$DATA_DIR/annotations.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/annotations.zip"
    echo "Annotations extracted to $DATA_DIR/annotations/"
else
    echo "annotations/ already exists, skipping."
fi

echo "=== Done ==="
echo "Images:      $DATA_DIR/val2017/ ($(ls $DATA_DIR/val2017/ | wc -l) files)"
echo "Annotations: $DATA_DIR/annotations/"
