#!/bin/bash

# Byte-Level NanoGPT Speedrun Script
# Based on the original modded-nanogpt speedrun but using byte-level tokenization (256 vocab)

set -e

echo "Byte-Level NanoGPT Speedrun Starting..."

# Check if converted data exists, if not run conversion
if [ ! -d "data/fineweb10B_bytes" ]; then
    echo "Byte-level data not found. Converting original data..."
    if [ ! -d "data/fineweb10B" ]; then
        echo "Original FineWeb data not found. Please run:"
        echo "python data/cached_fineweb10B.py 8"
        exit 1
    fi
    python convert_data_to_bytes.py
fi

# Set environment for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Record start time
START_TIME=$(date +%s.%3N)

# Run distributed training with H-Net
torchrun --standalone --nproc_per_node=4 train_byte_level.py

# Record end time
END_TIME=$(date +%s.%3N)

# Calculate total time
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
TOTAL_TIME_MIN=$(echo "scale=3; $TOTAL_TIME / 60" | bc)

echo "Byte-Level Training Complete!"
echo "Total time: ${TOTAL_TIME_MIN} minutes"
echo "Expected improvement: Faster training due to smaller vocabulary (256 vs ~50k) and smaller embedding/lm_head layers" 