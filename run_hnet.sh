#!/bin/bash

echo "==================================="
echo "H-Net NanoGPT Speedrun Script"
echo "==================================="

# Check if byte-level data exists
if [ ! -d "data/fineweb10B_bytes" ]; then
    echo "Byte-level data not found. Converting BPE data to bytes..."
    
    # Check if original data exists
    if [ ! -d "data/fineweb10B" ]; then
        echo "Error: Original FineWeb data not found at data/fineweb10B"
        echo "Please download the original data first."
        exit 1
    fi
    
    # Convert data to bytes
    python data/convert_data_to_bytes.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Data conversion failed"
        exit 1
    fi
    
    echo "Data conversion completed!"
else
    echo "Byte-level data found at data/fineweb10B_bytes"
fi

# Check if H-Net config exists
if [ ! -f "hnet/configs/hnet_2stage_L.json" ]; then
    echo "Error: H-Net configuration not found at hnet/configs/hnet_2stage_L.json"
    echo "Please ensure the hnet directory is properly set up."
    exit 1
fi

echo "Starting H-Net training..."
echo "Expected improvements:"
echo "- Byte-level tokenization (vocab_size=256)"
echo "- Hierarchical processing with dynamic chunking"
echo "- Potentially better bits per byte due to learned compression"

# Run the training
torchrun --standalone --nproc_per_node=4 train_hnet.py

echo "H-Net training completed!" 