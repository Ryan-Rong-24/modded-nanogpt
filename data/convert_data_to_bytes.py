#!/usr/bin/env python3
"""
Convert FineWeb data from BPE tokens to byte-level tokens for H-Net training.
This script reads the existing .bin files and converts them to byte-level tokenization.
"""

import os
import glob
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tiktoken

def load_gpt2_tokenizer():
    """Load the GPT-2 tokenizer used in the original data"""
    return tiktoken.get_encoding("gpt2")

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to byte-level tokens"""
        text_bytes = text.encode("utf-8")
        if add_bos:
            text_bytes = bytes([self.bos_idx]) + text_bytes
        if add_eos:
            text_bytes = text_bytes + bytes([self.eos_idx])
        return np.array(bytearray(text_bytes), dtype=self.dtype)

def convert_bin_file(input_path: Path, output_path: Path, gpt2_enc, byte_enc):
    """Convert a single .bin file from BPE to byte-level tokens"""
    
    # Read original file
    header = torch.from_file(str(input_path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    
    with input_path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    
    print(f"Converting {input_path.name}: {num_tokens} BPE tokens")
    
    # Convert to byte-level tokens in chunks to manage memory
    chunk_size = 100000  # Process 100k tokens at a time
    all_byte_tokens = []
    
    for i in tqdm(range(0, len(tokens), chunk_size), desc="Converting chunks"):
        chunk = tokens[i:i + chunk_size].tolist()
        
        # Decode BPE tokens to text
        try:
            text = gpt2_enc.decode(chunk)
        except Exception as e:
            print(f"Warning: Failed to decode chunk starting at {i}: {e}")
            # Skip this chunk or handle error appropriately
            continue
        
        # Encode text to byte-level tokens
        byte_tokens = byte_enc.encode(text)
        all_byte_tokens.extend(byte_tokens)
    
    # Convert to numpy array
    byte_tokens_array = np.array(all_byte_tokens, dtype=np.uint8)
    print(f"Converted to {len(byte_tokens_array)} byte-level tokens")
    
    # Write new file with same header format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create new header
    new_header = header.clone()
    new_header[2] = len(byte_tokens_array)  # Update token count
    
    with output_path.open("wb") as f:
        # Write header
        header_bytes = new_header.numpy().tobytes()
        f.write(header_bytes)
        
        # Write byte tokens (convert to uint16 for compatibility with original format)
        byte_tokens_uint16 = byte_tokens_array.astype(np.uint16)
        f.write(byte_tokens_uint16.tobytes())
    
    print(f"Saved byte-level data to {output_path}")
    return len(byte_tokens_array)

def main():
    """Convert all FineWeb data files to byte-level tokenization"""
    
    # Initialize tokenizers
    gpt2_enc = load_gpt2_tokenizer()
    byte_enc = ByteTokenizer()
    
    # Input and output directories
    input_dir = "data/fineweb10B"
    output_dir = "data/fineweb10B_bytes"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found!")
        print("Please ensure you have downloaded the FineWeb data first.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .bin files
    train_files = glob.glob(f"{input_dir}/fineweb_train_*.bin")
    val_files = glob.glob(f"{input_dir}/fineweb_val_*.bin")
    
    if not train_files and not val_files:
        print(f"Error: No .bin files found in {input_dir}")
        return
    
    print(f"Found {len(train_files)} training files and {len(val_files)} validation files")
    
    # Convert training files
    total_train_tokens = 0
    for file_path in tqdm(train_files, desc="Converting training files"):
        input_path = Path(file_path)
        output_path = Path(output_dir) / input_path.name
        
        try:
            num_tokens = convert_bin_file(input_path, output_path, gpt2_enc, byte_enc)
            total_train_tokens += num_tokens
        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            continue
    
    # Convert validation files
    total_val_tokens = 0
    for file_path in tqdm(val_files, desc="Converting validation files"):
        input_path = Path(file_path)
        output_path = Path(output_dir) / input_path.name
        
        try:
            num_tokens = convert_bin_file(input_path, output_path, gpt2_enc, byte_enc)
            total_val_tokens += num_tokens
        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            continue
    
    print(f"\nConversion complete!")
    print(f"Total training tokens: {total_train_tokens:,}")
    print(f"Total validation tokens: {total_val_tokens:,}")
    print(f"Byte-level data saved to: {output_dir}")
    
    # Update the training script to use the new data path
    print(f"\nTo use the converted data, update train_hnet.py:")
    print(f'train_files = "{output_dir}/fineweb_train_*.bin"')
    print(f'val_files = "{output_dir}/fineweb_val_*.bin"')

if __name__ == "__main__":
    main() 