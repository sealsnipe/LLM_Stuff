#!/usr/bin/env python3
"""
Debug script to analyze the reverse cache chunks and find out what's wrong
"""

import torch
import lz4.frame
import io
import os

def analyze_chunk(chunk_file):
    """Analyze a chunk file and show detailed information"""
    print(f"\n🔍 ANALYZING: {chunk_file}")
    print(f"File size: {os.path.getsize(chunk_file) / (1024*1024):.1f} MB")
    
    try:
        # Try loading as compressed first
        try:
            with open(chunk_file, 'rb') as f:
                compressed_data = f.read()
            print(f"Raw file size: {len(compressed_data)} bytes")
            
            decompressed_data = lz4.frame.decompress(compressed_data)
            print(f"Decompressed size: {len(decompressed_data)} bytes")
            
            chunk_data = torch.load(io.BytesIO(decompressed_data), map_location='cpu', weights_only=False)
            compression_used = True
            print("✅ Successfully loaded as LZ4 compressed")
            
        except Exception as e:
            print(f"❌ LZ4 decompression failed: {e}")
            # Fallback to uncompressed
            try:
                chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
                compression_used = False
                print("✅ Successfully loaded as uncompressed")
            except Exception as e2:
                print(f"❌ Uncompressed loading also failed: {e2}")
                return None
        
        print(f"Compression used: {compression_used}")
        print(f"Keys in chunk: {list(chunk_data.keys())}")
        
        # Analyze input_ids
        if 'input_ids' in chunk_data:
            input_ids = chunk_data['input_ids']
            print(f"\n📊 INPUT_IDS ANALYSIS:")
            print(f"   Shape: {input_ids.shape}")
            print(f"   Dtype: {input_ids.dtype}")
            print(f"   Total elements: {input_ids.numel():,}")
            print(f"   Memory size: {input_ids.numel() * 4 / (1024*1024):.1f} MB (if int32)")
            
            # Check for padding
            if hasattr(input_ids, 'eq'):
                # Assume pad_token_id is 0 or 50256 (common values)
                for pad_id in [0, 50256, 49152]:
                    pad_count = (input_ids == pad_id).sum().item()
                    if pad_count > 0:
                        utilization = (input_ids.numel() - pad_count) / input_ids.numel() * 100
                        print(f"   Padding tokens (id={pad_id}): {pad_count:,} ({100-utilization:.1f}%)")
                        print(f"   Utilization: {utilization:.1f}%")
                        break
            
            # Show some sample values
            print(f"   Sample values: {input_ids.flatten()[:20].tolist()}")
            print(f"   Min value: {input_ids.min().item()}")
            print(f"   Max value: {input_ids.max().item()}")
        
        # Analyze metadata
        if 'metadata' in chunk_data:
            metadata = chunk_data['metadata']
            print(f"\n📋 METADATA:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
        
        return chunk_data
        
    except Exception as e:
        print(f"❌ Failed to analyze chunk: {e}")
        return None

def main():
    print("🔍 REVERSE CHUNK ANALYSIS")
    print("=" * 50)

    # Analyze the NEW reverse chunk (fixed)
    new_reverse_chunk = "../cache/packed_sequences/512/FineWeb/packed_chunk_010000.pt"
    new_data = None
    if os.path.exists(new_reverse_chunk):
        new_data = analyze_chunk(new_reverse_chunk)

    # Analyze a good chunk for comparison
    good_chunk = "../cache/packed_sequences/512/FineWeb/packed_chunk_000000.pt"
    good_data = None
    if os.path.exists(good_chunk):
        good_data = analyze_chunk(good_chunk)
    
    # Compare if both loaded successfully
    if new_data and good_data:
        print(f"\n🔄 DETAILED COMPARISON:")

        if 'input_ids' in new_data and 'input_ids' in good_data:
            new_shape = new_data['input_ids'].shape
            good_shape = good_data['input_ids'].shape

            print(f"   NEW reverse chunk sequences: {new_shape[0]}")
            print(f"   GOOD original chunk sequences: {good_shape[0]}")
            print(f"   NEW chunk seq_length: {new_shape[1] if len(new_shape) > 1 else 'N/A'}")
            print(f"   GOOD chunk seq_length: {good_shape[1] if len(good_shape) > 1 else 'N/A'}")

            # Calculate expected file size
            new_expected_size = new_shape[0] * new_shape[1] * 8 / (1024*1024) if len(new_shape) > 1 else 0  # int64 = 8 bytes
            good_expected_size = good_shape[0] * good_shape[1] * 8 / (1024*1024) if len(good_shape) > 1 else 0

            print(f"   NEW expected size: {new_expected_size:.1f} MB")
            print(f"   GOOD expected size: {good_expected_size:.1f} MB")

            # Compare utilization
            if 'metadata' in new_data and 'metadata' in good_data:
                new_util = new_data['metadata'].get('utilization', 0)
                good_util = good_data['metadata'].get('utilization', 0)
                print(f"   NEW utilization: {new_util:.2f}%")
                print(f"   GOOD utilization: {good_util:.2f}%")

            # Check compression ratio
            new_file_size = os.path.getsize("../cache/packed_sequences/512/FineWeb/packed_chunk_010000.pt") / (1024*1024)
            good_file_size = os.path.getsize("../cache/packed_sequences/512/FineWeb/packed_chunk_000000.pt") / (1024*1024)

            new_compression_ratio = new_file_size / new_expected_size if new_expected_size > 0 else 0
            good_compression_ratio = good_file_size / good_expected_size if good_expected_size > 0 else 0

            print(f"   NEW compression ratio: {new_compression_ratio:.2f}")
            print(f"   GOOD compression ratio: {good_compression_ratio:.2f}")

            # Sequence count difference
            seq_diff = good_shape[0] - new_shape[0]
            print(f"   Sequence difference: {seq_diff} ({seq_diff/good_shape[0]*100:.1f}% fewer in NEW)")

if __name__ == "__main__":
    main()
