"""
Analyze Packed Cache Chunks

Deep analysis of packed cache chunks to understand the real token distribution
and verify our calculations.
"""

import os
import torch
import lz4.frame
import io
import glob
from typing import Dict, List


def analyze_chunk_file(chunk_file: str) -> Dict:
    """Analyze a single chunk file."""
    
    try:
        # Try loading as compressed first
        try:
            with open(chunk_file, 'rb') as f:
                compressed_data = f.read()
            decompressed_data = lz4.frame.decompress(compressed_data)
            chunk_data = torch.load(io.BytesIO(decompressed_data), map_location='cpu', weights_only=False)
            compression_used = True
        except:
            # Fallback to uncompressed
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            compression_used = False
        
        analysis = {
            'file': os.path.basename(chunk_file),
            'compression_used': compression_used,
            'keys': list(chunk_data.keys()),
            'file_size_mb': os.path.getsize(chunk_file) / (1024*1024)
        }
        
        # Analyze input_ids if present
        if 'input_ids' in chunk_data:
            input_ids = chunk_data['input_ids']
            analysis.update({
                'tensor_shape': input_ids.shape,
                'tensor_dtype': str(input_ids.dtype),
                'num_sequences': input_ids.shape[0],
                'sequence_length': input_ids.shape[1],
                'total_positions': input_ids.numel(),
            })
            
            # Count actual tokens (non-padding)
            # Assume padding token is 0 or a specific value
            padding_candidates = [0, 1, 2, 3]  # Common padding tokens
            
            for pad_token in padding_candidates:
                non_padding = (input_ids != pad_token).sum().item()
                padding_count = (input_ids == pad_token).sum().item()
                utilization = non_padding / input_ids.numel()
                
                analysis[f'pad_token_{pad_token}'] = {
                    'non_padding_tokens': non_padding,
                    'padding_tokens': padding_count,
                    'utilization': utilization
                }
            
            # Sample some sequences to see the data
            analysis['sample_sequences'] = []
            for i in range(min(3, input_ids.shape[0])):
                seq = input_ids[i]
                # Find first few non-zero tokens
                non_zero_indices = (seq != 0).nonzero().flatten()
                if len(non_zero_indices) > 0:
                    first_tokens = seq[non_zero_indices[:10]].tolist()
                    last_tokens = seq[non_zero_indices[-10:]].tolist()
                    analysis['sample_sequences'].append({
                        'sequence_idx': i,
                        'first_10_tokens': first_tokens,
                        'last_10_tokens': last_tokens,
                        'total_non_zero': len(non_zero_indices),
                        'utilization': len(non_zero_indices) / len(seq)
                    })
        
        # Analyze metadata if present
        if 'metadata' in chunk_data:
            metadata = chunk_data['metadata']
            analysis['chunk_metadata'] = metadata
        
        return analysis
        
    except Exception as e:
        return {
            'file': os.path.basename(chunk_file),
            'error': str(e),
            'file_size_mb': os.path.getsize(chunk_file) / (1024*1024)
        }


def analyze_cache_directory(cache_dir: str = "cache/packed_sequences/512/FineWeb"):
    """Analyze the entire cache directory."""
    
    print("🔍 DEEP CACHE ANALYSIS")
    print("=" * 50)
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    # Find chunk files
    chunk_files = sorted(glob.glob(os.path.join(cache_dir, "packed_chunk_*.pt")))
    
    if not chunk_files:
        print(f"❌ No chunk files found in {cache_dir}")
        return
    
    print(f"📦 Found {len(chunk_files)} chunk files")
    print(f"📁 Cache directory: {cache_dir}")
    print()
    
    # Analyze first few chunks in detail
    detailed_analysis = []
    for i, chunk_file in enumerate(chunk_files[:3]):
        print(f"🔍 Analyzing chunk {i+1}/3: {os.path.basename(chunk_file)}")
        analysis = analyze_chunk_file(chunk_file)
        detailed_analysis.append(analysis)
        
        if 'error' in analysis:
            print(f"   ❌ Error: {analysis['error']}")
            continue
        
        print(f"   📊 Shape: {analysis.get('tensor_shape', 'N/A')}")
        print(f"   💾 File size: {analysis['file_size_mb']:.1f} MB")
        print(f"   🗜️  Compression: {analysis['compression_used']}")
        
        # Show utilization for different padding assumptions
        if 'pad_token_0' in analysis:
            util_0 = analysis['pad_token_0']['utilization']
            util_1 = analysis['pad_token_1']['utilization']
            print(f"   📈 Utilization (pad=0): {util_0:.1%}")
            print(f"   📈 Utilization (pad=1): {util_1:.1%}")
        
        # Show sample sequences
        if 'sample_sequences' in analysis:
            for seq_info in analysis['sample_sequences'][:1]:  # Just first sequence
                print(f"   🔤 Sample seq {seq_info['sequence_idx']}: {seq_info['total_non_zero']}/512 tokens ({seq_info['utilization']:.1%})")
                print(f"      First tokens: {seq_info['first_10_tokens']}")
        
        print()
    
    # Quick analysis of all chunks
    print("📊 SUMMARY ANALYSIS")
    print("-" * 30)
    
    total_sequences = 0
    total_file_size = 0
    utilization_samples = []
    
    for i, chunk_file in enumerate(chunk_files):
        if i % 10 == 0:
            print(f"   Processing chunk {i+1}/{len(chunk_files)}...")
        
        try:
            analysis = analyze_chunk_file(chunk_file)
            if 'num_sequences' in analysis:
                total_sequences += analysis['num_sequences']
            
            total_file_size += analysis['file_size_mb']
            
            if 'pad_token_0' in analysis:
                utilization_samples.append(analysis['pad_token_0']['utilization'])
                
        except Exception as e:
            print(f"   ⚠️  Error with {os.path.basename(chunk_file)}: {e}")
    
    print(f"\n✅ FINAL RESULTS:")
    print(f"   Total sequences: {total_sequences:,}")
    print(f"   Total file size: {total_file_size:.1f} MB")
    
    if utilization_samples:
        avg_util = sum(utilization_samples) / len(utilization_samples)
        min_util = min(utilization_samples)
        max_util = max(utilization_samples)
        print(f"   Average utilization: {avg_util:.1%}")
        print(f"   Utilization range: {min_util:.1%} - {max_util:.1%}")
        
        # Calculate real tokens
        real_tokens = int(total_sequences * 512 * avg_util)
        possible_tokens = total_sequences * 512
        print(f"   Real tokens: {real_tokens:,}")
        print(f"   Possible tokens: {possible_tokens:,}")
        print(f"   Token efficiency: {real_tokens/possible_tokens:.1%}")
    
    return detailed_analysis


def main():
    """Main analysis function."""
    
    cache_dir = "cache/packed_sequences/512/FineWeb"
    
    print("🔍 PACKED CACHE DEEP ANALYSIS")
    print("=" * 50)
    print("Analyzing real token distribution in packed sequences...")
    print()
    
    detailed_analysis = analyze_cache_directory(cache_dir)
    
    print("\n🎯 CONCLUSION:")
    print("This analysis shows the real token distribution in your packed cache.")
    print("Compare these numbers with your training calculations!")


if __name__ == "__main__":
    main()
