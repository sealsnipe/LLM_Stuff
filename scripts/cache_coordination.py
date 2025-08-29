#!/usr/bin/env python3
"""
🤝 CACHE COORDINATION UTILITIES
Helps coordinate between normal and reverse cache creation scripts.
"""

import os
import json
import glob
import time
from typing import Dict, List, Optional

class CacheCoordinator:
    """
    Coordinates between multiple cache creation processes.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.coordination_file = os.path.join(output_dir, "cache_coordination.json")
        
    def analyze_input_files(self, input_dir: str) -> Dict:
        """
        Analyze all parquet files in input directory.
        
        Returns:
            Comprehensive analysis of input files
        """
        parquet_pattern = os.path.join(input_dir, "*.parquet")
        parquet_files = sorted(glob.glob(parquet_pattern))
        
        if not parquet_files:
            return {
                "total_files": 0,
                "files": [],
                "file_analysis": {},
                "chunk_strategy": {}
            }
        
        # Estimate chunks per file (conservative)
        estimated_chunks_per_file = 50
        estimated_total_chunks = len(parquet_files) * estimated_chunks_per_file
        
        analysis = {
            "total_files": len(parquet_files),
            "files": [os.path.basename(f) for f in parquet_files],
            "first_file": os.path.basename(parquet_files[0]),
            "last_file": os.path.basename(parquet_files[-1]),
            "file_analysis": {
                "total_files": len(parquet_files),
                "estimated_chunks_per_file": estimated_chunks_per_file,
                "estimated_total_chunks": estimated_total_chunks,
                "analysis_time": time.time()
            },
            "chunk_strategy": {
                "forward_range": [0, estimated_total_chunks - 1],
                "reverse_range": [estimated_total_chunks * 2, estimated_total_chunks * 3],
                "safe_separation": estimated_total_chunks
            }
        }
        
        return analysis
    
    def register_process(self, process_type: str, process_info: Dict) -> Dict:
        """
        Register a cache creation process.
        
        Args:
            process_type: "forward" or "reverse"
            process_info: Information about the process
            
        Returns:
            Coordination information for this process
        """
        coordination_data = self._load_coordination_data()
        
        process_id = f"{process_type}_{int(time.time())}"
        
        coordination_data["processes"][process_id] = {
            "type": process_type,
            "info": process_info,
            "start_time": time.time(),
            "status": "starting"
        }
        
        # Assign chunk ranges
        if process_type == "forward":
            chunk_range = coordination_data["chunk_strategy"]["forward_range"]
        else:  # reverse
            chunk_range = coordination_data["chunk_strategy"]["reverse_range"]
        
        coordination_data["processes"][process_id]["chunk_range"] = chunk_range
        
        self._save_coordination_data(coordination_data)
        
        return {
            "process_id": process_id,
            "chunk_range": chunk_range,
            "coordination_data": coordination_data
        }
    
    def update_process_status(self, process_id: str, status: str, progress_info: Dict = None):
        """Update process status."""
        coordination_data = self._load_coordination_data()
        
        if process_id in coordination_data["processes"]:
            coordination_data["processes"][process_id]["status"] = status
            coordination_data["processes"][process_id]["last_update"] = time.time()
            
            if progress_info:
                coordination_data["processes"][process_id]["progress"] = progress_info
        
        self._save_coordination_data(coordination_data)
    
    def get_safe_chunk_start(self, process_type: str) -> int:
        """
        Get safe starting chunk number for a process type.
        
        Args:
            process_type: "forward" or "reverse"
            
        Returns:
            Safe starting chunk number
        """
        coordination_data = self._load_coordination_data()
        
        if process_type == "forward":
            return coordination_data["chunk_strategy"]["forward_range"][0]
        else:  # reverse
            return coordination_data["chunk_strategy"]["reverse_range"][0]
    
    def _load_coordination_data(self) -> Dict:
        """Load coordination data from file."""
        if os.path.exists(self.coordination_file):
            try:
                with open(self.coordination_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default coordination data
        return {
            "created": time.time(),
            "processes": {},
            "chunk_strategy": {
                "forward_range": [0, 9999],
                "reverse_range": [10000, 19999],
                "safe_separation": 10000
            },
            "file_analysis": {}
        }
    
    def _save_coordination_data(self, data: Dict):
        """Save coordination data to file."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(self.coordination_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_coordination_status(self):
        """Print current coordination status."""
        coordination_data = self._load_coordination_data()
        
        print(f"\n🤝 CACHE COORDINATION STATUS:")
        print(f"   Coordination file: {self.coordination_file}")
        
        if coordination_data["processes"]:
            print(f"   Active processes: {len(coordination_data['processes'])}")
            for proc_id, proc_info in coordination_data["processes"].items():
                status = proc_info.get("status", "unknown")
                proc_type = proc_info.get("type", "unknown")
                chunk_range = proc_info.get("chunk_range", [0, 0])
                print(f"      {proc_id}: {proc_type} ({status}) chunks {chunk_range[0]}-{chunk_range[1]}")
        else:
            print(f"   No active processes")
        
        chunk_strategy = coordination_data["chunk_strategy"]
        print(f"   Chunk strategy:")
        print(f"      Forward: {chunk_strategy['forward_range'][0]}-{chunk_strategy['forward_range'][1]}")
        print(f"      Reverse: {chunk_strategy['reverse_range'][0]}-{chunk_strategy['reverse_range'][1]}")


def create_coordination_info(input_dir: str, output_dir: str) -> Dict:
    """
    Create coordination information for cache creation.
    
    Args:
        input_dir: Directory with parquet files
        output_dir: Output directory for cache
        
    Returns:
        Coordination information
    """
    coordinator = CacheCoordinator(output_dir)
    
    # Analyze input files
    analysis = coordinator.analyze_input_files(input_dir)
    
    # Update coordination data with analysis
    coordination_data = coordinator._load_coordination_data()
    coordination_data["file_analysis"] = analysis["file_analysis"]
    coordination_data["chunk_strategy"] = analysis["chunk_strategy"]
    coordinator._save_coordination_data(coordination_data)
    
    # Print analysis
    print(f"\n🤝 CACHE COORDINATION SETUP:")
    print(f"   Input files: {analysis['total_files']:,}")
    print(f"   Estimated chunks: ~{analysis['file_analysis']['estimated_total_chunks']:,}")
    print(f"   Forward range: {analysis['chunk_strategy']['forward_range']}")
    print(f"   Reverse range: {analysis['chunk_strategy']['reverse_range']}")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache Coordination Utilities")
    parser.add_argument('--input_dir', required=True, help='Input directory with parquet files')
    parser.add_argument('--output_dir', required=True, help='Output directory for cache')
    parser.add_argument('--action', choices=['analyze', 'status'], default='analyze', help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        create_coordination_info(args.input_dir, args.output_dir)
    elif args.action == 'status':
        coordinator = CacheCoordinator(args.output_dir)
        coordinator.print_coordination_status()
