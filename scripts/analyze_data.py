"""
Analyze training data to check block distribution
"""

import json
from pathlib import Path
import numpy as np
from collections import Counter
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.schematic_parser import SchematicParser

# Load config
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Load metadata
metadata_path = Path('out/metadata.json')
with open(metadata_path) as f:
    metadata = json.load(f)

print(f"Total schematics: {len(metadata)}")
print(f"Size distribution: {Counter([m['size_name'] for m in metadata])}")

# Sample analysis
parser = SchematicParser(config['blocks'])

total_blocks = Counter()
files_with_logs = 0
files_without_logs = 0

print("\n=== Analyzing first 100 schematics ===")
for i, meta in enumerate(metadata[:100]):
    try:
        file_path = Path('out') / meta['filename']
        voxels, _ = parser.parse_schematic(file_path)
        
        # Count blocks
        unique, counts = np.unique(voxels, return_counts=True)
        for block_id, count in zip(unique, counts):
            total_blocks[int(block_id)] += int(count)
        
        # Check for logs (1, 3, 5, 7, 9)
        log_blocks = [1, 3, 5, 7, 9]
        has_logs = any(block_id in unique for block_id in log_blocks)
        if has_logs:
            files_with_logs += 1
        else:
            files_without_logs += 1
            
    except Exception as e:
        print(f"Error loading {meta['filename']}: {e}")

print(f"\n=== Block Distribution (first 100 files) ===")
block_names = {v: k for k, v in config['blocks'].items()}
total_count = sum(total_blocks.values())

sorted_blocks = sorted(total_blocks.items(), key=lambda x: x[1], reverse=True)
for block_id, count in sorted_blocks[:15]:
    block_name = block_names.get(block_id, f"unknown_{block_id}")
    percentage = (count / total_count) * 100
    print(f"{block_name:20s} (ID {block_id:2d}): {count:8d} blocks ({percentage:5.2f}%)")

print(f"\n=== Log Analysis ===")
print(f"Files WITH logs: {files_with_logs}")
print(f"Files WITHOUT logs: {files_without_logs}")

# Check specific log types
print("\n=== Log Types ===")
for log_id, log_name in [(1, 'oak_log'), (3, 'birch_log'), (5, 'spruce_log'), (7, 'acacia_log'), (9, 'dark_oak_log')]:
    count = total_blocks.get(log_id, 0)
    if count > 0:
        percentage = (count / total_count) * 100
        print(f"{log_name:20s}: {count:6d} ({percentage:5.2f}%)")
    else:
        print(f"{log_name:20s}: 0 (NOT PRESENT!)")

# Show sample with logs
print("\n=== Sample Files with Logs ===")
for meta in metadata[:20]:
    try:
        file_path = Path('out') / meta['filename']
        voxels, _ = parser.parse_schematic(file_path)
        unique = np.unique(voxels)
        
        log_blocks = [1, 3, 5, 7, 9]
        has_logs = any(block_id in unique for block_id in log_blocks)
        if has_logs:
            log_counts = {block_names[b]: int(np.sum(voxels == b)) for b in log_blocks if b in unique}
            print(f"{meta['filename']:40s}: {log_counts}")
    except:
        pass
