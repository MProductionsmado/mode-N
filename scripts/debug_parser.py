"""
Debug: Check what blocks are ACTUALLY in the schematics
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.schematic_parser import SchematicParser
import yaml

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Get reverse mapping
block_names = {v: k for k, v in config['blocks'].items()}

# Load parser
parser = SchematicParser(config['blocks'])

# Test files that SHOULD have logs
test_files = [
    'big_birch_wood_birch_leaves_fall_08_01--3de2ea13-a85d-41e0-8d4f-bb23ebf8ec40.schem',
    'big_dead-tree_17.schem',
    'big_fall_01_25--24aa94c1-a09c-4d29-aa0f-e3fd39b92c1f.schem'
]

print("=== Checking specific files with 'wood' or 'tree' in name ===\n")

out_dir = Path('out')
for filename in test_files:
    file_path = out_dir / filename
    if not file_path.exists():
        # Try to find it
        matches = list(out_dir.glob(f"*{filename.split('--')[0]}*.schem"))
        if matches:
            file_path = matches[0]
        else:
            print(f"⚠️  File not found: {filename}")
            continue
    
    print(f"\n📄 {file_path.name}")
    try:
        voxels, meta = parser.parse_schematic(file_path)
        
        print(f"   Dimensions: {voxels.shape}")
        unique, counts = np.unique(voxels, return_counts=True)
        
        print(f"   Found {len(unique)} unique block types:")
        for block_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
            block_name = block_names.get(int(block_id), f"UNKNOWN_{int(block_id)}")
            percentage = (count / voxels.size) * 100
            if block_id != 0 or count < voxels.size * 0.5:  # Show non-air or if air < 50%
                print(f"      ID {int(block_id):3d} ({block_name:20s}): {count:6d} blocks ({percentage:5.2f}%)")
        
        # Check for any wood-like blocks
        wood_like = []
        for block_id in unique:
            name = block_names.get(int(block_id), f"unknown_{int(block_id)}")
            if 'log' in name or 'wood' in name or 'stem' in name or 'planks' in name:
                wood_like.append((int(block_id), name, int(np.sum(voxels == block_id))))
        
        if wood_like:
            print(f"   ✓ Wood-like blocks found:")
            for bid, name, count in wood_like:
                print(f"      ID {bid}: {name} ({count} blocks)")
        else:
            print(f"   ❌ NO wood-like blocks found!")
            
    except Exception as e:
        print(f"   ERROR: {e}")

# Now check RAW palette from NBT
print("\n\n=== Checking RAW NBT Palette ===\n")

from nbtlib import load

for filename in test_files:
    file_path = out_dir / filename
    if not file_path.exists():
        matches = list(out_dir.glob(f"*{filename.split('--')[0]}*.schem"))
        if matches:
            file_path = matches[0]
        else:
            continue
    
    print(f"\n📄 {file_path.name}")
    try:
        nbt_file = load(file_path)
        nbt = nbt_file.root
        
        if 'Palette' in nbt:
            palette = nbt['Palette']
            print(f"   Palette has {len(palette)} entries:")
            for block_name, palette_id in sorted(palette.items(), key=lambda x: x[1]):
                print(f"      [{int(palette_id):3d}] {block_name}")
                
                # Check if it's a log
                if 'log' in str(block_name).lower() or 'wood' in str(block_name).lower():
                    print(f"         ^^ THIS IS A LOG! ^^")
        else:
            print(f"   ⚠️  No Palette found in NBT!")
            
    except Exception as e:
        print(f"   ERROR: {e}")
