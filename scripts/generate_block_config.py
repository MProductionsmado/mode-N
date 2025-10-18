"""
Generate block vocabulary from top 20 most common blocks in dataset
Analyzes .schem files in 'out' directory and creates config block list
Uses nbtlib directly like schematic_parser.py
"""

import os
from pathlib import Path
from collections import Counter
import nbtlib


def decode_varint(data):
    """
    Decode varint-encoded block data
    Based on schematic_parser.py implementation
    """
    blocks = []
    i = 0
    
    while i < len(data):
        value = 0
        shift = 0
        
        while True:
            if i >= len(data):
                break
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            
            if (byte & 0x80) == 0:
                break
        
        blocks.append(value)
    
    return blocks


def get_all_blocks_from_schematics(schematic_dir: str):
    """
    Extract all blocks from all schematics in directory
    
    Args:
        schematic_dir: Path to directory containing .schem files
    
    Returns:
        Counter object with block counts
    """
    schematic_dir = Path(schematic_dir)
    block_counter = Counter()
    
    schem_files = list(schematic_dir.glob("*.schem"))
    print(f"Found {len(schem_files)} schematic files")
    
    for i, schem_file in enumerate(schem_files):
        try:
            # Load NBT file directly (like schematic_parser.py)
            nbt = nbtlib.load(schem_file)
            
            # Extract palette (block_id -> block_name mapping)
            if 'Palette' not in nbt:
                continue
                
            palette = {}
            for block_name, block_id in nbt['Palette'].items():
                # Clean block name: remove "minecraft:" and properties "[...]"
                clean_name = str(block_name).replace('minecraft:', '').split('[')[0]
                palette[int(block_id)] = clean_name
            
            # Extract and decode block data
            block_data = list(nbt['BlockData'])
            blocks = decode_varint(block_data)
            
            # Count all blocks
            for block_idx in blocks:
                if block_idx in palette:
                    block_counter[palette[block_idx]] += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(schem_files)} files...")
                
        except Exception as e:
            print(f"Error loading {schem_file.name}: {e}")
            continue
    
    return block_counter


def generate_block_config(block_counter: Counter, top_n: int = 20):
    """
    Generate YAML config block list from most common blocks
    
    Args:
        block_counter: Counter with block frequencies
        top_n: Number of top blocks to include
    
    Returns:
        String with YAML block config
    """
    # Always include air as block 0
    config_lines = ["blocks:", "  air: 0"]
    
    # Get top N blocks (excluding air)
    most_common = block_counter.most_common(top_n + 5)  # Get extra in case air is in there
    
    # Filter out air and take top N
    top_blocks = [block for block, count in most_common if block != 'air'][:top_n]
    
    # Generate config
    block_id = 1
    for block_name in top_blocks:
        count = block_counter[block_name]
        config_lines.append(f"  {block_name}: {block_id}  # Count: {count:,}")
        block_id += 1
    
    return '\n'.join(config_lines)


def main():
    # Directory containing schematics
    out_dir = "out"
    
    if not os.path.exists(out_dir):
        print(f"Error: Directory '{out_dir}' not found!")
        return
    
    print("="*60)
    print("Analyzing Minecraft Schematics")
    print("="*60)
    print()
    
    # Count all blocks
    print("Counting blocks in all schematics...")
    block_counter = get_all_blocks_from_schematics(out_dir)
    
    print()
    print("="*60)
    print("Top 30 Most Common Blocks:")
    print("="*60)
    for block, count in block_counter.most_common(30):
        print(f"  {block:30s} {count:>8,} blocks")
    
    print()
    print("="*60)
    print("Generated Config (Top 20 + Air):")
    print("="*60)
    print()
    
    # Generate config for top 20
    config = generate_block_config(block_counter, top_n=20)
    print(config)
    
    print()
    print("="*60)
    print("Copy the block config above to your config.yaml file!")
    print("="*60)
    
    # Optionally save to file
    output_file = "block_config_generated.yaml"
    with open(output_file, 'w') as f:
        f.write(config)
    print(f"\nAlso saved to: {output_file}")


if __name__ == "__main__":
    main()
