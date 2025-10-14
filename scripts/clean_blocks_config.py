"""
Clean and optimize block list for Minecraft 3D diffusion
- Remove utility blocks (barrier, etc.)
- Keep only one variant of similar blocks (oak_log preferred over oak_wood)
- Ensure continuous numbering (0, 1, 2, 3, ...)
"""

# Manual curated list of blocks for NATURE ASSETS
# Prioritized by importance for trees, rocks, bushes

ESSENTIAL_BLOCKS = [
    # Air & Base
    'air',
    
    # Tree Leaves (most important)
    'oak_leaves',
    'spruce_leaves',
    'birch_leaves',
    'jungle_leaves',
    'dark_oak_leaves',
    'acacia_leaves',
    'azalea_leaves',
    'flowering_azalea_leaves',
    'cherry_leaves',
    'mangrove_leaves',
    
    # Tree Logs (NOT wood - logs are natural!)
    'oak_log',
    'spruce_log',
    'birch_log',
    'jungle_log',
    'dark_oak_log',
    'acacia_log',
    
    # Grass & Plants
    'tall_grass',
    'short_grass',
    'fern',
    'grass_block',
    'moss_block',
    'moss_carpet',
    
    # Flowers
    'dandelion',
    'poppy',
    'blue_orchid',
    'allium',
    'azure_bluet',
    'red_tulip',
    'white_tulip',
    'pink_tulip',
    'oxeye_daisy',
    'sunflower',
    'peony',
    'pink_petals',
    
    # Mushrooms
    'brown_mushroom',
    'red_mushroom',
    'brown_mushroom_block',
    
    # Rocks & Stone
    'stone',
    'cobblestone',
    'mossy_cobblestone',
    'andesite',
    'granite',
    'diorite',
    'dirt',
    'coarse_dirt',
    
    # Snow & Ice
    'snow',
    'snow_block',
    'ice',
    'packed_ice',
    
    # Other Natural
    'cactus',
    'vine',
    'lily_pad',
    'dead_bush',
    
    # Saplings (for small trees)
    'oak_sapling',
    'spruce_sapling',
    'birch_sapling',
    'jungle_sapling',
    'dark_oak_sapling',
    'acacia_sapling',
    'azalea',
    'flowering_azalea',
]

# Blocks to REMOVE (utility, decorative, not natural)
REMOVE_BLOCKS = [
    'barrier',  # Utility block
    'oak_wood', # Duplicate of oak_log
    'spruce_wood', # Duplicate of spruce_log
    'birch_wood', # Duplicate of birch_log
    'jungle_wood', # Duplicate of jungle_log
    'dark_oak_wood', # Duplicate of dark_oak_log
    'acacia_wood', # Duplicate of acacia_log
    'oak_planks', # Crafted, not natural
    'spruce_planks', # Crafted
    'oak_fence', # Crafted
    'oak_button', # Crafted
    'lever', # Redstone
    'note_block', # Redstone
    'piston_head', # Redstone
    'daylight_detector_inverted', # Redstone
    'redstone_lamp', # Redstone
    'yellow_stained_glass', # Crafted/Decorative
    'yellow_stained_glass_pane', # Crafted
    'orange_stained_glass', # Crafted
    'pink_stained_glass', # Crafted
    'yellow_wool', # Crafted
    'yellow_concrete', # Crafted
    'yellow_glazed_terracotta', # Crafted
    'terracotta', # Crafted
    'bone_block', # Usually not in nature
    'barrel', # Crafted
    'beehive', # Rare, can remove
    'composter', # Crafted
]

def generate_clean_config():
    """Generate cleaned block config"""
    
    # Use essential blocks as base
    clean_blocks = {}
    
    for idx, block_name in enumerate(ESSENTIAL_BLOCKS):
        clean_blocks[block_name] = idx
    
    print("="*80)
    print(f"CLEANED BLOCK CONFIG - {len(clean_blocks)} BLOCKS")
    print("="*80)
    print("\nblocks:")
    
    for block_name, idx in clean_blocks.items():
        print(f"  {block_name}: {idx}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"✅ Total blocks: {len(clean_blocks)}")
    print(f"✅ Lückenlos nummeriert: 0 bis {len(clean_blocks)-1}")
    print(f"✅ Entfernt: barrier, oak_wood, utility blocks")
    print(f"✅ Behalten: oak_log (natural tree logs)")
    
    print("\n" + "="*80)
    print("BLOCK CATEGORIES:")
    print("="*80)
    
    categories = {
        'Air': ['air'],
        'Leaves': [b for b in clean_blocks if 'leaves' in b],
        'Logs': [b for b in clean_blocks if 'log' in b],
        'Grass/Plants': [b for b in clean_blocks if any(x in b for x in ['grass', 'fern', 'moss'])],
        'Flowers': [b for b in clean_blocks if any(x in b for x in ['flower', 'tulip', 'orchid', 'dandelion', 'poppy', 'allium', 'azure', 'oxeye', 'sunflower', 'peony', 'petal'])],
        'Mushrooms': [b for b in clean_blocks if 'mushroom' in b],
        'Stone/Rock': [b for b in clean_blocks if any(x in b for x in ['stone', 'cobble', 'andesite', 'granite', 'diorite', 'dirt'])],
        'Snow/Ice': [b for b in clean_blocks if any(x in b for x in ['snow', 'ice'])],
    }
    
    # Calculate "Other" category after all others are defined
    all_categorized = sum(categories.values(), [])
    categories['Other'] = [b for b in clean_blocks if b not in all_categorized]
    
    for category, blocks in categories.items():
        if blocks:
            print(f"\n{category}: {len(blocks)} blocks")
            for b in blocks:
                print(f"  - {b}")
    
    # Save to file
    import yaml
    output_file = "blocks_config_cleaned.yaml"
    with open(output_file, 'w') as f:
        yaml.dump({'blocks': clean_blocks}, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"📝 Copy to config/config.yaml")

if __name__ == "__main__":
    generate_clean_config()
