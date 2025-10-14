"""
Preprocess schematic dataset
- Parse all .schem files
- Extract metadata and tags
- Convert to voxel arrays
- Create train/val/test splits
"""

import argparse
import yaml
from pathlib import Path
import logging
import sys
import json
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.schematic_parser import SchematicParser
from src.data.preprocessing import (
    FilenameParser, SizeClassifier, DatasetSplitter,
    create_text_prompt, analyze_dataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Minecraft schematic dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--input', type=str, default='out',
                       help='Input directory containing .schem files')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory for processed data')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze dataset without processing')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output directories
    processed_dir = output_dir / 'processed'
    splits_dir = output_dir / 'splits'
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze dataset
    logger.info("Analyzing dataset...")
    stats = analyze_dataset(input_dir)
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Unique tags: {stats['unique_tags']}")
    logger.info(f"Size distribution: {stats['size_distribution']}")
    logger.info(f"Files with UUID: {stats['files_with_uuid']}")
    logger.info(f"\nTop 10 tags:")
    for tag, count in stats['top_tags'][:10]:
        logger.info(f"  {tag}: {count}")
    
    # Save statistics
    stats_file = output_dir / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    if args.analyze_only:
        return
    
    # Process all schematics
    logger.info("\nProcessing schematics...")
    
    filename_parser = FilenameParser()
    schematic_parser = SchematicParser(config['blocks'])
    
    all_files = list(input_dir.glob('*.schem'))
    metadata_list = []
    
    for file_path in tqdm(all_files, desc="Processing schematics"):
        try:
            # Parse filename
            file_meta = filename_parser.parse_filename(file_path.name)
            
            # Parse schematic
            voxel_array, schem_meta = schematic_parser.parse_schematic(file_path)
            
            # Classify size
            size_category = SizeClassifier.classify_size(
                schem_meta['width'],
                schem_meta['height'],
                schem_meta['length'],
                filename_size=file_meta['size']
            )
            
            # Pad/crop to standard size
            voxel_array = SizeClassifier.pad_or_crop(voxel_array, size_category)
            
            # Create text prompt
            prompt = create_text_prompt(file_meta)
            
            # Save voxel array
            voxel_filename = f"{file_path.stem}.npy"
            voxel_path = processed_dir / voxel_filename
            np.save(voxel_path, voxel_array)
            
            # Create metadata entry
            metadata = {
                'original_file': file_path.name,
                'voxel_file': voxel_filename,
                'size': size_category,
                'dimensions': SizeClassifier.get_target_dimensions(size_category),
                'prompt': prompt,
                'tags': file_meta['tags'],
                'materials': file_meta['materials'],
                'num_blocks': int(schem_meta['num_blocks'])
            }
            
            metadata_list.append(metadata)
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    logger.info(f"\nSuccessfully processed {len(metadata_list)} / {len(all_files)} files")
    
    # Create train/val/test splits
    logger.info("\nCreating dataset splits...")
    splits = DatasetSplitter.split_dataset(
        metadata_list,
        train_ratio=config['data']['train_split'],
        val_ratio=config['data']['val_split'],
        test_ratio=config['data']['test_split'],
        random_seed=config['data']['random_seed']
    )
    
    # Save splits
    for split_name, split_data in splits.items():
        split_file = splits_dir / f'{split_name}_metadata.json'
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved {split_name} split: {len(split_data)} samples to {split_file}")
    
    # Save complete metadata
    all_metadata_file = output_dir / 'all_metadata.json'
    with open(all_metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    logger.info(f"Saved complete metadata to {all_metadata_file}")
    
    logger.info("\n✓ Preprocessing complete!")
    logger.info(f"  - Processed data: {processed_dir}")
    logger.info(f"  - Metadata: {splits_dir}")
    logger.info(f"  - Train: {len(splits['train'])} samples")
    logger.info(f"  - Val: {len(splits['val'])} samples")
    logger.info(f"  - Test: {len(splits['test'])} samples")


if __name__ == "__main__":
    main()
