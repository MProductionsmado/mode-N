"""
Preprocessing pipeline for schematic dataset
- Parse filenames for tags and metadata
- Extract text descriptions
- Determine size categories
- Create train/val/test splits
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class FilenameParser:
    """Parse schematic filenames to extract metadata and tags"""
    
    # UUID pattern
    UUID_PATTERN = re.compile(r'--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
    
    # Size tags
    SIZE_TAGS = ['huge', 'big', 'small']
    
    def __init__(self):
        self.tag_counter = Counter()
        self.material_counter = Counter()
        
    def parse_filename(self, filename: str) -> Dict:
        """
        Parse a schematic filename into components
        
        Args:
            filename: e.g., "big_birch_wood_birch_leaves_fall_09_05--uuid.schem"
            
        Returns:
            Dictionary with: size, tags, materials, description, numeric_suffix
        """
        # Remove extension
        name = filename.replace('.schem', '')
        
        # Remove UUID if present
        name_no_uuid = self.UUID_PATTERN.sub('', name)
        
        # Split by underscores
        parts = name_no_uuid.strip('-_').split('_')
        
        # Extract size tag
        size = 'normal'
        if parts[0] in self.SIZE_TAGS:
            size = parts[0]
            parts = parts[1:]  # Remove size from parts
        
        # Extract numeric suffix (e.g., "09_05")
        numeric_suffix = []
        while parts and self._is_numeric(parts[-1]):
            numeric_suffix.insert(0, parts.pop())
        
        # Remaining parts are tags/materials
        tags = parts
        
        # Identify materials (wood types, leaf types, etc.)
        materials = self._extract_materials(tags)
        
        # Create human-readable description
        description = ' '.join(parts)
        
        # Update counters
        for tag in tags:
            self.tag_counter[tag] += 1
        for material in materials:
            self.material_counter[material] += 1
        
        return {
            'filename': filename,
            'size': size,
            'tags': tags,
            'materials': materials,
            'description': description,
            'numeric_suffix': '_'.join(numeric_suffix) if numeric_suffix else '',
            'has_uuid': '--' in filename
        }
    
    def _is_numeric(self, s: str) -> bool:
        """Check if string is numeric"""
        try:
            int(s)
            return True
        except ValueError:
            return False
    
    def _extract_materials(self, tags: List[str]) -> List[str]:
        """Extract material tags from tag list"""
        materials = []
        material_keywords = [
            'oak', 'birch', 'spruce', 'acacia', 'dark_oak', 'jungle',
            'wood', 'log', 'leaves', 'planks',
            'stone', 'cobblestone', 'dirt', 'grass'
        ]
        
        for tag in tags:
            if any(keyword in tag.lower() for keyword in material_keywords):
                materials.append(tag)
        
        return materials
    
    def get_statistics(self) -> Dict:
        """Get statistics about parsed filenames"""
        return {
            'total_files': sum(self.tag_counter.values()),
            'unique_tags': len(self.tag_counter),
            'top_tags': self.tag_counter.most_common(20),
            'unique_materials': len(self.material_counter),
            'top_materials': self.material_counter.most_common(10)
        }


class SizeClassifier:
    """Classify schematics into size categories based on dimensions"""
    
    SIZE_CATEGORIES = {
        'normal': {
            'target': (16, 16, 16),
            'max_dim': (16, 16, 16)
        },
        'big': {
            'target': (16, 32, 16),
            'max_dim': (16, 32, 16)
        },
        'huge': {
            'target': (24, 64, 24),
            'max_dim': (24, 64, 24)
        }
    }
    
    @classmethod
    def classify_size(cls, width: int, height: int, length: int, 
                     filename_size: Optional[str] = None) -> str:
        """
        Classify schematic size based on dimensions
        
        Args:
            width, height, length: Actual dimensions
            filename_size: Size hint from filename
            
        Returns:
            Size category: 'normal', 'big', or 'huge'
        """
        max_dim = max(width, height, length)
        
        # Use filename hint if available
        if filename_size and filename_size in cls.SIZE_CATEGORIES:
            return filename_size
        
        # Otherwise classify by dimensions
        if max_dim <= 16:
            return 'normal'
        elif max_dim <= 32:
            return 'big'
        else:
            return 'huge'
    
    @classmethod
    def get_target_dimensions(cls, size_category: str) -> Tuple[int, int, int]:
        """Get target dimensions for a size category"""
        return cls.SIZE_CATEGORIES[size_category]['target']
    
    @classmethod
    def pad_or_crop(cls, voxel_array: np.ndarray, target_size: str) -> np.ndarray:
        """
        Pad or crop voxel array to target size
        
        Args:
            voxel_array: Input 3D array
            target_size: 'normal', 'big', or 'huge'
            
        Returns:
            Resized array
        """
        target_dims = cls.get_target_dimensions(target_size)
        current_dims = voxel_array.shape
        
        result = np.zeros(target_dims, dtype=voxel_array.dtype)
        
        # Calculate how much to copy
        copy_dims = tuple(min(c, t) for c, t in zip(current_dims, target_dims))
        
        # Copy data (centered)
        x_offset = (target_dims[0] - copy_dims[0]) // 2
        y_offset = 0  # Start from bottom for Y (structures grow upward)
        z_offset = (target_dims[2] - copy_dims[2]) // 2
        
        result[
            x_offset:x_offset + copy_dims[0],
            y_offset:y_offset + copy_dims[1],
            z_offset:z_offset + copy_dims[2]
        ] = voxel_array[:copy_dims[0], :copy_dims[1], :copy_dims[2]]
        
        return result


class DatasetSplitter:
    """Create train/val/test splits with stratification"""
    
    @staticmethod
    def split_dataset(metadata_list: List[Dict], 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     random_seed: int = 42) -> Dict[str, List[Dict]]:
        """
        Split dataset with stratification by size category
        
        Args:
            metadata_list: List of metadata dictionaries
            train_ratio, val_ratio, test_ratio: Split ratios
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        np.random.seed(random_seed)
        
        # Group by size category
        size_groups = {}
        for item in metadata_list:
            size = item.get('size', 'normal')
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(item)
        
        # Split each group
        splits = {'train': [], 'val': [], 'test': []}
        
        for size, items in size_groups.items():
            # Shuffle
            np.random.shuffle(items)
            
            n = len(items)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits['train'].extend(items[:train_end])
            splits['val'].extend(items[train_end:val_end])
            splits['test'].extend(items[val_end:])
        
        # Shuffle final splits
        for split in splits.values():
            np.random.shuffle(split)
        
        logger.info(f"Dataset split: Train={len(splits['train'])}, "
                   f"Val={len(splits['val'])}, Test={len(splits['test'])}")
        
        return splits


def create_text_prompt(metadata: Dict) -> str:
    """
    Create a text prompt from metadata for conditioning
    
    Args:
        metadata: Parsed filename metadata
        
    Returns:
        Text prompt string
    """
    parts = []
    
    # Add size
    if metadata['size'] != 'normal':
        parts.append(metadata['size'])
    
    # Add description (cleaned up)
    description = metadata['description'].replace('_', ' ')
    parts.append(description)
    
    return ' '.join(parts)


def analyze_dataset(input_dir: Path) -> Dict:
    """
    Analyze the entire dataset and return statistics
    
    Args:
        input_dir: Directory containing .schem files
        
    Returns:
        Statistics dictionary
    """
    parser = FilenameParser()
    
    files = list(input_dir.glob('*.schem'))
    logger.info(f"Found {len(files)} schematic files")
    
    metadata_list = []
    for file in files:
        meta = parser.parse_filename(file.name)
        metadata_list.append(meta)
    
    # Get statistics
    stats = parser.get_statistics()
    
    # Size distribution
    size_dist = Counter(m['size'] for m in metadata_list)
    stats['size_distribution'] = dict(size_dist)
    
    # UUID presence
    uuid_count = sum(1 for m in metadata_list if m['has_uuid'])
    stats['files_with_uuid'] = uuid_count
    stats['files_without_uuid'] = len(files) - uuid_count
    
    return stats


if __name__ == "__main__":
    # Test preprocessing
    from pathlib import Path
    
    test_names = [
        "big_birch_wood_birch_leaves_fall_09_05--a76398a7-f427-401f-99b1-af10895c252b.schem",
        "beehive_oak_planks_04_01.schem",
        "huge_oak_tree_13_42.schem"
    ]
    
    parser = FilenameParser()
    
    for name in test_names:
        result = parser.parse_filename(name)
        prompt = create_text_prompt(result)
        print(f"\nFile: {name}")
        print(f"  Size: {result['size']}")
        print(f"  Tags: {result['tags']}")
        print(f"  Materials: {result['materials']}")
        print(f"  Prompt: {prompt}")
