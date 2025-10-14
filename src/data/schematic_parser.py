"""
Schematic Parser for Minecraft .schem files
Converts NBT format to 3D voxel arrays
"""

import nbtlib
from nbtlib import File, Compound, List, Int, Short, Byte, String, ByteArray
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SchematicParser:
    """Parse Minecraft .schem files (Sponge format)"""
    
    def __init__(self, block_vocab: Dict[str, int]):
        """
        Args:
            block_vocab: Dictionary mapping block names to integer IDs
        """
        self.block_vocab = block_vocab
        self.reverse_vocab = {v: k for k, v in block_vocab.items()}
        
    def parse_schematic(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Parse a .schem file and return voxel array
        
        Args:
            file_path: Path to .schem file
            
        Returns:
            voxel_array: 3D numpy array of block IDs (x, y, z)
            metadata: Dictionary with width, height, length, palette
        """
        try:
            # Load NBT file
            nbt = nbtlib.load(file_path)
            
            # Extract dimensions
            width = int(nbt['Width'])
            height = int(nbt['Height'])
            length = int(nbt['Length'])
            
            # Extract palette
            palette = {}
            if 'Palette' in nbt:
                for block_name, block_id in nbt['Palette'].items():
                    palette[int(block_id)] = block_name
            
            # Extract block data
            block_data = np.array(nbt['BlockData'], dtype=np.uint8)
            
            # Decode varint-encoded block data
            blocks = self._decode_block_data(block_data, width, height, length)
            
            # Convert palette indices to our vocabulary
            voxel_array = self._convert_to_vocab(blocks, palette, (width, height, length))
            
            metadata = {
                'width': width,
                'height': height,
                'length': length,
                'palette': palette,
                'num_blocks': np.sum(voxel_array != 0)
            }
            
            return voxel_array, metadata
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise
    
    def _decode_block_data(self, data: np.ndarray, width: int, height: int, length: int) -> np.ndarray:
        """
        Decode varint-encoded block data
        
        Args:
            data: Raw byte array
            width, height, length: Schematic dimensions
            
        Returns:
            Array of palette indices
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
        
        # Reshape to 3D array (y, z, x) -> transpose to (x, y, z)
        blocks_array = np.array(blocks, dtype=np.int32)
        
        # Pad if necessary
        expected_size = width * height * length
        if len(blocks_array) < expected_size:
            blocks_array = np.pad(blocks_array, (0, expected_size - len(blocks_array)))
        elif len(blocks_array) > expected_size:
            blocks_array = blocks_array[:expected_size]
        
        # Reshape: Sponge format uses (y, z, x) ordering
        blocks_3d = blocks_array.reshape(height, length, width)
        
        # Convert to (x, y, z)
        blocks_3d = np.transpose(blocks_3d, (2, 0, 1))
        
        return blocks_3d
    
    def _convert_to_vocab(self, blocks: np.ndarray, palette: Dict[int, str], 
                         dims: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert palette indices to vocabulary IDs
        
        Args:
            blocks: Array of palette indices
            palette: Mapping of palette index to block name
            dims: Target dimensions
            
        Returns:
            Array of vocabulary IDs
        """
        voxel_array = np.zeros(dims, dtype=np.int32)
        
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    palette_idx = blocks[i, j, k]
                    if palette_idx in palette:
                        block_name = palette[palette_idx]
                        # Extract base block name (remove properties)
                        base_name = self._extract_base_block(block_name)
                        vocab_id = self.block_vocab.get(base_name, 0)  # 0 = air
                        voxel_array[i, j, k] = vocab_id
        
        return voxel_array
    
    def _extract_base_block(self, block_string: str) -> str:
        """
        Extract base block name from full block string
        e.g., 'minecraft:oak_log[axis=y]' -> 'oak_log'
        """
        # Remove namespace
        if ':' in block_string:
            block_string = block_string.split(':')[1]
        
        # Remove properties
        if '[' in block_string:
            block_string = block_string.split('[')[0]
        
        return block_string
    
    def create_schematic(self, voxel_array: np.ndarray, output_path: Path,
                        palette: Optional[Dict[int, str]] = None) -> None:
        """
        Create a .schem file from a voxel array
        
        Args:
            voxel_array: 3D array of block IDs (x, y, z)
            output_path: Path to save .schem file
            palette: Optional custom palette, otherwise use default
        """
        width, height, length = voxel_array.shape
        
        # Create default palette if not provided
        if palette is None:
            palette = {i: f"minecraft:{name}" for name, i in self.block_vocab.items()}
        
        # Convert voxel array to palette indices
        # Transpose from (x, y, z) to (y, z, x) for Sponge format
        blocks_yxz = np.transpose(voxel_array, (1, 2, 0))
        blocks_flat = blocks_yxz.flatten()
        
        # Encode to varint
        block_data = self._encode_varint(blocks_flat)
        
        # Create NBT structure
        nbt = Compound({
            'Version': Int(2),
            'DataVersion': Int(2975),  # Minecraft 1.19
            'Width': Short(width),
            'Height': Short(height),
            'Length': Short(length),
            'Palette': Compound({palette[i]: Int(i) for i in set(blocks_flat) if i in palette}),
            'BlockData': ByteArray(list(block_data)),
            'Metadata': Compound({
                'WEOffsetX': Int(0),
                'WEOffsetY': Int(0),
                'WEOffsetZ': Int(0)
            })
        })
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        File(nbt).save(output_path, gzipped=True)
        logger.info(f"Saved schematic to {output_path}")
    
    def _encode_varint(self, values: np.ndarray) -> bytes:
        """Encode array of integers to varint format"""
        result = bytearray()
        
        for value in values:
            value = int(value)
            while True:
                byte = value & 0x7F
                value >>= 7
                if value != 0:
                    byte |= 0x80
                result.append(byte)
                if value == 0:
                    break
        
        return bytes(result)


def test_parser():
    """Test the schematic parser"""
    from pathlib import Path
    
    # Simple vocabulary
    vocab = {
        'air': 0,
        'oak_log': 1,
        'oak_leaves': 2,
        'birch_log': 3,
        'birch_leaves': 4
    }
    
    parser = SchematicParser(vocab)
    
    # Test parsing
    test_file = Path('out/beehive_oak_planks_04_01.schem')
    if test_file.exists():
        voxel, meta = parser.parse_schematic(test_file)
        print(f"Loaded schematic: {meta['width']}x{meta['height']}x{meta['length']}")
        print(f"Number of non-air blocks: {meta['num_blocks']}")
        print(f"Unique blocks: {np.unique(voxel)}")


if __name__ == "__main__":
    test_parser()
