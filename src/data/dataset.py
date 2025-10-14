"""
PyTorch Dataset for Minecraft Schematics
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MinecraftSchematicDataset(Dataset):
    """Dataset for loading preprocessed Minecraft schematics"""
    
    def __init__(
        self,
        metadata_file: Path,
        data_dir: Path,
        text_encoder_name: str = "all-MiniLM-L6-v2",
        size_filter: Optional[List[str]] = None,
        transform=None,
        num_classes: int = 26
    ):
        """
        Args:
            metadata_file: Path to metadata JSON file
            data_dir: Directory containing .npy voxel files
            text_encoder_name: Sentence transformer model name
            size_filter: Optional list of sizes to include ['normal', 'big', 'huge']
            transform: Optional transform to apply to voxel data (True for augmentation, None for no augmentation)
            num_classes: Number of block types in vocabulary
        """
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter by size if specified
        if size_filter:
            self.metadata = [m for m in self.metadata if m['size'] in size_filter]
        
        logger.info(f"Loaded {len(self.metadata)} samples")
        
        # Load text encoder
        logger.info(f"Loading text encoder: {text_encoder_name}")
        self.text_encoder = SentenceTransformer(text_encoder_name)
        
        # Precompute text embeddings
        self._precompute_text_embeddings()
    
    def _precompute_text_embeddings(self):
        """Precompute all text embeddings for efficiency"""
        logger.info("Precomputing text embeddings...")
        prompts = [item['prompt'] for item in self.metadata]
        embeddings = self.text_encoder.encode(
            prompts,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        
        # Add embeddings to metadata
        for i, item in enumerate(self.metadata):
            item['text_embedding'] = embeddings[i]
        
        logger.info("Text embeddings computed")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Returns:
            Dictionary with:
                - voxels: (C, D, H, W) tensor of voxel data
                - text_embedding: Text embedding vector
                - size: Size category as integer
                - prompt: Original text prompt (for debugging)
        """
        item = self.metadata[idx]
        
        # Load voxel data
        voxel_file = self.data_dir / item['voxel_file']
        voxels = np.load(voxel_file)
        
        # Convert to tensor - use .clone() to ensure we have our own storage
        voxels = torch.from_numpy(voxels).long().clone()
        
        # One-hot encode
        voxels_onehot = torch.nn.functional.one_hot(
            voxels, 
            num_classes=self.num_classes
        )
        # Transpose to (C, D, H, W) and make contiguous
        voxels_onehot = voxels_onehot.permute(3, 0, 1, 2).float().contiguous()
        
        # Apply augmentation if this is training data
        if self.transform is not None:
            # Get spatial dimensions (D, H, W)
            _, D, H, W = voxels_onehot.shape
            
            # Only apply rotation if structure is cubic in the horizontal plane (D == W)
            # This prevents dimension mismatches for non-square structures
            if D == W:
                # Random 90-degree rotation around Y axis (vertical)
                if torch.rand(1) > 0.5:
                    k = torch.randint(1, 4, (1,)).item()
                    # Rotate in the D-W plane (dims 1 and 3, skipping H)
                    voxels_onehot = torch.rot90(voxels_onehot, k=k, dims=(1, 3))
            
            # Random flip along D axis (depth)
            if torch.rand(1) > 0.5:
                voxels_onehot = torch.flip(voxels_onehot, dims=[1])
            
            # Random flip along W axis (width) 
            if torch.rand(1) > 0.5:
                voxels_onehot = torch.flip(voxels_onehot, dims=[3])
        
        # Get text embedding - clone to ensure we have our own storage
        text_embedding = torch.from_numpy(item['text_embedding']).float().clone()
        
        # Size as categorical
        size_map = {'normal': 0, 'big': 1, 'huge': 2}
        size_category = torch.tensor(size_map[item['size']], dtype=torch.long)
        
        return {
            'voxels': voxels_onehot,
            'text_embedding': text_embedding,
            'size': size_category,
            'size_name': item['size'],
            'prompt': item['prompt']
        }


class OneHotVoxelTransform(nn.Module):
    """Transform voxel IDs to one-hot encoded format"""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of block types in vocabulary
        """
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxels: (D, H, W) tensor of block IDs
            
        Returns:
            (C, D, H, W) one-hot encoded tensor
        """
        # One-hot encode
        voxels_onehot = torch.nn.functional.one_hot(
            voxels, 
            num_classes=self.num_classes
        )
        
        # Transpose to (C, D, H, W)
        voxels_onehot = voxels_onehot.permute(3, 0, 1, 2).float()
        
        return voxels_onehot


class VoxelAugmentation(nn.Module):
    """Data augmentation for voxel data"""
    
    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        noise: float = 0.0
    ):
        """
        Args:
            rotation: Enable 90-degree rotations
            flip: Enable horizontal flips
            noise: Probability of random block replacement
        """
        super().__init__()
        self.rotation = rotation
        self.flip = flip
        self.noise = noise
    
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations
        
        Args:
            voxels: (C, D, H, W) or (D, H, W) tensor
            
        Returns:
            Augmented tensor
        """
        # Random 90-degree rotation around Y axis
        if self.rotation and torch.rand(1) > 0.5:
            k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 rotations
            if len(voxels.shape) == 4:
                voxels = torch.rot90(voxels, k=k, dims=(2, 3))
            else:
                voxels = torch.rot90(voxels, k=k, dims=(1, 2))
        
        # Random flip along X axis
        if self.flip and torch.rand(1) > 0.5:
            if len(voxels.shape) == 4:
                voxels = torch.flip(voxels, dims=[2])
            else:
                voxels = torch.flip(voxels, dims=[0])
        
        # Random flip along Z axis
        if self.flip and torch.rand(1) > 0.5:
            if len(voxels.shape) == 4:
                voxels = torch.flip(voxels, dims=[3])
            else:
                voxels = torch.flip(voxels, dims=[2])
        
        return voxels


def size_aware_collate(batch):
    """
    Custom collate function that only batches samples of the same size.
    If samples have different sizes, it will filter to keep only the most common size.
    """
    # Group by size
    size_groups = {}
    for item in batch:
        size_name = item['size_name']
        if size_name not in size_groups:
            size_groups[size_name] = []
        size_groups[size_name].append(item)
    
    # Use the largest group
    largest_group = max(size_groups.values(), key=len)
    
    # Now collate the largest group
    collated = {}
    collated['voxels'] = torch.stack([item['voxels'] for item in largest_group])
    collated['text_embedding'] = torch.stack([item['text_embedding'] for item in largest_group])
    collated['size'] = torch.stack([item['size'] for item in largest_group])
    collated['size_name'] = [item['size_name'] for item in largest_group]
    collated['prompt'] = [item['prompt'] for item in largest_group]
    
    return collated


def create_dataloaders(
    config: Dict,
    train_metadata: Path,
    val_metadata: Path,
    test_metadata: Optional[Path] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration dictionary
        train_metadata: Path to train metadata
        val_metadata: Path to val metadata
        test_metadata: Optional path to test metadata
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(config['data']['processed_dir'])
    text_encoder = config['model']['text_encoder']['model_name']
    batch_size = config['training']['batch_size']
    num_blocks = len(config['blocks'])
    
    # Create datasets (transforms are now applied in __getitem__)
    train_dataset = MinecraftSchematicDataset(
        train_metadata,
        data_dir,
        text_encoder_name=text_encoder,
        transform=True,  # Enable augmentation for training
        num_classes=num_blocks
    )
    
    val_dataset = MinecraftSchematicDataset(
        val_metadata,
        data_dir,
        text_encoder_name=text_encoder,
        transform=None,  # No augmentation for validation
        num_classes=num_blocks
    )
    
    # Determine if we should use pin_memory (only on CUDA)
    use_pin_memory = torch.cuda.is_available()
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=size_aware_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=size_aware_collate
    )
    
    test_loader = None
    if test_metadata:
        test_dataset = MinecraftSchematicDataset(
            test_metadata,
            data_dir,
            text_encoder_name=text_encoder,
            transform=None,  # No augmentation for test
            num_classes=num_blocks
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=size_aware_collate
        )
    
    logger.info(f"Created dataloaders: Train={len(train_dataset)}, "
               f"Val={len(val_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    import yaml
    
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Test dataset creation
        metadata_file = Path('data/splits/train_metadata.json')
        if metadata_file.exists():
            dataset = MinecraftSchematicDataset(
                metadata_file,
                Path('data/processed')
            )
            
            print(f"Dataset size: {len(dataset)}")
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Voxel shape: {sample['voxels'].shape}")
            print(f"Text embedding shape: {sample['text_embedding'].shape}")
            print(f"Prompt: {sample['prompt']}")
