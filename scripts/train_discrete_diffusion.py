"""
Train Discrete Diffusion Model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import logging
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MinecraftSchematicDataset
from src.training.trainer_discrete_diffusion import DiscreteDiffusionLightningModule, create_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('medium')


def main():
    parser = argparse.ArgumentParser(description='Train Discrete Diffusion Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (fast_dev_run)')
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    logger.info("Creating datasets...")
    data_dir = Path(config['data']['processed_dir'])
    splits_dir = data_dir.parent / 'splits'
    num_classes = len(config['blocks'])
    
    train_dataset = MinecraftSchematicDataset(
        metadata_file=splits_dir / 'train_metadata.json',
        data_dir=data_dir,
        text_encoder_name=config['data'].get('text_encoder_name', 'all-MiniLM-L6-v2'),
        num_classes=num_classes
    )
    
    val_dataset = MinecraftSchematicDataset(
        metadata_file=splits_dir / 'val_metadata.json',
        data_dir=data_dir,
        text_encoder_name=config['data'].get('text_encoder_name', 'all-MiniLM-L6-v2'),
        num_classes=num_classes
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Custom collate function to handle different sizes
    def size_aware_collate(batch):
        """Group batch items by size to avoid dimension mismatch"""
        # Group by size
        size_groups = {}
        for item in batch:
            size = item['size_name']
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(item)
        
        # If all same size, use default collate
        if len(size_groups) == 1:
            size_name = list(size_groups.keys())[0]
            items = size_groups[size_name]
            return {
                'voxels': torch.stack([item['voxels'] for item in items]),
                'text_embedding': torch.stack([item['text_embedding'] for item in items]),
                'size': torch.stack([item['size'] for item in items]),
                'size_name': [item['size_name'] for item in items],
                'prompt': [item['prompt'] for item in items]
            }
        else:
            # Mixed sizes - just return first size group
            # (This shouldn't happen often with shuffling)
            size_name = list(size_groups.keys())[0]
            items = size_groups[size_name]
            return {
                'voxels': torch.stack([item['voxels'] for item in items]),
                'text_embedding': torch.stack([item['text_embedding'] for item in items]),
                'size': torch.stack([item['size'] for item in items]),
                'size_name': [item['size_name'] for item in items],
                'prompt': [item['prompt'] for item in items]
            }
    
    # Create dataloaders
    # Use num_workers=0 to avoid multiprocessing issues with tensor storage
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=False,
        collate_fn=size_aware_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=False,
        collate_fn=size_aware_collate
    )
    
    # Create model
    logger.info("Creating model...")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = DiscreteDiffusionLightningModule.load_from_checkpoint(args.resume, config=config)
    else:
        model = DiscreteDiffusionLightningModule(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(config, logger_name="minecraft_discrete_diffusion")
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Model: Discrete Diffusion (Multinomial)")
    logger.info(f"Noise schedule: {config['model']['diffusion']['noise_schedule']}")
    logger.info(f"Timesteps: {config['model']['diffusion']['num_timesteps']}")
    logger.info(f"Max epochs: {config['training']['num_epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    
    if args.debug:
        logger.info("Running in debug mode (fast_dev_run)")
        trainer.fast_dev_run = True
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
