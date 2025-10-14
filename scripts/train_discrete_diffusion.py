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

from src.data.dataset import MinecraftDataset
from src.training.trainer_discrete_diffusion import DiscreteDiffusionLightningModule, create_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    train_dataset = MinecraftDataset(
        data_dir=config['data']['processed_dir'],
        split='train'
    )
    
    val_dataset = MinecraftDataset(
        data_dir=config['data']['processed_dir'],
        split='val'
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False
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
