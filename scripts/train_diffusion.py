"""
Train the 3D Diffusion Model for Minecraft Assets
"""

import argparse
import yaml
from pathlib import Path
import logging
import sys
import os
import torch

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.training.trainer_diffusion import DiffusionLightningModule, create_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Minecraft 3D Diffusion Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (fast_dev_run)')
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Check for CUDA
    if config['hardware']['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config['hardware']['device'] = 'cpu'
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    data_dir = Path(config['data']['output_dir'])
    
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        train_metadata=data_dir / 'splits' / 'train_metadata.json',
        val_metadata=data_dir / 'splits' / 'val_metadata.json',
        test_metadata=data_dir / 'splits' / 'test_metadata.json',
        num_workers=config['hardware']['num_workers']
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating Diffusion Model...")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = DiffusionLightningModule.load_from_checkpoint(args.resume, config=config)
    else:
        model = DiffusionLightningModule(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(config, logger_name="minecraft_diffusion")
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Model: 3D Diffusion (DDPM)")
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
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    
    # Test if test loader exists
    if test_loader:
        logger.info("\nRunning test evaluation...")
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
