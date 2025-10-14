"""
PyTorch Lightning Trainer for Conditional 3D VAE
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional
import logging

from ..models.vae_3d import ConditionalVAE3D
from ..training.losses import VAELoss, AccuracyMetric, NonAirAccuracyMetric

logger = logging.getLogger(__name__)


class VAELightningModule(pl.LightningModule):
    """PyTorch Lightning module for VAE training"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = ConditionalVAE3D(config)
        
        # Create loss
        self.loss_fn = VAELoss(
            reconstruction_weight=1.0,
            kl_weight=config['model']['vae']['beta'],
            kl_annealing=config['model']['vae'].get('kl_annealing', True),
            kl_annealing_epochs=config['model']['vae'].get('kl_annealing_epochs', 50)
        )
        
        # Metrics
        self.accuracy_metric = AccuracyMetric()
        self.non_air_accuracy = NonAirAccuracyMetric(air_id=0)
        
    def forward(self, voxels, text_emb, size_name):
        """Forward pass"""
        return self.model(voxels, text_emb, size_name)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step"""
        voxels = batch['voxels']
        text_emb = batch['text_embedding']
        
        # Get size_name - verify all are the same
        size_names = batch['size_name']
        size_name = size_names[0]
        
        # Forward pass
        recon, mu, logvar = self(voxels, text_emb, size_name)
        
        # Calculate loss
        loss, loss_dict = self.loss_fn(recon, voxels, mu, logvar)
        
        # Calculate metrics
        accuracy = self.accuracy_metric(recon, voxels)
        non_air_acc = self.non_air_accuracy(recon, voxels)
        
        # Get batch size for logging
        batch_size = voxels.size(0)
        
        # Log metrics
        self.log('train/loss', loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train/recon_loss', loss_dict['recon_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/kl_loss', loss_dict['kl_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/kl_weight', loss_dict['kl_weight'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train/non_air_accuracy', non_air_acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        voxels = batch['voxels']
        text_emb = batch['text_embedding']
        size_names = batch['size_name']
        size_name = size_names[0]
        
        # Forward pass
        recon, mu, logvar = self(voxels, text_emb, size_name)
        
        # Calculate loss
        loss, loss_dict = self.loss_fn(recon, voxels, mu, logvar)
        
        # Calculate metrics
        accuracy = self.accuracy_metric(recon, voxels)
        non_air_acc = self.non_air_accuracy(recon, voxels)
        
        # Get batch size for logging
        batch_size = voxels.size(0)
        
        # Log metrics
        self.log('val/loss', loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/recon_loss', loss_dict['recon_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/kl_loss', loss_dict['kl_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/non_air_accuracy', non_air_acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Test step - same as validation"""
        voxels = batch['voxels']
        text_emb = batch['text_embedding']
        size_names = batch['size_name']
        size_name = size_names[0]
        
        # Forward pass
        recon, mu, logvar = self(voxels, text_emb, size_name)
        
        # Calculate loss
        loss, loss_dict = self.loss_fn(recon, voxels, mu, logvar)
        
        # Calculate metrics
        accuracy = self.accuracy_metric(recon, voxels)
        non_air_acc = self.non_air_accuracy(recon, voxels)
        
        # Get batch size for logging
        batch_size = voxels.size(0)
        
        # Log metrics
        self.log('test/loss', loss_dict['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('test/recon_loss', loss_dict['recon_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test/kl_loss', loss_dict['kl_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('test/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('test/non_air_accuracy', non_air_acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Update KL annealing
        self.loss_fn.update_epoch(self.current_epoch)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Optimizer
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 1e-5)
        
        if optimizer_name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        scheduler_name = self.config['training'].get('scheduler', 'cosine').lower()
        num_epochs = self.config['training']['num_epochs']
        warmup_epochs = self.config['training'].get('warmup_epochs', 10)
        
        if scheduler_name == 'cosine':
            # Warmup + Cosine annealing
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=lr * 0.01
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def create_trainer(config: Dict, logger_name: str = "minecraft_vae") -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with callbacks
    
    Args:
        config: Configuration dictionary
        logger_name: Name for tensorboard logger
        
    Returns:
        PyTorch Lightning Trainer
    """
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='minecraft-vae-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config['training'].get('early_stopping', {}).get('enabled', True):
        early_stop_callback = EarlyStopping(
            monitor=config['training']['early_stopping'].get('monitor', 'val/loss'),
            patience=config['training']['early_stopping'].get('patience', 20),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=config['training']['log_dir'],
        name=logger_name
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if config['hardware']['device'] == 'cuda' else 'cpu',
        devices=1,
        precision=config['hardware'].get('precision', 32),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=config['training'].get('log_every_n_steps', 50),
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True
    )
    
    return trainer


def test_training():
    """Test training setup"""
    import yaml
    from pathlib import Path
    
    # Load config
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        print("Config file not found, skipping test")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create module
    module = VAELightningModule(config)
    
    # Create dummy batch
    batch = {
        'voxels': torch.randn(2, 26, 16, 16, 16),
        'text_embedding': torch.randn(2, 384),
        'size_name': ['normal', 'normal']
    }
    
    # Test forward pass
    recon, mu, logvar = module(
        batch['voxels'],
        batch['text_embedding'],
        batch['size_name'][0]
    )
    
    print(f"Reconstruction shape: {recon.shape}")
    print("Training setup test passed!")


if __name__ == "__main__":
    test_training()
