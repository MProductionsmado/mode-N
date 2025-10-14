"""
Training module for 3D Diffusion Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from typing import Dict, Tuple, Optional
import logging

from src.models.diffusion_3d import DiffusionModel3D

logger = logging.getLogger(__name__)


class DiffusionLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training diffusion model"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        num_classes = len(config['blocks'])
        self.model = DiffusionModel3D(
            config=config,
            num_classes=num_classes,
            text_embed_dim=384  # Sentence-BERT embedding size
        )
        
        # Loss is simple MSE between predicted and actual noise
        self.loss_fn = nn.MSELoss()
        
    def forward(self, voxels: torch.Tensor, text_embed: torch.Tensor, size: str):
        """Forward pass"""
        return self.model(voxels, text_embed, size)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        voxels = batch['voxels']  # (B, D, H, W) - class indices
        text_embed = batch['text_embedding']  # (B, 384)
        size = batch['size_name'][0]  # All items in batch have same size
        
        # Convert to one-hot encoding
        num_classes = len(self.config['blocks'])
        voxels_onehot = F.one_hot(voxels.long(), num_classes=num_classes)
        voxels_onehot = voxels_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Forward pass (adds noise and predicts it)
        predicted_noise, actual_noise = self.forward(voxels_onehot, text_embed, size)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, actual_noise)
        
        # Log metrics
        self.log('train/loss_step', loss, prog_bar=True, batch_size=voxels.shape[0])
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Get average training loss
        avg_loss = self.trainer.callback_metrics.get('train/loss_step', 0.0)
        self.log('train/loss_epoch', avg_loss, prog_bar=True)
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        voxels = batch['voxels']
        text_embed = batch['text_embedding']
        size = batch['size_name'][0]
        
        # Convert to one-hot encoding
        num_classes = len(self.config['blocks'])
        voxels_onehot = F.one_hot(voxels.long(), num_classes=num_classes)
        voxels_onehot = voxels_onehot.permute(0, 4, 1, 2, 3).float()
        
        # Forward pass
        predicted_noise, actual_noise = self.forward(voxels_onehot, text_embed, size)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, actual_noise)
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, batch_size=voxels.shape[0])
        
        return loss
    
    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        voxels = batch['voxels']
        text_embed = batch['text_embedding']
        size = batch['size_name'][0]
        
        # Convert to one-hot encoding
        num_classes = len(self.config['blocks'])
        voxels_onehot = F.one_hot(voxels.long(), num_classes=num_classes)
        voxels_onehot = voxels_onehot.permute(0, 4, 1, 2, 3).float()
        
        # Forward pass
        predicted_noise, actual_noise = self.forward(voxels_onehot, text_embed, size)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, actual_noise)
        
        # Log metrics
        self.log('test/loss', loss, batch_size=voxels.shape[0])
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=self.config['training']['learning_rate'] * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def create_trainer(config: Dict, logger_name: str = "minecraft_diffusion") -> pl.Trainer:
    """Create PyTorch Lightning trainer with callbacks"""
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='minecraft-diffusion-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,  # Only save the best model
        save_last=True,  # And the last checkpoint
        every_n_epochs=config['training'].get('save_every_n_epochs', 50),
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
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if config['hardware']['device'] == 'cuda' else 'cpu',
        devices=1,
        precision=config['hardware']['precision'],
        callbacks=callbacks,
        log_every_n_steps=config['training']['log_every_n_steps'],
        gradient_clip_val=1.0,  # Clip gradients for stability
        accumulate_grad_batches=1,
        deterministic=False,  # Faster training
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer
