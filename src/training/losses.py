"""
Loss functions for 3D VAE training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VAELoss(nn.Module):
    """Combined loss for VAE: Reconstruction + KL Divergence"""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.5,
        kl_annealing: bool = True,
        kl_annealing_epochs: int = 50
    ):
        """
        Args:
            reconstruction_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence (beta in beta-VAE)
            kl_annealing: Whether to anneal KL weight during training
            kl_annealing_epochs: Number of epochs to anneal over
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_annealing_epochs = kl_annealing_epochs
        
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int):
        """Update current epoch for KL annealing"""
        self.current_epoch = epoch
    
    def get_kl_weight(self) -> float:
        """Get current KL weight with annealing"""
        if not self.kl_annealing:
            return self.kl_weight
        
        # Linear annealing from 0 to kl_weight
        progress = min(1.0, self.current_epoch / self.kl_annealing_epochs)
        return self.kl_weight * progress
    
    def reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruction loss (cross-entropy)
        
        Args:
            recon: (B, C, D, H, W) reconstructed logits
            target: (B, C, D, H, W) target one-hot
            
        Returns:
            Scalar loss
        """
        # Convert target from one-hot to class indices
        target_indices = torch.argmax(target, dim=1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(recon, target_indices, reduction='mean')
        
        return loss
    
    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence between N(mu, var) and N(0, 1)
        
        Args:
            mu: (B, latent_dim) mean
            logvar: (B, latent_dim) log variance
            
        Returns:
            Scalar loss
        """
        # KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Average over batch
        kl = kl / mu.size(0)
        
        return kl
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate total loss
        
        Args:
            recon: Reconstructed voxels (logits)
            target: Target voxels (one-hot)
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(recon, target)
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Get current KL weight
        current_kl_weight = self.get_kl_weight()
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            current_kl_weight * kl_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_weight': current_kl_weight
        }
        
        return total_loss, loss_dict


class PerceptualVoxelLoss(nn.Module):
    """Additional perceptual loss for better structure preservation"""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate perceptual loss based on feature similarity
        
        Args:
            recon: (B, C, D, H, W) reconstructed logits
            target: (B, C, D, H, W) target one-hot
            
        Returns:
            Scalar loss
        """
        # Convert recon to probabilities
        recon_probs = F.softmax(recon, dim=1)
        
        # Calculate feature maps (sum over spatial dimensions)
        recon_features = recon_probs.sum(dim=[2, 3, 4])  # (B, C)
        target_features = target.sum(dim=[2, 3, 4])  # (B, C)
        
        # Cosine similarity loss
        loss = 1 - F.cosine_similarity(recon_features, target_features, dim=1).mean()
        
        return loss


class AccuracyMetric(nn.Module):
    """Block-wise accuracy metric"""
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Calculate accuracy
        
        Args:
            recon: (B, C, D, H, W) reconstructed logits
            target: (B, C, D, H, W) target one-hot
            
        Returns:
            Accuracy as float
        """
        # Get predictions
        pred_indices = torch.argmax(recon, dim=1)
        target_indices = torch.argmax(target, dim=1)
        
        # Calculate accuracy
        correct = (pred_indices == target_indices).float()
        accuracy = correct.mean().item()
        
        return accuracy


class NonAirAccuracyMetric(nn.Module):
    """Accuracy metric for non-air blocks only"""
    
    def __init__(self, air_id: int = 0):
        super().__init__()
        self.air_id = air_id
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Calculate accuracy for non-air blocks
        
        Args:
            recon: (B, C, D, H, W) reconstructed logits
            target: (B, C, D, H, W) target one-hot
            
        Returns:
            Accuracy as float
        """
        # Get predictions
        pred_indices = torch.argmax(recon, dim=1)
        target_indices = torch.argmax(target, dim=1)
        
        # Mask out air blocks
        non_air_mask = target_indices != self.air_id
        
        if non_air_mask.sum() == 0:
            return 0.0
        
        # Calculate accuracy only for non-air blocks
        correct = (pred_indices == target_indices) & non_air_mask
        accuracy = correct.sum().float() / non_air_mask.sum().float()
        
        return accuracy.item()


def test_losses():
    """Test loss functions"""
    # Dummy data
    batch_size = 2
    num_classes = 26
    size = (8, 8, 8)
    
    recon = torch.randn(batch_size, num_classes, *size)
    target = torch.zeros(batch_size, num_classes, *size)
    
    # Random one-hot target
    for b in range(batch_size):
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    c = torch.randint(0, num_classes, (1,)).item()
                    target[b, c, i, j, k] = 1
    
    mu = torch.randn(batch_size, 128)
    logvar = torch.randn(batch_size, 128)
    
    # Test VAE loss
    vae_loss = VAELoss()
    total_loss, loss_dict = vae_loss(recon, target, mu, logvar)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test accuracy
    acc_metric = AccuracyMetric()
    accuracy = acc_metric(recon, target)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Test non-air accuracy
    non_air_acc = NonAirAccuracyMetric(air_id=0)
    non_air_accuracy = non_air_acc(recon, target)
    print(f"Non-air accuracy: {non_air_accuracy:.4f}")
    
    print("Loss tests passed!")


if __name__ == "__main__":
    test_losses()
