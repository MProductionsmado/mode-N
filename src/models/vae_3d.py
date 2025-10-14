"""
3D Conditional Variational Autoencoder for Minecraft Assets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class Encoder3D(nn.Module):
    """3D CNN Encoder for voxel data"""
    
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        channels: list = [64, 128, 256, 512],
        input_size: Tuple[int, int, int] = (16, 16, 16)
    ):
        """
        Args:
            input_channels: Number of input channels (num_blocks for one-hot)
            latent_dim: Dimension of latent space
            channels: List of channel sizes for each layer
            input_size: Input spatial dimensions (D, H, W)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Build encoder layers
        layers = []
        in_ch = input_channels
        
        for i, out_ch in enumerate(channels):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(min(32, out_ch), out_ch),  # Use GroupNorm instead of BatchNorm
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        self.flat_size = self._get_flat_size()
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
    
    def _get_flat_size(self) -> int:
        """Calculate flattened size after convolutions"""
        dummy = torch.zeros(1, self.input_channels, *self.input_size)
        with torch.no_grad():
            x = self.encoder(dummy)
        return int(np.prod(x.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution
        
        Args:
            x: (B, C, D, H, W) input voxels
            
        Returns:
            mu: (B, latent_dim) mean
            logvar: (B, latent_dim) log variance
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder3D(nn.Module):
    """3D CNN Decoder with text conditioning"""
    
    def __init__(
        self,
        output_channels: int,
        latent_dim: int,
        text_dim: int,
        channels: list = [512, 256, 128, 64],
        output_size: Tuple[int, int, int] = (16, 16, 16)
    ):
        """
        Args:
            output_channels: Number of output channels (num_blocks)
            latent_dim: Dimension of latent space
            text_dim: Dimension of text embeddings
            channels: List of channel sizes for each layer
            output_size: Output spatial dimensions (D, H, W)
        """
        super().__init__()
        
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.output_size = output_size
        
        # Calculate initial spatial size
        # We have (len(channels) - 1) ConditionalDecoderBlocks + 1 final ConvTranspose3d
        # Each does 2x upsampling, so total is 2^len(channels)
        num_upsample_layers = len(channels)
        
        # Calculate required initial size to reach target output_size
        # Use ceiling division to handle non-power-of-2 dimensions
        import math
        self.init_size = tuple(math.ceil(s / (2 ** num_upsample_layers)) for s in output_size)
        
        # Combine latent and text
        self.fc = nn.Linear(latent_dim + text_dim, channels[0] * np.prod(self.init_size))
        
        # Build decoder layers with text injection
        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            # Add text conditioning to each layer
            layers.append(
                ConditionalDecoderBlock(
                    in_ch, out_ch, text_dim,
                    kernel_size=4, stride=2, padding=1
                )
            )
        
        self.decoder_blocks = nn.ModuleList(layers)
        
        # Final output layer with adaptive upsampling
        self.pre_output = nn.Sequential(
            nn.ConvTranspose3d(channels[-1], channels[-1], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(32, channels[-1]), channels[-1]),  # Use GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True)
        )
        
        # Final conv to get correct channels
        self.output_conv = nn.Conv3d(channels[-1], output_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to voxels
        
        Args:
            z: (B, latent_dim) latent vector
            text_emb: (B, text_dim) text embeddings
            
        Returns:
            (B, C, D, H, W) output voxels (logits)
        """
        # Combine latent and text
        x = torch.cat([z, text_emb], dim=1)
        x = self.fc(x)
        
        # Reshape to 3D
        batch_size = x.size(0)
        x = x.view(batch_size, -1, *self.init_size)
        
        # Decode with text conditioning
        for block in self.decoder_blocks:
            x = block(x, text_emb)
        
        # Upsample to near target size
        x = self.pre_output(x)
        
        # Interpolate to exact output size if needed
        current_size = x.shape[2:]  # (D, H, W)
        if current_size != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='trilinear', align_corners=False)
        
        # Final conv to get correct number of channels
        x = self.output_conv(x)
        
        return x


class ConditionalDecoderBlock(nn.Module):
    """Decoder block with text conditioning via FiLM"""
    
    def __init__(self, in_channels: int, out_channels: int, text_dim: int,
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.GroupNorm(min(32, out_channels), out_channels)  # Use GroupNorm instead of BatchNorm
        
        # FiLM: Feature-wise Linear Modulation
        self.film_gamma = nn.Linear(text_dim, out_channels)
        self.film_beta = nn.Linear(text_dim, out_channels)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)
            text_emb: (B, text_dim)
            
        Returns:
            (B, C_out, D', H', W')
        """
        x = self.conv(x)
        x = self.bn(x)
        
        # Apply FiLM conditioning
        gamma = self.film_gamma(text_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_beta(text_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x = gamma * x + beta
        x = self.activation(x)
        
        return x


class ConditionalVAE3D(nn.Module):
    """Conditional Variational Autoencoder for 3D voxels"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Get model parameters
        num_blocks = len(config['blocks'])
        text_dim = config['model']['text_encoder']['embedding_dim']
        encoder_channels = config['model']['encoder']['channels']
        decoder_channels = config['model']['decoder']['channels']
        
        # Create encoders/decoders for each size
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        for size_name, size_config in config['model']['sizes'].items():
            dims = tuple(size_config['dims'])
            latent_dim = size_config['latent_dim']
            
            self.encoders[size_name] = Encoder3D(
                input_channels=num_blocks,
                latent_dim=latent_dim,
                channels=encoder_channels,
                input_size=dims
            )
            
            self.decoders[size_name] = Decoder3D(
                output_channels=num_blocks,
                latent_dim=latent_dim,
                text_dim=text_dim,
                channels=decoder_channels,
                output_size=dims
            )
        
        # VAE parameters
        self.beta = config['model']['vae']['beta']
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        Args:
            mu: Mean
            logvar: Log variance
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        voxels: torch.Tensor, 
        text_emb: torch.Tensor,
        size_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            voxels: (B, C, D, H, W) input voxels
            text_emb: (B, text_dim) text embeddings
            size_name: Size category ('normal', 'big', 'huge')
            
        Returns:
            reconstruction: (B, C, D, H, W) reconstructed voxels (logits)
            mu: (B, latent_dim) latent mean
            logvar: (B, latent_dim) latent log variance
        """
        # DEBUG: Verify size_name is in the dictionaries
        if size_name not in self.encoders:
            raise KeyError(f"size_name '{size_name}' not found in encoders. Available: {list(self.encoders.keys())}")
        if size_name not in self.decoders:
            raise KeyError(f"size_name '{size_name}' not found in decoders. Available: {list(self.decoders.keys())}")
        
        # Encode
        encoder = self.encoders[size_name]
        mu, logvar = encoder(voxels)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoder = self.decoders[size_name]
        reconstruction = decoder(z, text_emb)
        
        # DEBUG: Check output size
        expected_size = self.config['model']['sizes'][size_name]['dims']
        actual_size = reconstruction.shape[2:]  # Skip batch and channel
        if tuple(actual_size) != tuple(expected_size):
            raise ValueError(
                f"Decoder output size mismatch for '{size_name}'! "
                f"Expected {expected_size}, got {tuple(actual_size)}"
            )
        
        return reconstruction, mu, logvar
    
    def generate(
        self,
        text_emb: torch.Tensor,
        size_name: str,
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate new voxels from text
        
        Args:
            text_emb: (B, text_dim) text embeddings
            size_name: Size category
            num_samples: Number of samples per prompt
            temperature: Sampling temperature
            
        Returns:
            (B*num_samples, C, D, H, W) generated voxels (logits)
        """
        batch_size = text_emb.size(0)
        latent_dim = self.config['model']['sizes'][size_name]['latent_dim']
        
        # Repeat text embeddings
        text_emb = text_emb.repeat(num_samples, 1)
        
        # Sample from prior
        z = torch.randn(batch_size * num_samples, latent_dim, device=text_emb.device)
        z = z * temperature
        
        # Decode
        decoder = self.decoders[size_name]
        generated = decoder(z, text_emb)
        
        return generated
    
    def encode(
        self,
        voxels: torch.Tensor,
        size_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode voxels to latent space"""
        encoder = self.encoders[size_name]
        return encoder(voxels)
    
    def decode(
        self,
        z: torch.Tensor,
        text_emb: torch.Tensor,
        size_name: str
    ) -> torch.Tensor:
        """Decode latent vector to voxels"""
        decoder = self.decoders[size_name]
        return decoder(z, text_emb)


def test_model():
    """Test model architecture"""
    # Dummy config
    config = {
        'blocks': {f'block_{i}': i for i in range(26)},
        'model': {
            'sizes': {
                'normal': {'dims': [16, 16, 16], 'latent_dim': 128},
                'big': {'dims': [16, 32, 16], 'latent_dim': 192},
            },
            'text_encoder': {'embedding_dim': 384},
            'encoder': {'channels': [64, 128, 256, 512]},
            'decoder': {'channels': [512, 256, 128, 64]},
            'vae': {'beta': 0.5}
        }
    }
    
    model = ConditionalVAE3D(config)
    
    # Test forward pass
    batch_size = 2
    voxels = torch.randn(batch_size, 26, 16, 16, 16)
    text_emb = torch.randn(batch_size, 384)
    
    recon, mu, logvar = model(voxels, text_emb, 'normal')
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test generation
    generated = model.generate(text_emb, 'normal', num_samples=3)
    print(f"Generated shape: {generated.shape}")
    
    print("Model test passed!")


if __name__ == "__main__":
    test_model()
