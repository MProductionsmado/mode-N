"""
UNet3D with integrated Minecraft-Aware Attention

This module provides a modified UNet3D that uses MinecraftAwareAttention3D
instead of standard AttentionBlock3D. It maintains full compatibility with
the existing DiscreteDiscreteDiffusionModel3D.

Usage:
    # In your training script:
    from src.models.unet_with_minecraft_attention import create_minecraft_aware_unet
    
    # Instead of UNet3D(...), use:
    unet = create_minecraft_aware_unet(
        config=config,
        in_channels=num_classes,
        out_channels=num_classes,
        blocks_dict=config['blocks']
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .discrete_diffusion_3d import (
    SinusoidalPositionEmbeddings,
    ConditionalGroupNorm,
    ResidualBlock3D
)
from .minecraft_attention import MinecraftAwareAttention3D


class MinecraftUNet3D(nn.Module):
    """
    Enhanced 3D UNet with Minecraft-aware attention
    
    This is a modified version of UNet3D that:
    1. Uses MinecraftAwareAttention3D instead of standard attention
    2. Passes block type information through the network
    3. Maintains full compatibility with existing training code
    
    Key differences:
    - Attention blocks are MinecraftAwareAttention3D
    - Forward pass computes current block predictions for attention masking
    - Better coherence for Minecraft structures (trees, buildings, etc.)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks_dict: Dict[str, int],
        model_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        cond_dim: int = 512,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1,
        use_minecraft_attention: bool = True
    ):
        """
        Args:
            in_channels: Input channels (num_classes for one-hot)
            out_channels: Output channels (num_classes for logits)
            blocks_dict: Block name -> ID mapping from config
            model_channels: Base channel count
            channel_multipliers: Channel multipliers per level
            num_res_blocks: Residual blocks per level
            cond_dim: Conditioning dimension (time + text)
            attention_levels: Which levels to add attention
            dropout: Dropout rate
            use_minecraft_attention: If True, use Minecraft-aware attention
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = out_channels
        self.channel_multipliers = channel_multipliers
        self.num_levels = len(channel_multipliers)
        self.use_minecraft_attention = use_minecraft_attention
        self.blocks_dict = blocks_dict
        
        # Initial convolution
        self.input_conv = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_ops = nn.ModuleList()
        
        ch = model_channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = model_channels * mult
            
            # Residual blocks for this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock3D(ch, out_ch, cond_dim, dropout))
                ch = out_ch
            
            # Attention at deeper levels
            if level in attention_levels:
                if use_minecraft_attention:
                    attn = MinecraftAwareAttention3D(
                        channels=out_ch,
                        num_classes=out_channels,
                        num_heads=8,
                        dropout=dropout
                    )
                    attn.set_block_categories(blocks_dict)
                    blocks.append(attn)
                else:
                    # Standard attention (for comparison)
                    from .discrete_diffusion_3d import AttentionBlock3D
                    blocks.append(AttentionBlock3D(out_ch, num_heads=8))
            
            self.encoder_blocks.append(blocks)
            
            # Downsample (except last level)
            if level < self.num_levels - 1:
                self.downsample_ops.append(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_ops.append(nn.Identity())
        
        # Bottleneck
        bottleneck_blocks = [
            ResidualBlock3D(ch, ch, cond_dim, dropout),
        ]
        
        if use_minecraft_attention:
            attn = MinecraftAwareAttention3D(
                channels=ch,
                num_classes=out_channels,
                num_heads=8,
                dropout=dropout
            )
            attn.set_block_categories(blocks_dict)
            bottleneck_blocks.append(attn)
        else:
            from .discrete_diffusion_3d import AttentionBlock3D
            bottleneck_blocks.append(AttentionBlock3D(ch, num_heads=8))
        
        bottleneck_blocks.append(ResidualBlock3D(ch, ch, cond_dim, dropout))
        self.bottleneck = nn.ModuleList(bottleneck_blocks)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = model_channels * mult
            
            # Upsample first (except first decoder level)
            if level < self.num_levels - 1:
                self.upsample_ops.append(nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1))
            else:
                self.upsample_ops.append(nn.Identity())
            
            # Residual blocks for this level (with skip connection)
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                # First block gets skip connection
                in_ch = ch + out_ch if i == 0 else out_ch
                blocks.append(ResidualBlock3D(in_ch, out_ch, cond_dim, dropout))
            
            # Attention at deeper levels
            if level in attention_levels:
                if use_minecraft_attention:
                    attn = MinecraftAwareAttention3D(
                        channels=out_ch,
                        num_classes=out_channels,
                        num_heads=8,
                        dropout=dropout
                    )
                    attn.set_block_categories(blocks_dict)
                    blocks.append(attn)
                else:
                    from .discrete_diffusion_3d import AttentionBlock3D
                    blocks.append(AttentionBlock3D(out_ch, num_heads=8))
            
            self.decoder_blocks.append(blocks)
            ch = out_ch
        
        # Output - predicts LOGITS for each block category
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with optional Minecraft-aware attention
        
        Args:
            x: Input (B, in_channels, D, H, W) - probability distributions
            t: Timesteps (B,)
            text_embed: Text embeddings (B, text_dim)
        
        Returns:
            Logits for each category (B, out_channels, D, H, W)
        """
        # Condition: concatenate time and text embeddings
        cond = torch.cat([t, text_embed], dim=1)  # (B, cond_dim)
        
        # Get current block predictions (for Minecraft attention)
        block_types = None
        if self.use_minecraft_attention:
            with torch.no_grad():
                # Argmax of input probabilities
                block_types = torch.argmax(x, dim=1)  # (B, D, H, W)
        
        # Initial conv
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skips = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsample_ops):
            for block in blocks:
                if isinstance(block, MinecraftAwareAttention3D):
                    h = block(h, block_types)
                elif hasattr(block, 'forward') and 'cond' in block.forward.__code__.co_varnames:
                    h = block(h, cond)
                else:
                    h = block(h)
            skips.append(h)
            h = downsample(h)
            
            # Update block_types after downsampling (if using Minecraft attention)
            if self.use_minecraft_attention and block_types is not None:
                # Downsample block_types to match h
                if not isinstance(downsample, nn.Identity):
                    block_types = F.max_pool3d(
                        block_types.unsqueeze(1).float(),
                        kernel_size=2,
                        stride=2
                    ).squeeze(1).long()
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, MinecraftAwareAttention3D):
                h = block(h, block_types)
            elif hasattr(block, 'forward') and 'cond' in block.forward.__code__.co_varnames:
                h = block(h, cond)
            else:
                h = block(h)
        
        # Decoder with skip connections
        for blocks, upsample in zip(self.decoder_blocks, self.upsample_ops):
            h = upsample(h)
            
            # Upsample block_types to match h (if using Minecraft attention)
            if self.use_minecraft_attention and block_types is not None:
                if not isinstance(upsample, nn.Identity):
                    block_types = F.interpolate(
                        block_types.unsqueeze(1).float(),
                        size=h.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
            
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, MinecraftAwareAttention3D):
                    h = block(h, block_types)
                elif hasattr(block, 'forward') and 'cond' in block.forward.__code__.co_varnames:
                    h = block(h, cond)
                else:
                    h = block(h)
        
        # Output logits
        return self.output_conv(h)


def create_minecraft_aware_unet(
    config: Dict,
    in_channels: int,
    out_channels: int,
    blocks_dict: Dict[str, int]
) -> MinecraftUNet3D:
    """
    Factory function to create a Minecraft-aware UNet from config
    
    Args:
        config: Full config dict
        in_channels: Input channels (num_classes)
        out_channels: Output channels (num_classes)
        blocks_dict: Block name -> ID mapping
    
    Returns:
        MinecraftUNet3D ready for training
    """
    model_config = config['model']
    
    # Calculate conditioning dimension
    time_embed_dim = model_config['diffusion'].get('time_embed_dim', 256)
    text_embed_dim = model_config['text_encoder']['embedding_dim']
    text_proj_dim = 256
    cond_dim = time_embed_dim + text_proj_dim
    
    unet = MinecraftUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        blocks_dict=blocks_dict,
        model_channels=model_config['encoder']['channels'][0],
        channel_multipliers=tuple(
            c // model_config['encoder']['channels'][0]
            for c in model_config['encoder']['channels']
        ),
        num_res_blocks=model_config['diffusion']['num_res_blocks'],
        cond_dim=cond_dim,
        attention_levels=tuple(model_config['diffusion']['attention_levels']),
        dropout=model_config['diffusion']['dropout'],
        use_minecraft_attention=True  # Enable Minecraft-aware attention
    )
    
    print(f"[MinecraftUNet] Created with {sum(p.numel() for p in unet.parameters()):,} parameters")
    print(f"[MinecraftUNet] Minecraft-aware attention: ENABLED")
    print(f"[MinecraftUNet] Attention levels: {model_config['diffusion']['attention_levels']}")
    
    return unet
