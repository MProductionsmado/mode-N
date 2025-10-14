"""
3D Diffusion Model for Minecraft Asset Generation
Based on DDPM (Denoising Diffusion Probabilistic Models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time step embeddings"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ConditionalGroupNorm(nn.Module):
    """Group normalization with FiLM conditioning"""
    
    def __init__(self, num_groups: int, num_channels: int, condition_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
        # FiLM: scale and shift parameters from conditioning
        self.film = nn.Linear(condition_dim, num_channels * 2)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        # condition: (B, condition_dim)
        normalized = self.norm(x)
        
        # Generate FiLM parameters
        film_params = self.film(condition)  # (B, 2*C)
        scale, shift = torch.chunk(film_params, 2, dim=1)  # Each (B, C)
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        
        return scale * normalized + shift


class ResidualBlock3D(nn.Module):
    """3D Residual block with time and text conditioning"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = ConditionalGroupNorm(8, out_channels, condition_dim)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalGroupNorm(8, out_channels, condition_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h, condition)
        h = F.silu(h)
        h = self.dropout(h)
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h, condition)
        h = F.silu(h)
        
        # Skip connection
        return h + self.skip(x)


class AttentionBlock3D(nn.Module):
    """3D Self-attention block for capturing long-range dependencies"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Q, K, V
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, D * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, DHW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        
        # Apply attention
        h = attn @ v  # (B, num_heads, DHW, head_dim)
        h = h.permute(0, 1, 3, 2).reshape(B, C, D, H, W)
        
        # Project and skip
        return x + self.proj(h)


class Downsample3D(nn.Module):
    """3D downsampling using strided convolution"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling using transposed convolution"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """3D UNet for diffusion model with time and text conditioning"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        text_embed_dim: int = 384,
        time_embed_dim: int = 256,
        channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # Combine time and text embeddings
        condition_dim = time_embed_dim + text_embed_dim
        
        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        
        ch = channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = channels * mult
            
            # Residual blocks at this level
            for _ in range(num_res_blocks):
                self.encoder.append(ResidualBlock3D(ch, out_ch, condition_dim, dropout))
                ch = out_ch
            
            # Attention at specified levels
            if level in attention_levels:
                self.encoder_attns.append(AttentionBlock3D(ch))
            else:
                self.encoder_attns.append(nn.Identity())
            
            # Downsample (except last level)
            if level < len(channel_multipliers) - 1:
                self.encoder.append(Downsample3D(ch))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock3D(ch, ch, condition_dim, dropout),
            AttentionBlock3D(ch),
            ResidualBlock3D(ch, ch, condition_dim, dropout)
        ])
        
        # Decoder (upsampling) - stored as nested list for clarity
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_multipliers)):
            out_ch = channels * mult
            
            blocks = nn.ModuleList()
            # First block takes concatenated input (current + skip)
            blocks.append(ResidualBlock3D(ch + ch, out_ch, condition_dim, dropout))
            # Remaining blocks take single input
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock3D(out_ch, out_ch, condition_dim, dropout))
            
            self.decoder_blocks.append(blocks)
            ch = out_ch
            
            # Attention at specified levels
            rev_level = len(channel_multipliers) - 1 - level
            if rev_level in attention_levels:
                self.decoder_attns.append(AttentionBlock3D(ch))
            else:
                self.decoder_attns.append(nn.Identity())
            
            # Upsample (except last level)
            if level < len(channel_multipliers) - 1:
                self.decoder_upsamples.append(Upsample3D(ch))
            else:
                self.decoder_upsamples.append(nn.Identity())
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy voxel data (B, in_channels, D, H, W)
            time: Timestep (B,)
            text_embed: Text embeddings (B, text_embed_dim)
        
        Returns:
            Predicted noise (B, out_channels, D, H, W)
        """
        # Compute time embedding
        t_emb = self.time_embed(time)  # (B, time_embed_dim)
        
        # Combine time and text embeddings
        condition = torch.cat([t_emb, text_embed], dim=1)  # (B, condition_dim)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skip_connections = []
        
        block_idx = 0
        attn_idx = 0
        for level in range(len(self.encoder)):
            if isinstance(self.encoder[level], Downsample3D):
                h = self.encoder[level](h)
            else:
                h = self.encoder[level](h, condition)
                h = self.encoder_attns[attn_idx](h)
                skip_connections.append(h)
                attn_idx += 1
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, AttentionBlock3D):
                h = block(h)
            else:
                h = block(h, condition)
        
        # Decoder with skip connections
        for level in range(len(self.decoder_blocks)):
            # Concatenate skip connection
            if skip_connections:
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
            
            # Apply residual blocks
            for block in self.decoder_blocks[level]:
                h = block(h, condition)
            
            # Apply attention
            h = self.decoder_attns[level](h)
            
            # Upsample
            h = self.decoder_upsamples[level](h)
        
        # Output
        return self.conv_out(h)


class DiffusionModel3D(nn.Module):
    """
    Complete 3D Diffusion Model for Minecraft Assets
    Handles noise scheduling and denoising process
    """
    
    def __init__(
        self,
        config: dict,
        num_classes: int,
        text_embed_dim: int = 384
    ):
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # Create separate UNets for each size
        self.unets = nn.ModuleDict()
        for size_name, size_config in config['model']['sizes'].items():
            self.unets[size_name] = UNet3D(
                in_channels=num_classes,  # One-hot encoded blocks
                out_channels=num_classes,  # Predict noise for each class
                text_embed_dim=text_embed_dim,
                time_embed_dim=config['model']['diffusion']['time_embed_dim'],
                channels=config['model']['encoder']['channels'][0],
                channel_multipliers=tuple(
                    c // config['model']['encoder']['channels'][0] 
                    for c in config['model']['encoder']['channels']
                ),
                num_res_blocks=config['model']['diffusion']['num_res_blocks'],
                attention_levels=tuple(config['model']['diffusion']['attention_levels']),
                dropout=config['model']['diffusion']['dropout']
            )
        
        # Noise schedule (DDPM)
        self.num_timesteps = config['model']['diffusion']['num_timesteps']
        self.register_noise_schedule(
            schedule=config['model']['diffusion']['noise_schedule'],
            timesteps=self.num_timesteps
        )
    
    def register_noise_schedule(
        self,
        schedule: str = 'linear',
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """Register noise schedule for diffusion process"""
        
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to x_start at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        size: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            x: Clean voxel data (B, num_classes, D, H, W) - one-hot encoded
            text_embed: Text embeddings (B, text_embed_dim)
            size: Size category ('normal', 'big', 'huge')
        
        Returns:
            predicted_noise: Predicted noise (B, num_classes, D, H, W)
            noise: Actual noise added (B, num_classes, D, H, W)
        """
        B = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to input (forward diffusion)
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        predicted_noise = self.unets[size](x_noisy, t, text_embed)
        
        return predicted_noise, noise
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
        size: str
    ) -> torch.Tensor:
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        Single denoising step
        """
        # Predict noise
        predicted_noise = self.unets[size](x, t, text_embed)
        
        # Get coefficients
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(alpha_t.shape) < len(x.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        
        # Predict x_0
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clip for stability
        
        # Get posterior mean
        posterior_mean_coef1 = self.posterior_mean_coef1[t]
        posterior_mean_coef2 = self.posterior_mean_coef2[t]
        while len(posterior_mean_coef1.shape) < len(x.shape):
            posterior_mean_coef1 = posterior_mean_coef1.unsqueeze(-1)
            posterior_mean_coef2 = posterior_mean_coef2.unsqueeze(-1)
        
        mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x
        
        # Add noise (except at t=0)
        if t[0] > 0:
            posterior_variance = self.posterior_variance[t]
            while len(posterior_variance.shape) < len(x.shape):
                posterior_variance = posterior_variance.unsqueeze(-1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def generate(
        self,
        text_embed: torch.Tensor,
        size: str,
        num_samples: int = 1,
        sampling_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion
        
        Args:
            text_embed: Text embeddings (B, text_embed_dim)
            size: Size category ('normal', 'big', 'huge')
            num_samples: Number of samples to generate
            sampling_steps: Number of sampling steps (None = use all timesteps)
        
        Returns:
            Generated voxel data (B, num_classes, D, H, W)
        """
        device = text_embed.device
        
        # Get dimensions for this size
        dims = self.config['model']['sizes'][size]['dims']
        D, H, W = dims
        
        # Start from pure noise
        x = torch.randn(num_samples, self.num_classes, D, H, W, device=device)
        
        # Use all timesteps or subset (DDIM-style)
        if sampling_steps is None:
            timesteps = range(self.num_timesteps - 1, -1, -1)
        else:
            # DDIM: sample subset of timesteps
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device
            )
        
        # Iterative denoising
        for t in timesteps:
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, text_embed, size)
        
        return x
