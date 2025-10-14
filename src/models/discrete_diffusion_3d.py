"""
Discrete Diffusion Model for 3D Voxel Generation
Uses Multinomial Diffusion instead of Gaussian Noise
Better for categorical data like Minecraft blocks

Based on:
- "Argmax Flows and Multinomial Diffusion" (Hoogeboom et al. 2021)
- "Structured Denoising Diffusion Models in Discrete State-Spaces" (Austin et al. 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings (same as before)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalGroupNorm(nn.Module):
    """FiLM conditioning (same as before)"""
    
    def __init__(self, num_groups: int, num_channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scale = nn.Linear(cond_dim, num_channels)
        self.shift = nn.Linear(cond_dim, num_channels)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        # cond: (B, cond_dim)
        x = self.norm(x)
        scale = self.scale(cond)[:, :, None, None, None]
        shift = self.shift(cond)[:, :, None, None, None]
        return x * (1 + scale) + shift


class ResidualBlock3D(nn.Module):
    """3D Residual Block with FiLM conditioning (same as before)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = ConditionalGroupNorm(8, out_channels, cond_dim)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalGroupNorm(8, out_channels, cond_dim)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h, cond)
        h = self.activation(h)
        h = self.dropout(h)
        
        h = self.conv2(h)
        h = self.norm2(h, cond)
        h = self.activation(h)
        
        return h + self.shortcut(x)


class AttentionBlock3D(nn.Module):
    """3D Self-Attention (same as before)"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape to (B, num_heads, head_dim, D*H*W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, D, H, W)
        out = self.proj(out)
        
        return x + out


class UNet3D(nn.Module):
    """
    3D UNet for Discrete Diffusion
    REUSED from continuous diffusion - only output layer changes
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        cond_dim: int = 512,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_multipliers = channel_multipliers
        self.num_levels = len(channel_multipliers)
        
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
                blocks.append(AttentionBlock3D(out_ch))
            
            self.encoder_blocks.append(blocks)
            
            # Downsample (except last level)
            if level < self.num_levels - 1:
                self.downsample_ops.append(nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            else:
                self.downsample_ops.append(nn.Identity())
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock3D(ch, ch, cond_dim, dropout),
            AttentionBlock3D(ch),
            ResidualBlock3D(ch, ch, cond_dim, dropout)
        ])
        
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
                blocks.append(AttentionBlock3D(out_ch))
            
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
        Forward pass
        
        Args:
            x: Input (B, in_channels, D, H, W) - one-hot encoded
            t: Timesteps (B,)
            text_embed: Text embeddings (B, text_dim)
        
        Returns:
            Logits for each category (B, out_channels, D, H, W)
        """
        # Condition: concatenate time and text embeddings
        cond = torch.cat([t, text_embed], dim=1)  # (B, cond_dim)
        
        # Initial conv
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skips = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsample_ops):
            for block in blocks:
                if isinstance(block, AttentionBlock3D):
                    h = block(h)
                else:
                    h = block(h, cond)
            skips.append(h)
            h = downsample(h)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, AttentionBlock3D):
                h = block(h)
            else:
                h = block(h, cond)
        
        # Decoder with skip connections
        for blocks, upsample in zip(self.decoder_blocks, self.upsample_ops):
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, AttentionBlock3D):
                    h = block(h)
                else:
                    h = block(h, cond)
        
        # Output logits
        return self.output_conv(h)


class DiscreteDiscreteDiffusionModel3D(nn.Module):
    """
    Discrete Diffusion Model using Multinomial Transitions
    Better for categorical data like Minecraft blocks
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_classes = len(config['blocks'])
        self.num_timesteps = config['model']['diffusion']['num_timesteps']
        
        # Time embeddings
        time_embed_dim = 256
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU()
        )
        
        # Text embeddings projection
        text_embed_dim = config['model']['text_encoder']['embedding_dim']
        text_proj_dim = 256
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, text_proj_dim),
            nn.SiLU()
        )
        
        cond_dim = time_embed_dim + text_proj_dim
        
        # Create UNets for each size
        self.unets = nn.ModuleDict()
        for size_name, size_config in config['model']['sizes'].items():
            self.unets[size_name] = UNet3D(
                in_channels=self.num_classes,
                out_channels=self.num_classes,  # Predict logits for each class
                model_channels=config['model']['encoder']['channels'][0],
                channel_multipliers=tuple(
                    c // config['model']['encoder']['channels'][0] 
                    for c in config['model']['encoder']['channels']
                ),
                num_res_blocks=config['model']['diffusion']['num_res_blocks'],
                cond_dim=cond_dim,
                attention_levels=tuple(config['model']['diffusion']['attention_levels']),
                dropout=config['model']['diffusion']['dropout']
            )
        
        # Transition matrix schedule (probability of staying in same state)
        # At t=0: almost always stay, at t=T: uniform distribution
        betas = self._cosine_beta_schedule(self.num_timesteps)
        self.register_buffer('betas', betas)
        
        # Cumulative product of (1 - beta)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule for transition probabilities
        From "Improved Denoising Diffusion Probabilistic Models"
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        Gradually transition to uniform distribution
        
        Args:
            x_start: One-hot encoded voxels (B, C, D, H, W)
            t: Timesteps (B,)
        
        Returns:
            Noised one-hot (B, C, D, H, W)
        """
        # Get transition probability for timestep t
        alpha_cumprod_t = self.alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(alpha_cumprod_t.shape) < len(x_start.shape):
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
        
        # Probability of staying in same state
        stay_prob = alpha_cumprod_t
        
        # Probability of transitioning to uniform
        uniform_prob = (1.0 - alpha_cumprod_t) / self.num_classes
        
        # Create transition: stay in same state with prob stay_prob,
        # otherwise uniform over all states
        noised = x_start * stay_prob + uniform_prob
        
        return noised
    
    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        size: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            x: Clean one-hot voxels (B, C, D, H, W)
            text_embed: Text embeddings (B, text_dim)
            size: Size category
        
        Returns:
            predicted_logits: (B, C, D, H, W)
            target_onehot: (B, C, D, H, W)
            t: (B,)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Forward diffusion
        x_t = self.q_sample(x, t)
        
        # Project embeddings
        time_embed = self.time_embed(t.float())
        text_proj = self.text_proj(text_embed)
        
        # Predict original one-hot from noised version
        predicted_logits = self.unets[size](x_t, time_embed, text_proj)
        # Return predicted logits and original clean target
        return predicted_logits, x, t
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
        text_proj: torch.Tensor,
        size: str
    ) -> torch.Tensor:
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        Single denoising step for discrete data
        """
        # Predict logits for x_0 (clean data estimate)
        time_embed = self.time_embed(t.float())
        predicted_logits = self.unets[size](x, time_embed, text_proj)

        # Estimate x_0 distribution
        x_0_pred = F.softmax(predicted_logits, dim=1)

        if t[0] > 0:
            # Previous cumulative alpha (probability of staying in same state up to t-1)
            alpha_cumprod_t_prev = self.alphas_cumprod[t - 1]
            while len(alpha_cumprod_t_prev.shape) < len(x.shape):
                alpha_cumprod_t_prev = alpha_cumprod_t_prev.unsqueeze(-1)

            # Prior over x_{t-1} constructed from predicted clean data
            stay_prob_prev = alpha_cumprod_t_prev
            uniform_prob_prev = (1.0 - alpha_cumprod_t_prev) / self.num_classes
            prior_prev = x_0_pred * stay_prob_prev + uniform_prob_prev

            # Incorporate current noisy state x (acts like likelihood term)
            # Element-wise product then renormalize
            posterior_unnorm = prior_prev * (x + 1e-8)
            posterior = posterior_unnorm / (posterior_unnorm.sum(dim=1, keepdim=True) + 1e-8)
            return posterior
        else:
            # Final step: return categorical distribution for x_0
            return x_0_pred
    
    @torch.no_grad()
    def generate(
        self,
        text_embed: torch.Tensor,
        size: str,
        num_samples: int = 1,
        sampling_steps: Optional[int] = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion with Classifier-Free Guidance
        
        Args:
            text_embed: Text embeddings (B, text_embed_dim)
            size: Size category
            num_samples: Number of samples
            sampling_steps: Number of steps (None = all timesteps)
            guidance_scale: CFG scale (1.0=no guidance, 3.0=balanced, 7.5=strong)
        
        Returns:
            Generated one-hot voxels (B, C, D, H, W)
        """
        device = text_embed.device
        
        # Get dimensions
        dims = self.config['model']['sizes'][size]['dims']
        D, H, W = dims
        
        # Project text embeddings
        text_proj = self.text_proj(text_embed)
        
        # Create unconditional embedding (zeros) for CFG
        uncond_embed = torch.zeros_like(text_embed)
        uncond_proj = self.text_proj(uncond_embed)
        
        # Start from uniform distribution over classes
        x = torch.ones(num_samples, self.num_classes, D, H, W, device=device) / self.num_classes
        
        # Determine timesteps
        if sampling_steps is None:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            # DDIM-style: subset of timesteps
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device
            ).tolist()
        
        # Iterative denoising with CFG
        for t in timesteps:
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            if guidance_scale == 1.0:
                # No guidance - standard generation
                x = self.p_sample(x, t_batch, text_embed, text_proj, size)
            else:
                # Classifier-Free Guidance
                x = self.p_sample_cfg(x, t_batch, text_embed, text_proj, 
                                     uncond_embed, uncond_proj, size, guidance_scale)
        
        return x
    
    def p_sample_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_embed: torch.Tensor,
        cond_proj: torch.Tensor,
        uncond_embed: torch.Tensor,
        uncond_proj: torch.Tensor,
        size: str,
        guidance_scale: float
    ) -> torch.Tensor:
        """
        Sample with Classifier-Free Guidance
        
        Args:
            x: Current state (B, C, D, H, W)
            t: Timestep (B,)
            cond_embed: Conditional text embedding
            cond_proj: Projected conditional embedding
            uncond_embed: Unconditional (zero) embedding
            uncond_proj: Projected unconditional embedding
            size: Size category
            guidance_scale: Guidance strength
        
        Returns:
            Next state (B, C, D, H, W)
        """
        # Conditional prediction (with text)
        t_embed = self.time_embed(t)
        cond_context = torch.cat([t_embed, cond_proj], dim=1)
        logits_cond = self.unets[size](x, cond_context)
        
        # Unconditional prediction (without text)
        uncond_context = torch.cat([t_embed, uncond_proj], dim=1)
        logits_uncond = self.unets[size](x, uncond_context)
        
        # Classifier-Free Guidance formula
        logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        
        # Get posterior distribution q(x_{t-1} | x_t, x_0)
        x_0_pred = probs  # Predicted x_0 distribution
        
        # Sample from posterior
        return self._posterior_sample(x, x_0_pred, t)
    
    def _posterior_sample(self, x_t: torch.Tensor, x_0_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Helper to sample from posterior (reuse existing logic)"""
        # Get alpha values
        alpha_t = self.alpha[t][:, None, None, None, None]
        alpha_t_minus_1 = torch.where(
            t[:, None, None, None, None] > 0,
            self.alpha[t - 1][:, None, None, None, None],
            torch.ones_like(alpha_t)
        )
        
        # Posterior mean
        coef1 = alpha_t_minus_1 * (1 - alpha_t) / (1 - alpha_t * alpha_t_minus_1)
        coef2 = alpha_t * (1 - alpha_t_minus_1) / (1 - alpha_t * alpha_t_minus_1)
        
        posterior_mean = coef1 * x_0_pred + coef2 * x_t
        
        # Clamp to valid probability simplex
        posterior_mean = torch.clamp(posterior_mean, min=1e-8)
        posterior_mean = posterior_mean / posterior_mean.sum(dim=1, keepdim=True)
        
        return posterior_mean
