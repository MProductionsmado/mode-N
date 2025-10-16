# Minecraft-Spezifische Optimierungen für Discrete Diffusion 3D UNet

## Problem mit Standard-UNet

**Standard 3D UNet:**
- Behandelt alle Blöcke gleich
- Keine räumliche Kohärenz-Garantie
- Lernt lokale Muster, aber nicht globale Struktur
- Floating blocks, disconnected parts

**Minecraft-spezifische Herausforderungen:**
1. **Physik**: Blöcke sollten verbunden sein (keine schwebenden Teile)
2. **Struktur**: Bäume haben klare Hierarchie (Stamm unten → Krone oben)
3. **Materialien**: Manche Blöcke kommen nur zusammen vor (leaves + wood)
4. **Symmetrie**: Natürliche Objekte sind oft symmetrisch

## Optimierungen

### 1. **Block-Aware Attention** ⭐⭐⭐ (BESTE LÖSUNG)

Statt generischer Self-Attention: **Minecraft-Block-Bewusste Attention**

**Idee:**
- Air-Blöcke ignorieren bei Attention (sind nicht informativ)
- Solid blocks bekommen mehr Gewicht
- Gleiche Blocktypen "sprechen" mehr miteinander

```python
class MinecraftAwareAttention3D(nn.Module):
    """
    Attention that understands Minecraft block semantics
    - Ignores air blocks
    - Groups similar materials (all wood types together)
    - Enforces vertical coherence for trees
    """
    
    def __init__(self, channels: int, num_classes: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        
        # Block type embeddings (learn similarity between block types)
        self.block_type_embed = nn.Embedding(num_classes, channels)
        
        # Block category masks (hardcoded knowledge)
        self.register_buffer('is_air', torch.zeros(num_classes))
        self.register_buffer('is_solid', torch.zeros(num_classes))
        self.register_buffer('is_leaves', torch.zeros(num_classes))
        self.register_buffer('is_wood', torch.zeros(num_classes))
    
    def set_block_categories(self, block_id_to_name: Dict[int, str]):
        """Initialize block category masks from config"""
        for block_id, name in block_id_to_name.items():
            if 'air' in name.lower():
                self.is_air[block_id] = 1.0
            elif 'leaves' in name.lower() or 'leaf' in name.lower():
                self.is_leaves[block_id] = 1.0
            elif 'log' in name.lower() or 'wood' in name.lower():
                self.is_wood[block_id] = 1.0
            else:
                self.is_solid[block_id] = 1.0
    
    def forward(self, x: torch.Tensor, block_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, C, D, H, W)
            block_types: Block type indices (B, D, H, W) - argmax of current prediction
        """
        B, C, D, H, W = x.shape
        
        # Standard QKV
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Compute attention
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / math.sqrt(C // self.num_heads)
        
        # Create Minecraft-aware mask
        # Reshape block_types for masking
        block_flat = block_types.reshape(B, -1)  # (B, D*H*W)
        
        # Air blocks should not receive attention
        air_mask = self.is_air[block_flat]  # (B, D*H*W)
        air_mask = air_mask[:, None, None, :].expand(-1, self.num_heads, D*H*W, -1)
        attn = attn.masked_fill(air_mask.bool(), float('-inf'))
        
        # Boost attention between same material types
        # Wood attends more to wood, leaves to leaves
        wood_mask = self.is_wood[block_flat]
        leaves_mask = self.is_leaves[block_flat]
        
        # Same-type boost: (B, N) outer product -> (B, N, N)
        wood_boost = torch.einsum('bn,bm->bnm', wood_mask, wood_mask)
        leaves_boost = torch.einsum('bn,bm->bnm', leaves_mask, leaves_mask)
        
        boost = (wood_boost + leaves_boost)[:, None, :, :].expand(-1, self.num_heads, -1, -1)
        attn = attn + boost * 2.0  # Boost same-type attention
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, D, H, W)
        
        return self.proj(out) + x  # Residual


class MinecraftUNet3D(nn.Module):
    """UNet with Minecraft-aware attention"""
    
    def __init__(self, config: Dict):
        super().__init__()
        # ... standard UNet code ...
        
        # Replace standard attention with Minecraft-aware attention
        self.attention_blocks = nn.ModuleList([
            MinecraftAwareAttention3D(
                channels=model_channels * mult,
                num_classes=out_channels,
                num_heads=8
            )
            for level, mult in enumerate(channel_multipliers)
            if level in attention_levels
        ])
        
        # Initialize block categories
        block_id_to_name = {v: k for k, v in config['blocks'].items()}
        for attn in self.attention_blocks:
            attn.set_block_categories(block_id_to_name)
```

**Vorteile:**
- ✅ Air-Blöcke werden ignoriert (mehr Signal, weniger Noise)
- ✅ Holz-Blöcke "sprechen" mehr miteinander → bessere Stamm-Kohärenz
- ✅ Blätter "sprechen" mehr miteinander → bessere Kronen-Kohärenz
- ✅ Nutzt Minecraft-Domain-Knowledge

---

### 2. **Hierarchical Generation** ⭐⭐⭐

Statt alles auf einmal: **2-stufige Generation**

**Stufe 1: Grobe Struktur (Coarse)**
- Generiere 8×8×8 low-res Version
- Lernt globale Form (Stamm-Position, Kronen-Form)

**Stufe 2: Details (Fine)**
- Upsample zu 16×16×16
- Füge Details hinzu (einzelne Blätter, Stamm-Details)

```python
class HierarchicalDiscreteDiffusion(nn.Module):
    """
    Two-stage generation:
    1. Coarse: 8×8×8 → Learn global structure
    2. Fine: 16×16×16 → Add details conditioned on coarse
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Coarse model: 8×8×8
        self.coarse_unet = UNet3D(
            in_channels=config['num_classes'],
            out_channels=config['num_classes'],
            model_channels=64,
            # ... smaller model ...
        )
        
        # Fine model: 16×16×16, conditioned on coarse
        self.fine_unet = UNet3D(
            in_channels=config['num_classes'] * 2,  # Current + upsampled coarse
            out_channels=config['num_classes'],
            model_channels=128,
            # ... larger model ...
        )
    
    def generate_coarse(self, text_embed, size=(8,8,8)):
        """Generate coarse 8×8×8 structure"""
        # Standard discrete diffusion on small grid
        x = torch.ones(1, self.num_classes, *size) / self.num_classes
        
        for t in reversed(range(self.num_timesteps)):
            x = self.coarse_unet(x, t, text_embed)
        
        return x
    
    def generate_fine(self, text_embed, coarse, size=(16,16,16)):
        """Generate fine 16×16×16 details conditioned on coarse"""
        # Upsample coarse to 16×16×16
        coarse_up = F.interpolate(coarse, size=size, mode='nearest')
        
        # Start from uniform + coarse hint
        x = torch.ones(1, self.num_classes, *size) / self.num_classes
        
        for t in reversed(range(self.num_timesteps)):
            # Concatenate current state with upsampled coarse
            x_cond = torch.cat([x, coarse_up], dim=1)
            x = self.fine_unet(x_cond, t, text_embed)
        
        return x
    
    def generate(self, text_embed, size=(16,16,16)):
        """Two-stage generation"""
        # Stage 1: Coarse structure
        coarse = self.generate_coarse(text_embed, size=(8,8,8))
        
        # Stage 2: Fine details
        fine = self.generate_fine(text_embed, coarse, size=size)
        
        return fine
```

**Vorteile:**
- ✅ Coarse-Modell lernt globale Struktur (Stamm UNTEN, Krone OBEN)
- ✅ Fine-Modell fügt Details hinzu ohne Struktur zu zerstören
- ✅ Ähnlich wie Stable Diffusion Latent-Space
- ✅ Separiert globale vs lokale Muster

---

### 3. **Physics-Informed Loss** ⭐⭐

Zusätzliche Loss-Terms die Minecraft-Physik erzwingen

```python
class MinecraftPhysicsLoss(nn.Module):
    """Additional loss terms for Minecraft physics"""
    
    def __init__(self, num_classes: int, block_config: Dict):
        super().__init__()
        self.num_classes = num_classes
        
        # Create masks for block categories
        self.air_id = block_config['air']
        self.wood_ids = [v for k, v in block_config.items() if 'log' in k or 'wood' in k]
        self.leaves_ids = [v for k, v in block_config.items() if 'leaves' in k or 'leaf' in k]
    
    def gravity_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Penalize floating blocks (no support below)
        """
        B, C, D, H, W = logits.shape
        
        # Get predicted block types
        pred_blocks = torch.argmax(logits, dim=1)  # (B, D, H, W)
        
        # For each block, check if there's support below
        # (ignore bottom layer and air blocks)
        loss = 0.0
        
        for y in range(1, H):  # Start from second layer
            current_layer = pred_blocks[:, :, y, :]  # (B, D, W)
            below_layer = pred_blocks[:, :, y-1, :]  # (B, D, W)
            
            # Block is floating if:
            # - Current block is NOT air
            # - Block below IS air
            is_solid = (current_layer != self.air_id).float()
            is_air_below = (below_layer == self.air_id).float()
            
            floating = is_solid * is_air_below  # (B, D, W)
            loss += floating.sum()
        
        return loss / (B * D * H * W)
    
    def connectivity_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Penalize isolated blocks (no neighbors of same type)
        """
        B, C, D, H, W = logits.shape
        pred_blocks = torch.argmax(logits, dim=1)
        
        loss = 0.0
        
        # For each non-air block, check 6-connectivity
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            # Shift prediction
            if dx != 0:
                neighbor = torch.roll(pred_blocks, shifts=dx, dims=1)
            elif dy != 0:
                neighbor = torch.roll(pred_blocks, shifts=dy, dims=2)
            else:
                neighbor = torch.roll(pred_blocks, shifts=dz, dims=3)
            
            # Same type = connected
            same_type = (pred_blocks == neighbor).float()
            loss += (1.0 - same_type).mean()
        
        return loss / 6.0
    
    def tree_structure_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Enforce tree structure:
        - Wood should be mostly vertical
        - Leaves should be above wood
        """
        B, C, D, H, W = logits.shape
        pred_blocks = torch.argmax(logits, dim=1)
        
        loss = 0.0
        
        # 1. Wood should be continuous vertically
        wood_mask = torch.zeros_like(pred_blocks, dtype=torch.float32)
        for wood_id in self.wood_ids:
            wood_mask += (pred_blocks == wood_id).float()
        
        # Check vertical continuity
        for y in range(H - 1):
            wood_here = wood_mask[:, :, y, :]
            wood_above = wood_mask[:, :, y+1, :]
            
            # If wood here but not above → penalize
            discontinuity = wood_here * (1.0 - wood_above)
            loss += discontinuity.sum()
        
        # 2. Leaves should be in upper half
        leaves_mask = torch.zeros_like(pred_blocks, dtype=torch.float32)
        for leaves_id in self.leaves_ids:
            leaves_mask += (pred_blocks == leaves_id).float()
        
        # Penalize leaves in lower half
        lower_half = leaves_mask[:, :, :H//2, :]
        loss += lower_half.sum() * 2.0
        
        return loss / (B * D * H * W)
    
    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all physics losses"""
        return {
            'gravity': self.gravity_loss(logits),
            'connectivity': self.connectivity_loss(logits),
            'tree_structure': self.tree_structure_loss(logits)
        }


# In training loop:
physics_loss_fn = MinecraftPhysicsLoss(num_classes, config['blocks'])

# Combined loss
ce_loss = F.cross_entropy(logits, target)
physics_losses = physics_loss_fn(logits)

total_loss = (
    ce_loss + 
    0.1 * physics_losses['gravity'] +
    0.05 * physics_losses['connectivity'] +
    0.1 * physics_losses['tree_structure']
)
```

**Vorteile:**
- ✅ Erzwingt Minecraft-Physik (keine schwebenden Blöcke)
- ✅ Erzwingt Konnektivität (verbundene Strukturen)
- ✅ Erzwingt Baum-Struktur (vertikal, Krone oben)
- ⚠️ Kann Training verlangsamen

---

### 4. **Anisotropic Convolutions** ⭐⭐

Bäume sind **nicht isotrop**: Y-Achse (Höhe) ist wichtiger als X/Z

```python
class AnisotropicConv3D(nn.Module):
    """
    Different kernel sizes for different axes
    Taller kernel in Y (height) direction for trees
    """
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Vertical kernel: 1×5×1 (tall)
        self.vertical_conv = nn.Conv3d(in_ch, out_ch//2, kernel_size=(1, 5, 1), padding=(0, 2, 0))
        
        # Horizontal kernel: 3×1×3 (wide)
        self.horizontal_conv = nn.Conv3d(in_ch, out_ch//2, kernel_size=(3, 1, 3), padding=(1, 0, 1))
    
    def forward(self, x):
        v = self.vertical_conv(x)
        h = self.horizontal_conv(x)
        return torch.cat([v, h], dim=1)

# Replace standard Conv3d with AnisotropicConv3D in UNet
```

**Vorteile:**
- ✅ Besser für vertikale Strukturen (Baumstämme)
- ✅ Weniger Parameter als 3×3×3
- ✅ Schneller

---

### 5. **Classifier-Free Guidance** ⭐⭐⭐

Wie Stable Diffusion: Verstärke Text-Conditioning

```python
def generate_with_cfg(self, text_embed, guidance_scale=7.5):
    """
    Classifier-Free Guidance:
    pred = unconditional + guidance_scale * (conditional - unconditional)
    """
    # Unconditional prediction (empty text)
    empty_embed = torch.zeros_like(text_embed)
    
    x = torch.ones(1, self.num_classes, *size) / self.num_classes
    
    for t in reversed(range(self.num_timesteps)):
        # Conditional prediction
        logits_cond = self.unet(x, t, text_embed)
        
        # Unconditional prediction
        logits_uncond = self.unet(x, t, empty_embed)
        
        # Guided prediction
        logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        
        # Sample
        x = self.p_sample(x, t, logits)
    
    return x
```

**Braucht Training-Änderung:**
```python
# During training: randomly drop text conditioning
if random.random() < 0.1:  # 10% unconditional
    text_embed = torch.zeros_like(text_embed)

logits = model(x_noisy, t, text_embed)
```

**Vorteile:**
- ✅ Viel stärkere Text-Kontrolle
- ✅ Bessere Qualität (wie Stable Diffusion)
- ✅ Einfach zu implementieren

---

## Empfohlene Kombination

**Quick Wins (sofort umsetzbar):**
1. ✅ **Classifier-Free Guidance** (nur Training-Script ändern)
2. ✅ **Slower Learning Rate** (bereits implementiert)
3. ✅ **Mehr Sampling Steps** (bereits möglich)

**Mittlerer Aufwand:**
4. ✅ **Block-Aware Attention** (UNet anpassen)
5. ✅ **Anisotropic Convolutions** (UNet anpassen)

**Großer Aufwand:**
6. ✅ **Hierarchical Generation** (neues Modell)
7. ✅ **Physics Loss** (neuer Loss-Term)

---

## Nächste Schritte

**Option A: Schnell testen (1-2 Tage)**
```bash
# 1. Classifier-Free Guidance implementieren
# 2. Mit guidance_scale=7.5 generieren
# 3. Vergleichen mit bisherigen Ergebnissen
```

**Option B: Mittelfristig (1 Woche)**
```bash
# 1. Block-Aware Attention implementieren
# 2. Neu trainieren mit slow learning
# 3. CFG + Block-Aware Attention kombinieren
```

**Option C: Langfristig (2-3 Wochen)**
```bash
# 1. Hierarchical Generation implementieren
# 2. Coarse (8³) + Fine (16³) Models trainieren
# 3. CFG + Hierarchical + Block-Aware kombinieren
```

Welche Richtung interessiert dich am meisten? 🚀
