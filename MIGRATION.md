# Migration to Diffusion Model

## What Changed?

The project has been upgraded from **VAE (Variational Autoencoder)** to **Diffusion Model (DDPM)** for much better 3D structure generation.

### Why Diffusion?

**VAE Problems:**
- Generated chaotic "balls of logs" without clear tree structure
- Poor at learning spatial relationships between blocks
- Loss stuck around 0.28-0.29 with no improvement

**Diffusion Advantages:**
- ✅ Better at discrete 3D structures (Minecraft blocks)
- ✅ Learns hierarchical patterns (trunk → branches → leaves)
- ✅ State-of-the-art for generation tasks (like Stable Diffusion)
- ✅ More stable training

## What Was Reused?

✅ **100% of data pipeline:**
- `src/data/dataset.py` - No changes
- `src/data/schematic_parser.py` - No changes
- `data/` directory - No changes needed
- Block vocabulary (all 45 blocks) - Same IDs

✅ **Configuration structure:**
- Same `config/config.yaml` format
- Same size categories (normal, big, huge)
- Same text encoder (Sentence-BERT)

## New Files

1. **`src/models/diffusion_3d.py`** - 3D Diffusion Model architecture
   - `UNet3D` - U-Net with attention and FiLM conditioning
   - `DiffusionModel3D` - Complete diffusion pipeline (noise schedule, sampling)
   - Cosine noise schedule (better than linear)
   - DDIM sampling for fast generation (50 steps instead of 1000)

2. **`src/training/trainer_diffusion.py`** - Training module
   - Simple MSE loss (predict noise)
   - No class weights needed (diffusion handles imbalance naturally)
   - Gradient clipping for stability

3. **`scripts/train_diffusion.py`** - Training script
4. **`scripts/generate_diffusion.py`** - Generation script

## Configuration Changes

```yaml
# OLD (VAE)
vae:
  beta: 0.01
  kl_annealing: true

# NEW (Diffusion)
diffusion:
  num_timesteps: 1000
  noise_schedule: "cosine"
  sampling_steps: 50  # Fast generation
  attention_levels: [2, 3]  # Self-attention for spatial relationships
```

**Training settings adjusted:**
- Batch size: 24 → 16 (diffusion uses more memory)
- Learning rate: 0.002 → 0.0001 (diffusion needs stability)
- Epochs: 300 → 200 (converges faster)

## How to Train

```bash
# On RunPod
cd "model N"
git pull

# Remove old VAE models
rm -rf models && mkdir models

# Train diffusion model
python3 scripts/train_diffusion.py
```

## How to Generate

```bash
# Generate with trained diffusion model
python3 scripts/generate_diffusion.py \
  --checkpoint models/minecraft-diffusion-epoch=XX-val/loss=X.XXXX.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 3 \
  --sampling-steps 50
```

**Sampling steps:**
- `--sampling-steps 50` = Fast (~5 seconds), good quality
- `--sampling-steps 100` = Medium (~10 seconds), better quality
- `--sampling-steps 1000` = Slow (~100 seconds), best quality

## Expected Results

**Training:**
- Loss should decrease smoothly (no plateau like VAE)
- Converges to ~0.05-0.10 (much better than VAE's 0.28)
- Training time: ~10-15 minutes for 50 epochs on RTX A6000

**Generation:**
- Clear tree trunk (vertical logs)
- Proper crown structure (leaves around top)
- No more "chaotic balls of logs"
- Recognizable tree shapes

## Architecture Details

**3D UNet:**
- Encoder: 4 levels with downsampling (128 → 256 → 512 → 1024 channels)
- Bottleneck: Residual blocks + Self-attention
- Decoder: 4 levels with upsampling + skip connections
- Time embedding: Sinusoidal positional encoding
- Text conditioning: FiLM (Feature-wise Linear Modulation)
- Attention: Self-attention at deeper levels (2, 3) for global structure

**Total parameters:** ~85M (larger than VAE's 72M)

## Technical Notes

1. **One-hot encoding:** Diffusion works on continuous space, so blocks are one-hot encoded during training
2. **Noise schedule:** Cosine schedule is better than linear for 3D data
3. **FiLM conditioning:** Both time and text embeddings modulate the network via scale/shift
4. **Attention:** Captures long-range dependencies (e.g., "leaves should be near trunk")
5. **Gradient clipping:** Prevents instability during training

## Troubleshooting

**If training is unstable (loss spikes/NaN):**
- Reduce learning rate to 0.00005
- Increase gradient clipping to 2.0
- Check data for NaN values

**If generation is too slow:**
- Use `--sampling-steps 25` for even faster generation
- Trade-off: slightly lower quality

**If structures still look bad after training:**
- Train for more epochs (200+)
- Increase model capacity (channels to [256, 512, 1024, 2048])
- Add more attention levels

## Migration Checklist

- [x] Create diffusion model architecture
- [x] Create diffusion training module
- [x] Update config for diffusion parameters
- [x] Create new training script
- [x] Create new generation script
- [ ] Train diffusion model on RunPod
- [ ] Test generation quality
- [ ] Compare with VAE results

## Old Files (Can be deleted later)

- `src/models/vae_3d.py` - Old VAE model
- `src/training/trainer.py` - Old VAE training
- `src/training/losses.py` - Old VAE loss
- `scripts/train.py` - Old VAE training script
- `scripts/generate.py` - Old VAE generation script

**Don't delete yet** - keep for comparison until diffusion is proven better.
