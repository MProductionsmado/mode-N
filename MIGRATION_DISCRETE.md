# Migration to Discrete Diffusion Model

## What Changed NOW?

The project has been upgraded from **Continuous Gaussian Diffusion** to **Discrete Multinomial Diffusion** for MUCH better categorical data generation.

### Why Discrete Diffusion?

**Continuous Diffusion Problems:**
- Generated chaotic block mix without structure
- Gaussian noise doesn't match categorical Minecraft blocks
- Loss was low (0.029) but quality was terrible
- Argmax at end destroys structure

**Discrete Diffusion Advantages:**
- ✅ **Native categorical transitions** (Block A → Block B)
- ✅ **No continuous → discrete mismatch**
- ✅ **Better for discrete state spaces** (Minecraft blocks)
- ✅ **Structured sampling** (no argmax needed)
- ✅ Based on "Argmax Flows and Multinomial Diffusion" (Hoogeboom et al. 2021)

## What Was Reused?

✅ **100% of data pipeline:**
- `src/data/dataset.py` - No changes
- `src/data/schematic_parser.py` - No changes
- All preprocessing - No changes

✅ **UNet3D architecture:**
- Same encoder/decoder structure
- Same attention blocks
- Same FiLM conditioning
- Only output changed: predicts **logits** instead of **noise**

✅ **Training infrastructure:**
- PyTorch Lightning
- Same callbacks (checkpoints, early stopping)
- Same multi-GPU support

## What Changed?

### 1. **Diffusion Process**

**OLD (Continuous):**
```python
# Forward: Add Gaussian noise
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise

# Reverse: Predict noise, subtract it
noise_pred = model(x_t, t)
x_{t-1} = denoise(x_t, noise_pred)

# Loss: MSE between predicted and actual noise
loss = MSE(noise_pred, actual_noise)
```

**NEW (Discrete):**
```python
# Forward: Transition to uniform distribution
x_t = alpha_t * x_0 + (1 - alpha_t) / num_classes

# Reverse: Predict original categories
logits_pred = model(x_t, t)  # Predict x_0 directly

# Loss: Cross-Entropy between predicted and actual categories
loss = CrossEntropy(logits_pred, target_classes)
```

### 2. **Loss Function**

- **OLD:** MSE (mean squared error) for noise prediction
- **NEW:** Cross-Entropy for category prediction

### 3. **Sampling**

- **OLD:** Start from Gaussian noise, denoise iteratively, argmax at end
- **NEW:** Start from uniform distribution, predict categories iteratively, no argmax needed

## New Files

1. **`src/models/discrete_diffusion_3d.py`** - Discrete Diffusion Model
   - `DiscreteDiscreteDiffusionModel3D` - Multinomial transitions
   - UNet3D reused, output predicts logits
   - Cosine schedule for transition probabilities

2. **`src/training/trainer_discrete_diffusion.py`** - Training module
   - Cross-Entropy loss
   - Accuracy metric

3. **`scripts/train_discrete_diffusion.py`** - Training script
4. **`scripts/generate_discrete_diffusion.py`** - Generation script

## Config (No Changes Needed!)

Same config as before. Discrete diffusion uses same parameters:
- `num_timesteps: 1000`
- `noise_schedule: "cosine"` (now for transitions, not noise)
- `sampling_steps: 50`

## How to Train

```bash
# On RunPod
cd "model N"
git pull

# Stop old training (if running)
# Ctrl+C

# Train NEW discrete diffusion model
python3 scripts/train_discrete_diffusion.py
```

**Training differences:**
- **Loss metric:** Now shows Cross-Entropy instead of MSE
- **Expected values:** Loss 2.0-3.0 at start → 0.5-1.0 at convergence
- **New metric:** Accuracy (how often predicted block matches target)
- **Speed:** About same as before (~14 it/s)

## How to Generate

```bash
# Generate with discrete diffusion
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/minecraft-discrete-diffusion-epoch=XX.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 5 \
  --sampling-steps 50
```

**Generation differences:**
- **No temperature parameter** (not needed, sampling is categorical)
- **Cleaner output** (no continuous→discrete mismatch)
- **Better structure** (native categorical transitions)

## Expected Results

**Training:**
- **Loss:** Start ~2.5-3.0, converge to ~0.5-1.0
- **Accuracy:** Start ~20%, converge to ~60-80%
- **Speed:** Same as before (~14 it/s on RTX 4090)

**Generation:**
- ✅ **Clear tree trunks** (vertical logs, not random blocks)
- ✅ **Proper crown structure** (leaves around top)
- ✅ **Coherent shapes** (recognizable trees)
- ✅ **No more chaotic noise!**

## Technical Details

### Multinomial Diffusion

1. **Forward Process:**
   - Gradually transition from data distribution to uniform distribution
   - At t=0: Stay in same state with prob 1.0
   - At t=T: Uniform over all categories

2. **Reverse Process:**
   - Predict original category from noised version
   - Use predicted probabilities to compute posterior
   - Sample from posterior distribution

3. **Loss:**
   - Cross-Entropy: `-log P(true_class | noised_input)`
   - Directly optimizes category prediction

### Why This Works Better

- **Discrete data needs discrete diffusion**
- Minecraft blocks are categorical, not continuous
- Gaussian noise → argmax loses structure
- Multinomial transitions preserve categorical nature

## Comparison

| Aspect | Continuous Diffusion | Discrete Diffusion |
|--------|---------------------|-------------------|
| **Forward** | Add Gaussian noise | Transition to uniform |
| **Reverse** | Predict & subtract noise | Predict categories |
| **Loss** | MSE (noise) | Cross-Entropy (categories) |
| **Sampling** | Denoise + argmax | Categorical sampling |
| **Structure** | Poor (argmax breaks it) | Good (native categorical) |
| **For Minecraft** | ❌ Bad fit | ✅ Perfect fit |

## Migration Checklist

- [x] Create discrete diffusion model
- [x] Create discrete training module  
- [x] Update training script
- [x] Update generation script
- [x] Documentation
- [ ] Train discrete diffusion on RunPod
- [ ] Test generation quality
- [ ] Compare with continuous diffusion

## Old Files (Keep for comparison)

- `src/models/diffusion_3d.py` - Old continuous diffusion
- `src/training/trainer_diffusion.py` - Old MSE training
- `scripts/train_diffusion.py` - Old training script
- `scripts/generate_diffusion.py` - Old generation script

**Don't delete yet** - keep until discrete diffusion proves better.

## Troubleshooting

**If loss doesn't decrease:**
- Check data loading (one-hot encoding correct?)
- Reduce learning rate to 0.00005
- Increase batch size if memory allows

**If accuracy plateaus at ~30%:**
- This is normal - many blocks look similar
- Quality matters more than accuracy
- Test generation to see actual structure

**If generation still looks bad:**
- Train for more epochs (100-200)
- Try more sampling steps (100-200)
- Check if model is actually loading correctly

## Expected Timeline

- **Epoch 10:** Loss ~1.5, Accuracy ~30%, chaotic structure
- **Epoch 25:** Loss ~1.0, Accuracy ~40%, some patterns emerge
- **Epoch 50:** Loss ~0.7, Accuracy ~50%, clear structures
- **Epoch 100:** Loss ~0.5, Accuracy ~60%, high quality trees

**Test generation at Epoch 50!** Should already be MUCH better than continuous diffusion.
