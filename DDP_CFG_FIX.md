# DDP + CFG Fix

## Problem
When training with multi-GPU (DDP) and Classifier-Free Guidance (CFG), PyTorch Lightning throws an error:

```
RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
```

## Root Cause
- **CFG** randomly drops text conditioning 10% of the time (conditioning_dropout=0.1)
- When text embeddings are set to zeros, the text encoder parameters don't contribute to gradients
- **DDP** (Distributed Data Parallel) expects all parameters to be used in every forward pass
- This causes a parameter synchronization conflict across GPUs

## Solution
Changed the DDP strategy from `'ddp'` to `'ddp_find_unused_parameters_true'`:

```python
# Before (in src/training/trainer_discrete_diffusion.py)
strategy = 'ddp' if num_devices > 1 else 'auto'

# After
strategy = 'ddp_find_unused_parameters_true' if num_devices > 1 else 'auto'
```

## How It Works
- `ddp_find_unused_parameters_true` enables PyTorch's `find_unused_parameters=True` in DDP
- This tells DDP to detect which parameters are unused in each forward pass
- Unused parameters are excluded from gradient synchronization for that step
- Allows CFG to work correctly with multi-GPU training

## Performance Impact
- Minimal overhead: DDP does one extra graph traversal to detect unused parameters
- Worth it for CFG functionality across multiple GPUs
- Alternative would be to disable CFG or train on single GPU (both worse options)

## Technical Details
- **File Modified**: `src/training/trainer_discrete_diffusion.py` (line ~174)
- **Lightning Version**: Compatible with PyTorch Lightning 2.x
- **Multi-GPU Setup**: 4× RTX 4090, DDP strategy, batch_size=4 per GPU
- **CFG Parameters**: conditioning_dropout=0.1, guidance_scale=3.0

## Training Impact
- ✅ No change to model architecture
- ✅ No change to training loss or convergence
- ✅ Allows CFG + multi-GPU to work together
- ✅ All 4 GPUs can be used with conditional dropout enabled

## References
- PyTorch DDP Documentation: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch Lightning DDP Strategy: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
- Classifier-Free Guidance Paper: Ho & Salimans (2022)

---
**Status**: Fixed and tested ✅  
**Date**: 2025-01-14  
**Training Ready**: Yes - Training can now proceed with 4 GPUs + CFG
