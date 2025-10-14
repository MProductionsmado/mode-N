"""
Test generation to debug empty outputs
"""

import yaml
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import VAELightningModule
from sentence_transformers import SentenceTransformer

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Load model
checkpoint_path = 'models/minecraft-vae-epoch=49-val/loss=0.0453.ckpt'
print(f"Loading checkpoint: {checkpoint_path}")
module = VAELightningModule.load_from_checkpoint(
    checkpoint_path,
    config=config,
    map_location='cuda'
)
model = module.model
model.eval()
model.cuda()

# Load text encoder
text_encoder = SentenceTransformer(config['model']['text_encoder']['model_name'])

# Test 1: Generate from prior (current method)
print("\n=== Test 1: Generate from Prior ===")
prompt = "oak tree"
text_emb = text_encoder.encode([prompt], convert_to_tensor=True).cuda()

with torch.no_grad():
    generated_logits = model.generate(text_emb, 'normal', num_samples=1, temperature=1.0)
    generated_voxels = torch.argmax(generated_logits, dim=1)
    
print(f"Shape: {generated_voxels.shape}")
print(f"Unique blocks: {torch.unique(generated_voxels).cpu().numpy()}")
print(f"Non-air blocks: {torch.sum(generated_voxels != 0).item()}")
print(f"Logits stats: min={generated_logits.min().item():.3f}, max={generated_logits.max().item():.3f}, mean={generated_logits.mean().item():.3f}")

# Test 2: Check logits distribution
print("\n=== Test 2: Logits Distribution ===")
for block_id in range(min(5, generated_logits.shape[1])):
    block_logits = generated_logits[0, block_id]
    print(f"Block {block_id}: mean={block_logits.mean().item():.3f}, std={block_logits.std().item():.3f}, max={block_logits.max().item():.3f}")

# Test 3: Sample with different temperatures
print("\n=== Test 3: Different Temperatures ===")
for temp in [0.5, 1.0, 1.5, 2.0]:
    with torch.no_grad():
        generated_logits = model.generate(text_emb, 'normal', num_samples=1, temperature=temp)
        generated_voxels = torch.argmax(generated_logits, dim=1)
    non_air = torch.sum(generated_voxels != 0).item()
    print(f"Temperature {temp}: {non_air} non-air blocks")

# Test 4: Check if model can reconstruct
print("\n=== Test 4: Reconstruction Test ===")
# Load a real sample
from src.data.dataset import MinecraftSchematicDataset
metadata_path = Path('out/metadata.json')
if metadata_path.exists():
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Get first normal-sized sample
    normal_samples = [m for m in metadata if m['size_name'] == 'normal']
    if normal_samples:
        sample = normal_samples[0]
        print(f"Testing reconstruction on: {sample['filename']}")
        
        # Load it
        from src.data.dataset import MinecraftSchematicDataset
        dataset = MinecraftSchematicDataset(
            [sample],
            'out',
            text_encoder_name=config['model']['text_encoder']['model_name'],
            transform=None,
            num_classes=len(config['blocks'])
        )
        
        data = dataset[0]
        voxels = data['voxels'].unsqueeze(0).cuda()
        text_emb = data['text_embedding'].unsqueeze(0).cuda()
        
        # Reconstruct
        with torch.no_grad():
            recon, mu, logvar = model(voxels, text_emb, 'normal')
            recon_voxels = torch.argmax(recon, dim=1)
        
        original_non_air = torch.sum(voxels[0] != 0).item()
        recon_non_air = torch.sum(recon_voxels != 0).item()
        print(f"Original non-air blocks: {original_non_air}")
        print(f"Reconstructed non-air blocks: {recon_non_air}")
        print(f"Reconstruction accuracy: {(recon_voxels[0] == voxels[0, 0]).float().mean().item():.3f}")
        
        # Check latent stats
        print(f"\nLatent stats:")
        print(f"  mu: mean={mu.mean().item():.3f}, std={mu.std().item():.3f}")
        print(f"  logvar: mean={logvar.mean().item():.3f}, std={logvar.std().item():.3f}")
