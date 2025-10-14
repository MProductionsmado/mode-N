"""
Analyze why the model doesn't differentiate between prompts
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
print(f"Loading OLD checkpoint (without class weights): {checkpoint_path}")
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

# Test different prompts
prompts = [
    "oak tree",
    "pine tree",
    "birch tree",
    "bush",
    "flower",
    "normal oak tree",
    "big oak tree",
    "huge oak tree"
]

print("\n=== Text Embedding Analysis ===")
embeddings = []
for prompt in prompts:
    emb = text_encoder.encode([prompt], convert_to_tensor=True)
    embeddings.append(emb)
    print(f"{prompt:20s}: mean={emb.mean().item():.3f}, std={emb.std().item():.3f}, norm={emb.norm().item():.3f}")

# Check similarity between embeddings
print("\n=== Embedding Similarities (Cosine) ===")
from torch.nn.functional import cosine_similarity
for i in range(len(prompts)):
    for j in range(i+1, len(prompts)):
        sim = cosine_similarity(embeddings[i], embeddings[j], dim=1).item()
        print(f"{prompts[i]:20s} <-> {prompts[j]:20s}: {sim:.3f}")

# Test generation with different prompts
print("\n=== Generation Test ===")
results = []
for prompt in ["oak tree", "pine tree", "bush"]:
    text_emb = text_encoder.encode([prompt], convert_to_tensor=True).cuda()
    
    with torch.no_grad():
        # Generate 3 samples
        generated_logits = model.generate(text_emb, 'normal', num_samples=3, temperature=1.0)
        generated_voxels = torch.argmax(generated_logits, dim=1)
    
    # Statistics
    unique_blocks = []
    non_air_counts = []
    for i in range(3):
        voxels = generated_voxels[i].cpu().numpy()
        unique = np.unique(voxels)
        non_air = np.sum(voxels != 0)
        unique_blocks.append(unique)
        non_air_counts.append(non_air)
    
    print(f"\n{prompt}:")
    print(f"  Non-air blocks: {non_air_counts}")
    print(f"  Unique blocks: {[list(u) for u in unique_blocks]}")
    print(f"  Variation: {np.std(non_air_counts):.1f} (should be >0)")
    
    results.append({
        'prompt': prompt,
        'non_air': non_air_counts,
        'unique': unique_blocks
    })

# Check if all generations are identical
print("\n=== Checking if generations are identical ===")
for i in range(len(results)):
    prompt = results[i]['prompt']
    non_air = results[i]['non_air']
    if len(set(non_air)) == 1:
        print(f"⚠️  {prompt}: All 3 samples have IDENTICAL block counts ({non_air[0]})!")
    else:
        print(f"✓  {prompt}: Samples vary ({non_air})")

# Test FiLM conditioning
print("\n=== Testing FiLM Conditioning ===")
decoder = model.decoders['normal']
print(f"Decoder has {len(decoder.decoder_blocks)} blocks")
for i, block in enumerate(decoder.decoder_blocks):
    if hasattr(block, 'film'):
        print(f"Block {i}: FiLM layer exists")
        # Check if FiLM weights are meaningful
        film_weight_norm = block.film.scale.weight.norm().item()
        print(f"  FiLM scale weight norm: {film_weight_norm:.3f}")
    else:
        print(f"Block {i}: NO FiLM layer!")

print("\n=== Diagnosis ===")
print("If prompts have high similarity (>0.95): Text encoder is not differentiating")
print("If all samples identical: Temperature/sampling is broken")
print("If no FiLM layers: Conditioning is not connected")
print("If FiLM weights near 0: FiLM is not being used")
