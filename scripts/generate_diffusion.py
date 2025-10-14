"""
Generate Minecraft assets using trained Diffusion Model
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import logging
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer_diffusion import DiffusionLightningModule
from src.data.schematic_parser import create_schematic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate Minecraft assets with Diffusion Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text description (e.g., "oak tree", "birch tree big")')
    parser.add_argument('--size', type=str, default=None,
                       help='Size override: normal, big, or huge (auto-detected from prompt if not specified)')
    parser.add_argument('--output', type=str, default='generated',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--sampling-steps', type=int, default=50,
                       help='Number of denoising steps (50=fast, 1000=slow but better quality)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Determine size from prompt if not specified
    if args.size is None:
        prompt_lower = args.prompt.lower()
        if 'huge' in prompt_lower or 'large' in prompt_lower or 'giant' in prompt_lower:
            size = 'huge'
        elif 'big' in prompt_lower or 'tall' in prompt_lower:
            size = 'big'
        else:
            size = 'normal'
    else:
        size = args.size
    
    logger.info(f"Size: {size}")
    dims = config['model']['sizes'][size]['dims']
    logger.info(f"Dimensions: {dims}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionLightningModule.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=device
    )
    model.eval()
    model.to(device)
    
    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoder = SentenceTransformer(config['model']['text_encoder']['model_name'])
    text_encoder.to(device)
    
    # Encode prompt
    logger.info(f"Encoding prompt: '{args.prompt}'")
    text_embed = text_encoder.encode(
        [args.prompt] * args.num_samples,
        convert_to_tensor=True,
        device=device
    )
    
    # Generate
    logger.info(f"Generating {args.num_samples} sample(s)...")
    logger.info(f"Sampling steps: {args.sampling_steps} (this will take ~{args.sampling_steps * 0.1:.1f}s)")
    
    with torch.no_grad():
        # Generate using diffusion model
        generated_onehot = model.model.generate(
            text_embed=text_embed,
            size=size,
            num_samples=args.num_samples,
            sampling_steps=args.sampling_steps
        )
        
        # Convert from one-hot to class indices
        generated_voxels = torch.argmax(generated_onehot, dim=1)  # (B, D, H, W)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get block vocabulary
    block_id_to_name = {v: k for k, v in config['blocks'].items()}
    
    # Save generated assets
    for i in range(args.num_samples):
        voxels = generated_voxels[i].cpu().numpy()  # (D, H, W)
        
        # Count non-air blocks
        non_air_blocks = (voxels != 0).sum()
        logger.info(f"Sample {i+1}: {non_air_blocks} non-air blocks")
        
        # Create schematic file
        filename = f"{args.prompt.replace(' ', '_')}_{size}_{i+1}.schem"
        output_path = output_dir / filename
        
        create_schematic(
            voxel_data=voxels,
            block_id_to_name=block_id_to_name,
            output_path=str(output_path)
        )
        
        logger.info(f"✓ Saved: {output_path}")
    
    logger.info(f"\n✓ Generated {args.num_samples} sample(s) successfully!")
    logger.info(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
