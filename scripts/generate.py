"""
Generate new Minecraft assets from text prompts
"""

import argparse
import yaml
from pathlib import Path
import logging
import sys
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae_3d import ConditionalVAE3D
from src.data.schematic_parser import SchematicParser
from src.training.trainer import VAELightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_size_from_prompt(prompt: str) -> str:
    """
    Extract size category from prompt
    
    Args:
        prompt: Text prompt
        
    Returns:
        Size category: 'normal', 'big', or 'huge'
    """
    prompt_lower = prompt.lower()
    
    if 'huge' in prompt_lower:
        return 'huge'
    elif 'big' in prompt_lower or 'large' in prompt_lower:
        return 'big'
    else:
        return 'normal'


def generate_from_prompt(
    model: ConditionalVAE3D,
    text_encoder: SentenceTransformer,
    prompt: str,
    num_samples: int = 1,
    temperature: float = 1.0,
    device: str = 'cuda',
    size_override: str = None
) -> tuple:
    """
    Generate voxel arrays from text prompt
    
    Args:
        model: Trained VAE model
        text_encoder: Sentence transformer for text encoding
        prompt: Text description
        num_samples: Number of variations to generate
        temperature: Sampling temperature (higher = more variation)
        device: Device to run on
        
    Returns:
        (voxel_arrays, size_category)
    """
    model.eval()
    model.to(device)
    
    # Encode text
    text_embedding = text_encoder.encode([prompt], convert_to_tensor=True)
    text_embedding = text_embedding.to(device)
    
    # Determine size
    size_category = size_override if size_override else parse_size_from_prompt(prompt)
    
    logger.info(f"Generating {num_samples} sample(s) for prompt: '{prompt}'")
    logger.info(f"Size category: {size_category}")
    logger.info(f"Temperature: {temperature}")
    
    # Generate
    with torch.no_grad():
        generated_logits = model.generate(
            text_embedding,
            size_category,
            num_samples=num_samples,
            temperature=temperature
        )
        
        # Convert logits to block IDs
        generated_voxels = torch.argmax(generated_logits, dim=1)
    
    # Convert to numpy
    voxel_arrays = generated_voxels.cpu().numpy()
    
    return voxel_arrays, size_category


def main():
    parser = argparse.ArgumentParser(description='Generate Minecraft assets from text prompts')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt describing the asset')
    parser.add_argument('--output', type=str, default='generated.schem',
                       help='Output .schem file path')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of variations to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (0.5-2.0)')
    parser.add_argument('--size', type=str, default=None,
                       choices=['normal', 'big', 'huge'],
                       help='Override size detection (extracted from prompt if not provided)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    module = VAELightningModule.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=args.device
    )
    model = module.model
    
    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoder_name = config['model']['text_encoder']['model_name']
    text_encoder = SentenceTransformer(text_encoder_name)
    
    # Generate
    voxel_arrays, size_category = generate_from_prompt(
        model,
        text_encoder,
        args.prompt,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
        size_override=args.size
    )
    
    # Create schematic parser for saving
    schematic_parser = SchematicParser(config['blocks'])
    
    # Save generated schematics
    output_path = Path(args.output)
    
    if args.num_samples == 1:
        # Save single file
        logger.info(f"Saving schematic to {output_path}")
        schematic_parser.create_schematic(voxel_arrays[0], output_path)
        logger.info("✓ Generation complete!")
    else:
        # Save multiple files
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem
        suffix = output_path.suffix
        
        for i, voxel_array in enumerate(voxel_arrays):
            file_path = output_path.parent / f"{stem}_{i:03d}{suffix}"
            logger.info(f"Saving schematic {i+1}/{args.num_samples} to {file_path}")
            schematic_parser.create_schematic(voxel_array, file_path)
        
        logger.info(f"\n✓ Generated {args.num_samples} variations!")
    
    # Print statistics
    for i, voxel_array in enumerate(voxel_arrays):
        non_air_blocks = np.sum(voxel_array != 0)
        unique_blocks = len(np.unique(voxel_array))
        logger.info(f"Sample {i+1} statistics:")
        logger.info(f"  - Dimensions: {voxel_array.shape}")
        logger.info(f"  - Non-air blocks: {non_air_blocks}")
        logger.info(f"  - Unique block types: {unique_blocks}")


def interactive_mode():
    """Interactive generation mode"""
    import yaml
    
    # Load config
    config_path = 'config/config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Get checkpoint path
    checkpoint_path = input("Enter checkpoint path: ").strip()
    
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    logger.info("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    module = VAELightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=device
    )
    model = module.model
    
    # Load text encoder
    text_encoder_name = config['model']['text_encoder']['model_name']
    text_encoder = SentenceTransformer(text_encoder_name)
    
    logger.info("✓ Model loaded!")
    logger.info("\nInteractive generation mode. Type 'quit' to exit.\n")
    
    while True:
        prompt = input("Enter prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        # Generate
        try:
            voxel_arrays, size_category = generate_from_prompt(
                model,
                text_encoder,
                prompt,
                num_samples=1,
                temperature=1.0,
                device=device
            )
            
            # Save
            output_path = Path('generated') / f"{prompt.replace(' ', '_')[:50]}.schem"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            schematic_parser = SchematicParser(config['blocks'])
            schematic_parser.create_schematic(voxel_arrays[0], output_path)
            
            logger.info(f"✓ Saved to {output_path}\n")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1 or '--interactive' in sys.argv:
        interactive_mode()
    else:
        main()
