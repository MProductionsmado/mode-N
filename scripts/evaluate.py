"""
Evaluate trained model on test set
"""

import argparse
import yaml
from pathlib import Path
import logging
import sys
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.training.trainer import VAELightningModule
from src.training.losses import AccuracyMetric, NonAirAccuracyMetric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    model.to(device)
    
    accuracy_metric = AccuracyMetric()
    non_air_metric = NonAirAccuracyMetric(air_id=0)
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_accuracy = 0.0
    total_non_air_accuracy = 0.0
    num_batches = 0
    
    size_metrics = {'normal': [], 'big': [], 'huge': []}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            voxels = batch['voxels'].to(device)
            text_emb = batch['text_embedding'].to(device)
            size_name = batch['size_name'][0]
            
            # Forward pass
            recon, mu, logvar = model(voxels, text_emb, size_name)
            
            # Calculate losses
            loss, loss_dict = model.loss_fn(recon, voxels, mu, logvar)
            
            # Calculate metrics
            accuracy = accuracy_metric(recon, voxels)
            non_air_acc = non_air_metric(recon, voxels)
            
            # Accumulate
            total_loss += loss_dict['loss']
            total_recon_loss += loss_dict['recon_loss']
            total_kl_loss += loss_dict['kl_loss']
            total_accuracy += accuracy
            total_non_air_accuracy += non_air_acc
            num_batches += 1
            
            # Track by size
            size_metrics[size_name].append({
                'accuracy': accuracy,
                'non_air_accuracy': non_air_acc
            })
    
    # Average metrics
    results = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'non_air_accuracy': total_non_air_accuracy / num_batches
    }
    
    # Size-specific metrics
    for size_name, metrics in size_metrics.items():
        if metrics:
            results[f'{size_name}_accuracy'] = np.mean([m['accuracy'] for m in metrics])
            results[f'{size_name}_non_air_accuracy'] = np.mean([m['non_air_accuracy'] for m in metrics])
    
    return results


def visualize_samples(model, test_loader, num_samples=5, device='cuda', output_dir='evaluation'):
    """
    Generate and visualize sample reconstructions
    """
    model.eval()
    model.to(device)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get a batch
    batch = next(iter(test_loader))
    voxels = batch['voxels'][:num_samples].to(device)
    text_emb = batch['text_embedding'][:num_samples].to(device)
    prompts = batch['prompt'][:num_samples]
    size_name = batch['size_name'][0]
    
    with torch.no_grad():
        recon, mu, logvar = model(voxels, text_emb, size_name)
        
        # Convert to block IDs
        original_ids = torch.argmax(voxels, dim=1).cpu().numpy()
        recon_ids = torch.argmax(recon, dim=1).cpu().numpy()
    
    # Visualize each sample
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original (top-down view)
        orig_view = np.max(original_ids[i], axis=1)  # Max projection along Y
        axes[0].imshow(orig_view, cmap='tab20', interpolation='nearest')
        axes[0].set_title(f'Original\n{prompts[i]}')
        axes[0].axis('off')
        
        # Reconstruction
        recon_view = np.max(recon_ids[i], axis=1)
        axes[1].imshow(recon_view, cmap='tab20', interpolation='nearest')
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f'sample_{i:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved visualizations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Minecraft 3D VAE')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization samples')
    parser.add_argument('--num-vis-samples', type=int, default=10,
                       help='Number of samples to visualize')
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
    
    # Create test dataloader
    logger.info("Loading test data...")
    data_dir = Path(config['data']['output_dir'])
    
    _, _, test_loader = create_dataloaders(
        config,
        train_metadata=data_dir / 'splits' / 'train_metadata.json',
        val_metadata=data_dir / 'splits' / 'val_metadata.json',
        test_metadata=data_dir / 'splits' / 'test_metadata.json',
        num_workers=config['hardware']['num_workers']
    )
    
    if test_loader is None:
        logger.error("No test data found!")
        return
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = VAELightningModule.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=args.device
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, device=args.device)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    for metric, value in results.items():
        logger.info(f"{metric:30s}: {value:.4f}")
    logger.info("="*50 + "\n")
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_file}")
    
    # Visualize samples
    if args.visualize:
        logger.info("Generating visualizations...")
        visualize_samples(
            model,
            test_loader,
            num_samples=args.num_vis_samples,
            device=args.device,
            output_dir=args.output
        )
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
