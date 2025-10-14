"""
Visualizer for 3D voxel data and generated assets
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def visualize_voxel_2d(
    voxel_array: np.ndarray,
    title: str = "Voxel Visualization",
    view: str = "top",
    save_path: Path = None
):
    """
    Create 2D projection visualization of voxel data
    
    Args:
        voxel_array: (X, Y, Z) array of block IDs
        title: Plot title
        view: 'top', 'front', or 'side'
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create projection
    if view == "top":
        projection = np.max(voxel_array, axis=1)  # Max along Y (height)
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
    elif view == "front":
        projection = np.max(voxel_array, axis=2)  # Max along Z
        ax.set_xlabel("X")
        ax.set_ylabel("Y (Height)")
    elif view == "side":
        projection = np.max(voxel_array, axis=0)  # Max along X
        ax.set_xlabel("Z")
        ax.set_ylabel("Y (Height)")
    else:
        raise ValueError(f"Unknown view: {view}")
    
    # Plot
    im = ax.imshow(projection, cmap='tab20', interpolation='nearest', origin='lower')
    ax.set_title(f"{title} - {view.capitalize()} View")
    plt.colorbar(im, ax=ax, label="Block ID")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_voxel_3d(
    voxel_array: np.ndarray,
    title: str = "3D Voxel Visualization",
    save_path: Path = None,
    max_blocks: int = 1000
):
    """
    Create 3D scatter plot of voxel data
    
    Args:
        voxel_array: (X, Y, Z) array of block IDs
        title: Plot title
        save_path: Optional path to save figure
        max_blocks: Maximum number of blocks to plot (for performance)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get non-air blocks
    non_air = np.where(voxel_array != 0)
    x, y, z = non_air
    colors = voxel_array[non_air]
    
    # Subsample if too many blocks
    if len(x) > max_blocks:
        indices = np.random.choice(len(x), max_blocks, replace=False)
        x = x[indices]
        y = y[indices]
        z = z[indices]
        colors = colors[indices]
    
    # Plot
    scatter = ax.scatter(x, y, z, c=colors, cmap='tab20', marker='s', s=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.colorbar(scatter, ax=ax, label="Block ID", shrink=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 3D visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_voxels(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Comparison",
    save_path: Path = None
):
    """
    Compare original and reconstructed voxels side by side
    
    Args:
        original: Original voxel array
        reconstructed: Reconstructed voxel array
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    views = ['top', 'front', 'side']
    
    for i, view in enumerate(views):
        # Original
        if view == "top":
            orig_proj = np.max(original, axis=1)
            recon_proj = np.max(reconstructed, axis=1)
        elif view == "front":
            orig_proj = np.max(original, axis=2)
            recon_proj = np.max(reconstructed, axis=2)
        else:  # side
            orig_proj = np.max(original, axis=0)
            recon_proj = np.max(reconstructed, axis=0)
        
        # Plot original
        axes[0, i].imshow(orig_proj, cmap='tab20', interpolation='nearest', origin='lower')
        axes[0, i].set_title(f'Original - {view.capitalize()}')
        axes[0, i].axis('off')
        
        # Plot reconstruction
        axes[1, i].imshow(recon_proj, cmap='tab20', interpolation='nearest', origin='lower')
        axes[1, i].set_title(f'Reconstructed - {view.capitalize()}')
        axes[1, i].axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    log_file: Path,
    metrics: list = ['loss', 'accuracy'],
    save_path: Path = None
):
    """
    Plot training curves from log file
    
    Args:
        log_file: Path to training log JSON
        metrics: List of metrics to plot
        save_path: Optional path to save figure
    """
    import json
    
    with open(log_file) as f:
        logs = [json.loads(line) for line in f]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        train_values = [log[f'train/{metric}'] for log in logs if f'train/{metric}' in log]
        val_values = [log[f'val/{metric}'] for log in logs if f'val/{metric}' in log]
        
        if train_values:
            axes[i].plot(train_values, label='Train')
        if val_values:
            axes[i].plot(val_values, label='Validation')
        
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} over Training')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test visualization
    test_voxel = np.random.randint(0, 26, (16, 16, 16))
    
    visualize_voxel_2d(test_voxel, title="Test Voxel", view="top")
    visualize_voxel_3d(test_voxel, title="Test 3D Voxel")
    
    print("Visualization test complete!")
