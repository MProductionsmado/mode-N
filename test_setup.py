"""
Quick test script to verify installation and basic functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
        
        import nbtlib
        print(f"✓ nbtlib installed")
        
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning {pl.__version__}")
        
        from sentence_transformers import SentenceTransformer
        print(f"✓ Sentence Transformers installed")
        
        import yaml
        print(f"✓ PyYAML installed")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - training will use CPU")
            return False
            
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def test_file_structure():
    """Test if required directories exist"""
    print("\nTesting file structure...")
    
    required_dirs = [
        'config',
        'src/data',
        'src/models',
        'src/training',
        'src/inference',
        'scripts',
        'out'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ not found")
            all_exist = False
    
    return all_exist


def test_config():
    """Test if config file is valid"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        
        config_path = Path('config/config.yaml')
        if not config_path.exists():
            print("✗ config/config.yaml not found")
            return False
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print("✓ Config file loaded")
        print(f"  Model sizes: {list(config['model']['sizes'].keys())}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Device: {config['hardware']['device']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def test_schematic_files():
    """Test if schematic files are accessible"""
    print("\nTesting schematic files...")
    
    out_dir = Path('out')
    if not out_dir.exists():
        print("✗ 'out' directory not found")
        return False
    
    schem_files = list(out_dir.glob('*.schem'))
    
    if len(schem_files) == 0:
        print("✗ No .schem files found in 'out' directory")
        return False
    
    print(f"✓ Found {len(schem_files)} .schem files")
    print(f"  Sample: {schem_files[0].name}")
    
    return True


def test_model_creation():
    """Test if model can be instantiated"""
    print("\nTesting model creation...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        
        import yaml
        import torch
        from src.models.vae_3d import ConditionalVAE3D
        
        # Load config
        with open('config/config.yaml') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = ConditionalVAE3D(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Encoders: {list(model.encoders.keys())}")
        print(f"  Decoders: {list(model.decoders.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Minecraft 3D Asset Generator - System Test")
    print("="*60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Package imports", test_imports()))
    results.append(("CUDA support", test_cuda()))
    results.append(("File structure", test_file_structure()))
    results.append(("Configuration", test_config()))
    results.append(("Schematic files", test_schematic_files()))
    results.append(("Model creation", test_model_creation()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {test_name}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. python scripts/preprocess_data.py")
        print("2. python scripts/train.py")
        print("3. python scripts/generate.py --checkpoint models/best.ckpt --prompt 'oak tree'")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
