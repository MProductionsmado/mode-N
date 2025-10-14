#!/usr/bin/env python3
"""
Pre-download the sentence-transformers model to avoid issues during training.
Run this script before starting training, especially on cloud platforms like RunPod.
"""

import os
import sys

def setup_huggingface_cache():
    """Setup Hugging Face cache directory"""
    # Set cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
    os.environ['HF_HUB_CACHE'] = os.path.join(cache_dir, 'hub')
    
    print(f"✓ Cache directory set to: {cache_dir}")
    return cache_dir

def download_model(model_name='all-MiniLM-L6-v2'):
    """Download the sentence-transformers model"""
    print(f"\n{'='*60}")
    print(f"Downloading model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        cache_dir = setup_huggingface_cache()
        
        print("Loading model (this may take a few minutes on first run)...")
        print("Note: This will download ~90MB from Hugging Face...")
        model = SentenceTransformer(model_name)
        
        print("\n✓ Model downloaded successfully!")
        
        # Test the model
        print("\nTesting model...")
        test_text = "oak tree with green leaves"
        embedding = model.encode(test_text)
        print(f"✓ Model works! Embedding shape: {embedding.shape}")
        print(f"✓ Embedding dimension: {len(embedding)}")
        
        # Show model location
        print(f"\n✓ Model cached at: {cache_dir}")
        
        print(f"\n{'='*60}")
        print("SUCCESS! Model is ready for training.")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Check if you can access huggingface.co")
        print("3. Try setting environment variables:")
        print("   export HF_HOME=/workspace/.cache/huggingface")
        print("   export HF_HUB_DISABLE_SYMLINKS_WARNING=1")
        print("\n4. Or download manually:")
        print(f"   huggingface-cli download {model_name}")
        
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Sentence-Transformers Model Downloader")
    print("="*60 + "\n")
    
    success = download_model()
    
    if success:
        print("You can now run:")
        print("  python scripts/preprocess_data.py")
        print("  python scripts/train.py")
        sys.exit(0)
    else:
        print("\nFailed to download model. Please fix the issues above.")
        sys.exit(1)
