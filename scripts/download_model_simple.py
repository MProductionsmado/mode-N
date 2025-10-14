#!/usr/bin/env python3
"""
Simple direct download of sentence-transformers model
This bypasses hf_transfer and other complications
"""

import os
import sys

# Disable hf_transfer to avoid dependency issues
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)

print("="*60)
print("Downloading Sentence-Transformers Model")
print("="*60)
print()

try:
    from sentence_transformers import SentenceTransformer
    import torch
    
    print("Step 1: Setting up cache...")
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"✓ Cache directory: {cache_dir}")
    
    print("\nStep 2: Downloading model (this may take a few minutes)...")
    print("Model: sentence-transformers/all-MiniLM-L6-v2")
    print("Size: ~90 MB")
    print()
    
    # Download the model directly
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("\n✓ Model downloaded successfully!")
    
    print("\nStep 3: Testing model...")
    test_texts = [
        "oak tree with green leaves",
        "big spruce forest",
        "small birch tree"
    ]
    
    embeddings = model.encode(test_texts, show_progress_bar=False)
    
    print(f"✓ Model works correctly!")
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    # Show model info
    print("\nModel Information:")
    print(f"  Max sequence length: {model.max_seq_length}")
    print(f"  Device: {model.device}")
    
    print("\n" + "="*60)
    print("SUCCESS! Model is ready for training.")
    print("="*60)
    print("\nYou can now run:")
    print("  python3 scripts/preprocess_data.py")
    print("  python3 scripts/train.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nDiagnostics:")
    print(f"  Python version: {sys.version}")
    
    try:
        import sentence_transformers
        print(f"  sentence-transformers version: {sentence_transformers.__version__}")
    except:
        print("  sentence-transformers: NOT INSTALLED")
    
    try:
        import transformers
        print(f"  transformers version: {transformers.__version__}")
    except:
        print("  transformers: NOT INSTALLED")
    
    print("\nTroubleshooting:")
    print("1. Unset HF_HUB_ENABLE_HF_TRANSFER:")
    print("   unset HF_HUB_ENABLE_HF_TRANSFER")
    print()
    print("2. Install missing packages:")
    print("   pip install sentence-transformers transformers")
    print()
    print("3. Try manual download:")
    print("   from huggingface_hub import snapshot_download")
    print("   snapshot_download('sentence-transformers/all-MiniLM-L6-v2')")
    
    sys.exit(1)
