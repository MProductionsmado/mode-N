#!/usr/bin/env python3
"""
Test connectivity to Hugging Face and download the model using huggingface_hub
"""

import os
import sys

def test_connectivity():
    """Test if we can reach Hugging Face"""
    print("Testing connectivity to Hugging Face...")
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("✓ Can reach huggingface.co")
            return True
        else:
            print(f"⚠ Hugging Face returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach huggingface.co: {e}")
        return False

def download_with_hub():
    """Download model using huggingface_hub"""
    print("\nTrying to download model with huggingface_hub...")
    try:
        from huggingface_hub import snapshot_download
        
        # Disable hf_transfer to avoid dependency issues
        if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
            del os.environ['HF_HUB_ENABLE_HF_TRANSFER']
        
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        
        print(f"Downloading {model_id}...")
        print(f"Cache directory: {cache_dir}")
        
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir
        )
        
        print(f"✓ Model downloaded to: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return None

def test_sentence_transformers(model_path=None):
    """Test loading with sentence-transformers"""
    print("\nTesting sentence-transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Disable hf_transfer to avoid dependency issues
        if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
            del os.environ['HF_HUB_ENABLE_HF_TRANSFER']
        
        if model_path:
            print(f"Loading from local path: {model_path}")
            model = SentenceTransformer(model_path)
        else:
            # Try different model name formats
            print("Loading model: sentence-transformers/all-MiniLM-L6-v2")
            try:
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except:
                print("Trying alternative format: all-MiniLM-L6-v2")
                model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test encoding
        test_text = "oak tree with green leaves"
        embedding = model.encode(test_text)
        
        print(f"✓ Model loaded successfully!")
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"✓ Embedding dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Hugging Face Model Download Test")
    print("="*60)
    print()
    
    # Step 1: Test connectivity
    if not test_connectivity():
        print("\n⚠ Warning: Cannot reach Hugging Face. Check your internet connection.")
        print("If you're behind a proxy, you may need to set:")
        print("  export HTTP_PROXY=your_proxy")
        print("  export HTTPS_PROXY=your_proxy")
    
    # Step 2: Try downloading with huggingface_hub
    model_path = download_with_hub()
    
    # Step 3: Test sentence-transformers
    success = test_sentence_transformers(model_path)
    
    print("\n" + "="*60)
    if success:
        print("✓ SUCCESS! Model is ready.")
        print("="*60)
        sys.exit(0)
    else:
        print("❌ FAILED! See errors above.")
        print("="*60)
        sys.exit(1)
