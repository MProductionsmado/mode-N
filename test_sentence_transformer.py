"""Test if sentence-transformers can load the model"""
import os
print("Testing sentence-transformers model loading...")
print(f"Cache directory: {os.path.expanduser('~/.cache/huggingface')}")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence_transformers imported successfully")
    
    print("\nDownloading/loading model 'sentence-transformers/all-MiniLM-L6-v2'...")
    print("This may take a few minutes on first run...")
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✓ Model loaded successfully!")
    
    # Test encoding
    print("\nTesting encoding...")
    test_text = "oak tree with leaves"
    embedding = model.encode(test_text)
    print(f"✓ Encoding works! Embedding shape: {embedding.shape}")
    print(f"✓ Embedding dimension: {len(embedding)}")
    
    print("\n" + "="*50)
    print("SUCCESS! Sentence-Transformers is working correctly.")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Try setting HF_HOME environment variable:")
    print("   $env:HF_HOME = 'C:/Users/priva/.cache/huggingface'")
    print("3. Or manually download the model")
    import sys
    sys.exit(1)
