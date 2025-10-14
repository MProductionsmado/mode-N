"""
Download Sentence-Transformers model for offline use
Run this once on a new machine/pod to cache the model locally
"""

import logging
import os
from sentence_transformers import SentenceTransformer

# Disable HF_TRANSFER to avoid dependency issues
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    model_name = "all-MiniLM-L6-v2"
    
    logger.info(f"Downloading Sentence-Transformers model: {model_name}")
    logger.info("This will cache the model in ~/.cache/torch/sentence_transformers/")
    
    try:
        # This will download and cache the model
        model = SentenceTransformer(model_name)
        logger.info(f"✓ Model '{model_name}' downloaded successfully!")
        logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test encoding
        test_text = "oak tree"
        embedding = model.encode(test_text)
        logger.info(f"✓ Test encoding successful! Embedding shape: {embedding.shape}")
        
    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        raise


if __name__ == "__main__":
    main()
