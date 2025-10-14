#!/bin/bash
# Setup script for RunPod
# Run this after cloning the repository on RunPod

echo "=========================================="
echo "Minecraft 3D AI - RunPod Setup"
echo "=========================================="
echo ""

# Set Hugging Face cache to persistent storage
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "✓ Environment variables set"
echo "  HF_HOME=$HF_HOME"
echo ""

# Create cache directories
mkdir -p /workspace/.cache/huggingface/hub
mkdir -p /workspace/.cache/huggingface/transformers
echo "✓ Cache directories created"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies!"
    exit 1
fi
echo "✓ Dependencies installed"

# Download sentence-transformers model
echo ""
echo "Downloading sentence-transformers model..."
python3 scripts/download_model.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to download model!"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data data/processed data/splits models logs generated evaluation
echo "✓ Directories created"

# Check for GPU
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Preprocess data: python3 scripts/preprocess_data.py"
echo "2. Train model:     python3 scripts/train.py"
echo "3. Generate assets: python3 scripts/generate.py --checkpoint models/best.ckpt --prompt 'oak tree'"
echo ""
