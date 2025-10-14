#!/bin/bash
# Setup script for new RunPod instances
# Run this once when you start a new pod

set -e  # Exit on error

echo "🚀 Setting up Minecraft 3D AI on new RunPod instance..."
echo ""

# 1. Clone or pull repository
if [ ! -d "model N" ]; then
    echo "📦 Cloning repository..."
    git clone https://github.com/MProductionsmado/mode-N.git "model N"
    cd "model N"
else
    echo "📦 Repository exists, pulling latest changes..."
    cd "model N"
    git pull
fi

# 2. Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning sentence-transformers pyyaml nbtlib numpy

# 4. Download Sentence-Transformers model
echo "🤖 Downloading Sentence-Transformers model..."
python3 scripts/download_model.py

# 5. Create models directory
echo "📁 Creating models directory..."
mkdir -p models

# 6. Check CUDA
echo "🎮 Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "✅ Setup complete! Ready to train."
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python3 scripts/train_diffusion.py"
echo ""
