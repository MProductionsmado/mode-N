#!/bin/bash
# Setup script for new RunPod instances
# Run this once when you start a new pod

set -e  # Exit on error

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
unset HF_HUB_ENABLE_HF_TRANSFER
python3 scripts/download_model.py
python scripts/preprocess_data.py
python scripts/train_discrete_diffusion.py

echo ""
echo "✅ Setup complete! Ready to train."
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python3 scripts/train_diffusion.py"
echo ""
