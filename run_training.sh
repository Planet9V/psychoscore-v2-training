#!/bin/bash
# One-Click Training Script for Psychoscore v2 (CUDA)
# ===================================================

set -e  # Exit on error

echo "🚀 Starting Psychoscore v2 Training Pipeline"
echo "============================================"

# 1. Environment Check
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. CUDA might not be available."
    echo "   Training may fail or run extremely slowly on CPU."
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔌 Activating environment..."
source venv/bin/activate || source venv/Scripts/activate

echo "⬇️  Installing dependencies..."
pip install --upgrade pip
# Install torch first to ensure correct CUDA version if needed
# Assuming system CUDA 11.8/12.x compatible torch is simpler to let pip handle or pre-installed
pip install -r requirements_cuda.txt

# 3. Download Models
echo "📥 Checking model checkpoints..."
python scripts/download_models.py

# 4. Prepare Data
echo "📀 Preparing training data..."
# Ensure data directories exist
mkdir -p data/audio
python scripts/prepare_training_data.py

# 5. Run Training
echo "🔥 STARTING TRAINING LOOP..."
echo "   Output: checkpoints/lora_performance"
# Use arguments if passed to script, else default
python train_performance_cuda.py "${@:---epochs 3}"

# 6. Export
echo "📦 Exporting trained model..."
python scripts/export_model.py

echo "============================================"
echo "✅ PIPELINE COMPLETE"
echo "   Trained Model: psychoscore_v2_trained.tar.gz"
echo "============================================"
echo "To use this model:"
echo "1. Transfer 'psychoscore_v2_trained.tar.gz' to your main machine"
echo "2. Run import script or extract manually"
