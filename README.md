# PSYCHOSCORE v2 Training Repository

> **Standalone CUDA Training for MusicGen Audio Generation**

This repository contains everything needed to train the PSYCHOSCORE v2 audio generation model on a CUDA-equipped machine.

## ⚠️ Requirements

- **CUDA GPU** with 10GB+ VRAM (RTX 3080+, RTX 4090, 5070Ti recommended)
- **Python 3.10-3.12** (NOT 3.14)
- **CUDA 11.8+**

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Planet9V/psychoscore-v2-training.git
cd psychoscore-v2-training

# 2. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements_cuda.txt

# 3. Download models
python scripts/download_models.py

# 4. Prepare data (copy from main project or use sample)
python scripts/prepare_training_data.py

# 5. Train
python train_performance_cuda.py --epochs 3 --device cuda

# 6. Export
python scripts/export_model.py --output psychoscore_v2_trained.tar.gz
```

## Purpose

The main PSYCHOSCORE project (`NPM_Music_Conductor_App`) cannot train the full v2 
audio model on macOS due to xformers/CUDA dependencies. This standalone repo allows:

1. **Complete v2 training** on any CUDA machine
2. **Export trained weights** as a portable archive
3. **Import back** to the main project for inference

## What Gets Trained

| Component | Description |
|-----------|-------------|
| **LoRA_Performance** | Converts musical skeletons to audio via MusicGen |
| **DPO Refinement** | Aligns output to psychometric profiles (optional) |

## After Training

Copy the trained model back to your main project:

```bash
# On CUDA machine
tar -czvf psychoscore_v2_trained.tar.gz checkpoints/

# On Mac (main project)
scp user@cuda-server:~/psychoscore-v2-training/psychoscore_v2_trained.tar.gz .
cd NPM_Music_Conductor_App/ml/psychoscore_v2
tar -xzvf ~/psychoscore_v2_trained.tar.gz
```

Then update `config/model_selection.yaml` to enable v2.

## Directory Structure

```
.
├── README.md                      # This file
├── requirements_cuda.txt          # CUDA dependencies
├── train_performance_cuda.py      # Main training script
├── config/
│   └── training_config.yaml       # Hyperparameters
├── scripts/
│   ├── download_models.py         # Download base models
│   ├── prepare_training_data.py   # Data preparation
│   ├── validate_training.py       # Validate outputs
│   └── export_model.py            # Package for export
├── data/                          # Training data (gitignored)
└── checkpoints/                   # Outputs (gitignored)
```

## License

Same as main PSYCHOSCORE project.
