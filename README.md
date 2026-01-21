# PSYCHOSCORE v2 Training Repository

> **Standalone CUDA Training for MusicGen Audio Generation**

This repository contains everything needed to train the PSYCHOSCORE v2 audio generation model on a CUDA-equipped machine.

## 🚀 One-Click Training

If you have a Linux/WSL machine with an NVIDIA GPU (e.g., RTX 5070Ti):

1. **Clone:**
   ```bash
   git clone https://github.com/Planet9V/psychoscore-v2-training.git
   cd psychoscore-v2-training
   ```

2. **Run:**
   ```bash
   ./run_training.sh
   ```

That's it! The script will:
* Set up Python environment
* Install dependencies
* Download base models
* Generate/Prepare data
* Train the model
* Export `psychoscore_v2_trained.tar.gz`

## Manual Steps

If you prefer to run steps manually:

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements_cuda.txt

# 2. Download models
python scripts/download_models.py

# 3. Prepare data (copy from main project or use sample)
python scripts/prepare_training_data.py

# 4. Train
python train_performance_cuda.py --epochs 3 --device cuda

# 5. Export
python scripts/export_model.py --output psychoscore_v2_trained.tar.gz
```

## What Gets Trained

| Component | Description |
|-----------|-------------|
| **LoRA_Performance** | Converts musical skeletons to audio via MusicGen |
| **DPO Refinement** | Aligns output to psychometric profiles (optional) |

## After Training

Copy the trained model back to your main project:

```bash
# On CUDA machine
# (The tar.gz is created automatically by run_training.sh)

# On Mac (main project)
scp user@cuda-server:~/psychoscore-v2-training/psychoscore_v2_trained.tar.gz .
cd NPM_Music_Conductor_App/ml/psychoscore_v2
tar -xzvf ~/psychoscore_v2_trained.tar.gz
```

Then update `config/model_selection.yaml` to enable v2.

## Directory Structure

```
.
├── run_training.sh            # ONE-CLICK AUTOMATION SCRIPT
├── README.md                  # This file
├── requirements_cuda.txt      # CUDA dependencies
├── train_performance_cuda.py  # Main training script
├── config/
│   └── training_config.yaml   # Hyperparameters
├── scripts/
│   ├── download_models.py     # Download base models
│   ├── prepare_training_data.py # Data preparation
│   ├── validate_training.py   # Validate outputs
│   └── export_model.py        # Package for export
├── data/                      # Training data (gitignored)
└── checkpoints/               # Outputs (gitignored)
```

## License

Same as main PSYCHOSCORE project.
