"""
Prepare training data for Psychology-to-Music (Performance LoRA).

If no data exists, creates synthetic data to allow the pipeline to run.
This ensures the repository is self-sufficient.
"""

import os
import json
import argparse
import random
import torch
import torchaudio
from pathlib import Path

def generate_sine_wave(
    duration: float, 
    sample_rate: int, 
    frequency: float = 440.0
) -> torch.Tensor:
    """Generate a simple sine wave."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    return torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

def create_synthetic_data(
    output_file: str, 
    audio_dir: str, 
    samples: int = 10
):
    """Create synthetic training dataset."""
    print(f"Generating {samples} synthetic training samples...")
    
    os.makedirs(audio_dir, exist_ok=True)
    dataset = []
    
    conditions = [
        "A happy piano melody",
        "Sad violin solo",
        "Energetic drums",
        "Calm ambient texture",
        "Fast tempo jazz"
    ]
    
    for i in range(samples):
        # Generate random audio file
        filename = f"synthetic_{i:03d}.wav"
        path = os.path.join(audio_dir, filename)
        
        # Random duration 5-10s
        duration = random.uniform(5.0, 10.0)
        freq = random.uniform(220.0, 880.0)
        
        waveform = generate_sine_wave(duration, 32000, freq)
        torchaudio.save(path, waveform, 32000)
        
        # Create metadata
        entry = {
            "id": f"syn_{i}",
            "audio_path": filename,  # Relative to audio_dir
            "condition": random.choice(conditions),
            "psychometrics": {
                "trauma": random.random(),
                "entropy": random.random()
            }
        }
        dataset.append(entry)
    
    # Save JSONL
    with open(output_file, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"✓ Synthetic data created at {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/train_performance.jsonl")
    parser.add_argument("--audio_dir", default="data/audio")
    parser.add_argument("--force_synthetic", action="store_true", help="Force synthetic data generation")
    args = parser.parse_args()
    
    data_path = Path(args.data_file)
    
    if data_path.exists() and not args.force_synthetic:
        print(f"✓ Training data found at {data_path}")
        # Validate minimal content
        with open(data_path) as f:
            if sum(1 for _ in f) > 0:
                print("  Data file is valid.")
                return
    
    print("⚠️  No training data found (or empty). Creating synthetic dataset...")
    create_synthetic_data(args.data_file, args.audio_dir)
    print("✅ Data preparation complete.")

if __name__ == "__main__":
    main()
