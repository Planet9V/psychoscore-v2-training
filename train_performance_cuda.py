"""
PSYCHOSCORE v2 Performance LoRA Training (CUDA)

This script trains the LoRA_Performance adapter on a CUDA-equipped machine.
It requires xformers and CUDA to run.

Usage:
    python train_performance_cuda.py --epochs 3 --device cuda
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# These require CUDA
try:
    import xformers
    print(f"✓ xformers available: {xformers.__version__}")
except ImportError:
    print("✗ xformers not available - this script requires CUDA")
    exit(1)

from audiocraft.models import MusicGen
from peft import LoraConfig, get_peft_model


class PerformanceDataset(Dataset):
    """
    Dataset for Performance LoRA training.
    
    Each item contains:
    - audio: Target audio waveform
    - condition: Text condition (musical skeleton description)
    - psychometrics: Psychometric profile (optional)
    """
    
    def __init__(
        self,
        data_file: str,
        audio_dir: str,
        duration: float = 10.0,
        sample_rate: int = 32000
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSONL file with training data
            audio_dir: Directory containing audio files
            duration: Target duration in seconds
            sample_rate: Target sample rate
        """
        self.audio_dir = Path(audio_dir)
        self.duration = duration
        self.sample_rate = sample_rate
        
        # Load data
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Verify audio exists
                audio_path = self.audio_dir / item.get("audio_path", item.get("audio"))
                if audio_path.exists():
                    item["full_audio_path"] = str(audio_path)
                    self.data.append(item)
        
        print(f"Loaded {len(self.data)} training examples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(item["full_audio_path"])
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad/crop to target duration
        target_len = int(self.duration * self.sample_rate)
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_len]
        
        return {
            "audio": waveform,
            "condition": item.get("condition", item.get("text", "")),
            "psychometrics": item.get("psychometrics", {})
        }


def train_epoch(
    model: MusicGen,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.lm.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        audio = batch["audio"].to(device)
        conditions = batch["condition"]
        
        optimizer.zero_grad()
        
        # Encode audio to codes
        with torch.no_grad():
            codes, _ = model.compression_model.encode(audio)
        
        # Prepare conditioning
        attributes, _ = model._prepare_tokens_and_attributes(conditions, None)
        
        # Forward pass through LM
        # MusicGen uses internal loss computation
        # We need to access the LM's forward with teacher forcing
        
        # Flatten codes for LM input
        B, K, T = codes.shape
        
        # Create input sequence (shift right)
        input_codes = codes[:, :, :-1]
        target_codes = codes[:, :, 1:]
        
        # Get condition tensors
        condition_tensors = model._prepare_condition_tensors(attributes)
        
        # Forward LM
        logits = model.lm(input_codes, condition_tensors)
        
        # Compute loss (cross-entropy over codebook predictions)
        # logits shape: (B, K, T, vocab)
        # target shape: (B, K, T)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_codes.reshape(-1),
            ignore_index=-100
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Performance LoRA (CUDA)")
    parser.add_argument("--data_file", type=str, default="data/train_performance.jsonl",
                        help="Path to training data JSONL")
    parser.add_argument("--audio_dir", type=str, default="data/audio",
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_performance",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda required)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Audio duration in seconds")
    
    args = parser.parse_args()
    
    # Verify CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires CUDA.")
        exit(1)
    
    device = args.device
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MusicGen
    print("\n🎵 Loading MusicGen-Medium...")
    model = MusicGen.get_pretrained('medium', device=device)
    
    # Freeze compression model
    for param in model.compression_model.parameters():
        param.requires_grad = False
    
    # Apply LoRA to LM
    print("\n🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["out_proj", "linear1", "linear2"],
        lora_dropout=0.1,
        bias="none"
    )
    
    model.lm = get_peft_model(model.lm, lora_config)
    model.lm.print_trainable_parameters()
    
    # Load dataset
    print("\n📚 Loading dataset...")
    dataset = PerformanceDataset(
        args.data_file,
        args.audio_dir,
        duration=args.duration
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = AdamW(model.lm.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Total steps: {total_steps}")
    
    best_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, dataloader, optimizer, scheduler, device, epoch
        )
        
        print(f"Epoch {epoch}: avg_loss = {train_loss:.4f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            model.lm.save_pretrained(args.output_dir)
            print(f"  → Saved best model (loss={train_loss:.4f})")
    
    # Final save
    model.lm.save_pretrained(args.output_dir)
    
    print(f"\n✅ Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Checkpoints: {args.output_dir}")
    print(f"\n📦 To export for deployment:")
    print(f"   python scripts/export_model.py --checkpoint {args.output_dir}")


if __name__ == "__main__":
    main()
