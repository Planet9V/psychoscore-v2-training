"""
Download required models for v2 training.

Downloads:
- MusicGen-Medium (base model)
- CLAP (for alignment scoring)
"""

import os
from pathlib import Path


def download_musicgen():
    """Download MusicGen-Medium."""
    print("📥 Downloading MusicGen-Medium...")
    
    try:
        from audiocraft.models import MusicGen
        
        # This will download and cache the model
        model = MusicGen.get_pretrained('medium', device='cpu')
        print("✓ MusicGen-Medium downloaded and cached")
        
        # Get cache location
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        print(f"  Cache: {cache_dir}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download MusicGen: {e}")
        return False


def download_clap():
    """Download CLAP model."""
    print("\n📥 Downloading CLAP (Large Music)...")
    
    try:
        import laion_clap
        
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # Downloads if not present
        print("✓ CLAP downloaded and cached")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download CLAP: {e}")
        return False


def main():
    print("=" * 60)
    print("PSYCHOSCORE v2 Model Downloader")
    print("=" * 60)
    
    results = {}
    
    # MusicGen
    results["musicgen"] = download_musicgen()
    
    # CLAP
    results["clap"] = download_clap()
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for model, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model}")
    
    if all(results.values()):
        print("\n✅ All models downloaded successfully!")
    else:
        print("\n⚠️ Some downloads failed. Check errors above.")


if __name__ == "__main__":
    main()
