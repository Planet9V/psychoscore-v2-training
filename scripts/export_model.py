"""
Export trained v2 model for deployment.

Packages the trained LoRA adapters and config into a single archive
that can be imported to the main PSYCHOSCORE project.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/lora_performance
"""

import argparse
import os
import shutil
import tarfile
import json
from datetime import datetime
from pathlib import Path


def create_manifest(checkpoint_dir: str, output_dir: str) -> dict:
    """Create a manifest describing the exported model."""
    manifest = {
        "export_date": datetime.now().isoformat(),
        "model_type": "psychoscore_v2_performance",
        "base_model": "facebook/musicgen-medium",
        "adapter_type": "lora",
        "files": [],
        "import_instructions": {
            "target_dir": "ml/psychoscore_v2/checkpoints/lora_performance",
            "config_update": "config/model_selection.yaml",
            "verify_command": "python scripts/test_v2_inference.py"
        }
    }
    
    # List all files in checkpoint
    checkpoint_path = Path(checkpoint_dir)
    for file in checkpoint_path.rglob("*"):
        if file.is_file():
            rel_path = file.relative_to(checkpoint_path)
            manifest["files"].append({
                "path": str(rel_path),
                "size": file.stat().st_size
            })
    
    return manifest


def export_model(
    checkpoint_dir: str,
    output_path: str,
    include_dpo: bool = True
) -> str:
    """
    Export trained model as a tar.gz archive.
    
    Args:
        checkpoint_dir: Path to LoRA checkpoint
        output_path: Output archive path
        include_dpo: Also include DPO-aligned model if present
        
    Returns:
        Path to created archive
    """
    # Create temp staging directory
    staging_dir = Path("export_staging")
    staging_dir.mkdir(exist_ok=True)
    
    try:
        # Copy checkpoint
        checkpoint_path = Path(checkpoint_dir)
        dest_checkpoint = staging_dir / "lora_performance"
        
        if checkpoint_path.exists():
            shutil.copytree(checkpoint_path, dest_checkpoint)
            print(f"✓ Copied {checkpoint_dir}")
        else:
            print(f"✗ Checkpoint not found: {checkpoint_dir}")
            return None
        
        # Optionally include DPO model
        if include_dpo:
            dpo_path = checkpoint_path.parent / "dpo_aligned"
            if dpo_path.exists():
                shutil.copytree(dpo_path, staging_dir / "dpo_aligned")
                print(f"✓ Copied DPO aligned model")
        
        # Create manifest
        manifest = create_manifest(checkpoint_dir, str(staging_dir))
        with open(staging_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"✓ Created manifest.json")
        
        # Create import script
        import_script = '''#!/bin/bash
# Import trained v2 model into main project

set -e

echo "🔄 Importing PSYCHOSCORE v2 trained model..."

# Extract archive
tar -xzvf psychoscore_v2_trained.tar.gz -C .

# Move to correct location
mv lora_performance checkpoints/
if [ -d "dpo_aligned" ]; then
    mv dpo_aligned checkpoints/
fi

# Update config (manual step)
echo ""
echo "✅ Import complete!"
echo ""
echo "📝 Update config/model_selection.yaml:"
echo "   v2.lora_path: checkpoints/lora_performance"
echo ""
echo "🧪 Test with:"
echo "   python scripts/test_v2_inference.py"
'''
        with open(staging_dir / "import.sh", "w") as f:
            f.write(import_script)
        os.chmod(staging_dir / "import.sh", 0o755)
        print(f"✓ Created import.sh")
        
        # Create archive
        with tarfile.open(output_path, "w:gz") as tar:
            for item in staging_dir.iterdir():
                tar.add(item, arcname=item.name)
        
        print(f"\n✅ Export complete: {output_path}")
        print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        
        return output_path
        
    finally:
        # Cleanup staging
        shutil.rmtree(staging_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Export v2 trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_performance",
                        help="Path to LoRA checkpoint")
    parser.add_argument("--output", type=str, default="psychoscore_v2_trained.tar.gz",
                        help="Output archive path")
    parser.add_argument("--no-dpo", action="store_true",
                        help="Don't include DPO model")
    
    args = parser.parse_args()
    
    export_model(
        args.checkpoint,
        args.output,
        include_dpo=not args.no_dpo
    )


if __name__ == "__main__":
    main()
