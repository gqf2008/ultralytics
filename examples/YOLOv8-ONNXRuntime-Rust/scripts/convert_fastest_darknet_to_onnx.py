"""
Convert YOLO-Fastest from Darknet to ONNX using export_onnx.py

Based on: https://blog.csdn.net/linghu8812/article/details/109270320
Reference: https://github.com/linghu8812/tensorrt_inference

This script wraps the conversion tool from linghu8812's tensorrt_inference repo.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def download_conversion_tool():
    """Clone the tensorrt_inference repo if not exists"""
    repo_path = Path("tensorrt_inference")
    
    if repo_path.exists():
        print(f"✅ Conversion tool already exists at: {repo_path}")
        return repo_path
    
    print("📥 Cloning conversion tool from GitHub...")
    cmd = [
        "git", "clone",
        "https://github.com/linghu8812/tensorrt_inference.git"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully cloned to: {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repo: {e}")
        sys.exit(1)

def convert_yolo_fastest():
    """Convert YOLO-Fastest models to ONNX"""
    print("\n" + "="*60)
    print("🚀 YOLO-Fastest Darknet → ONNX Converter")
    print("="*60)
    
    # Model configurations
    models = {
        "1": {
            "name": "YOLO-Fastest-1.1",
            "cfg": "models/yolo-fastest-1.1/yolo-fastest-1.1.cfg",
            "weights": "models/yolo-fastest-1.1/yolo-fastest-1.1.weights",
            "output": "models/yolo-fastest-1.1.onnx",
            "strides": "32 16",  # YOLOv4-tiny uses FPN
            "neck": "FPN"
        },
        "2": {
            "name": "YOLO-Fastest-XL",
            "cfg": "models/yolo-fastest-1.1/yolo-fastest-1.1-xl.cfg",
            "weights": "models/yolo-fastest-1.1/yolo-fastest-1.1-xl.weights",
            "output": "models/yolo-fastest-xl.onnx",
            "strides": "32 16",
            "neck": "FPN"
        }
    }
    
    # Show menu
    print("\nAvailable models:")
    for key, model in models.items():
        print(f"  {key}. {model['name']}")
        print(f"     Config: {model['cfg']}")
        print(f"     Weights: {model['weights']}")
    print("  0. Exit")
    
    choice = input("\nSelect model to convert (1/2/0): ").strip()
    
    if choice == "0":
        print("\n👋 Goodbye!")
        return
    
    if choice not in models:
        print("❌ Invalid choice")
        return
    
    model = models[choice]
    
    # Check if files exist
    cfg_path = Path(model["cfg"])
    weights_path = Path(model["weights"])
    
    if not cfg_path.exists():
        print(f"❌ Config file not found: {cfg_path}")
        print("   Please ensure the model files are in the correct location")
        return
    
    if not weights_path.exists():
        print(f"❌ Weights file not found: {weights_path}")
        print("   Please ensure the model files are in the correct location")
        return
    
    # Download conversion tool
    repo_path = download_conversion_tool()
    
    # Prepare conversion command
    script_path = repo_path / "project" / "Yolov4" / "export_onnx.py"
    
    if not script_path.exists():
        print(f"❌ Conversion script not found: {script_path}")
        return
    
    cmd = [
        "python", str(script_path),
        "--cfg_file", str(cfg_path),
        "--weights_file", str(weights_path),
        "--output_file", model["output"],
        "--strides", model["strides"],
        "--neck", model["neck"]
    ]
    
    print(f"\n🔄 Converting {model['name']}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Successfully converted!")
        print(f"   Output: {model['output']}")
        
        # Verify output
        output_path = Path(model["output"])
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"   Size: {size_mb:.2f} MB")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nPossible issues:")
        print("  1. Missing dependencies (torch, onnx, numpy)")
        print("  2. Incompatible model format")
        print("  3. Path issues")
        print("\nTry installing dependencies:")
        print("  pip install torch torchvision onnx numpy")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert YOLO-Fastest from Darknet to ONNX"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-convert all models without prompts"
    )
    args = parser.parse_args()
    
    if not Path(".venv").exists():
        print("⚠️  Warning: Virtual environment not found")
        print("   Recommended: Activate .venv first")
        print("   .venv\\Scripts\\Activate.ps1  # Windows")
        print()
    
    if args.auto:
        # Auto mode: convert all
        print("🤖 Auto mode: Converting all models...")
        # Implementation for auto mode
        pass
    else:
        # Interactive mode
        convert_yolo_fastest()

if __name__ == "__main__":
    main()
