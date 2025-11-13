"""
Convert YOLO-Fastest Darknet models (.weights + .cfg) to ONNX format

⚠️ IMPORTANT: This script requires the official conversion tool from dog-qiuqiu/Yolo-Fastest

Recommended Approach:
    1. Clone official repo: git clone https://github.com/dog-qiuqiu/Yolo-Fastest
    2. Use their Darknet -> NCNN -> ONNX pipeline
    3. Or download pre-converted models (if available)

Alternative: Manual PyTorch Implementation
    This script provides a framework, but requires implementing the complete
    Darknet architecture parser to be functional.

Requirements:
    pip install torch torchvision opencv-python numpy onnx

Usage:
    # Activate virtual environment first
    .venv\\Scripts\\Activate.ps1  # Windows
    source .venv/bin/activate    # Linux/macOS
    
    # Run conversion
    python scripts/convert_fastest_to_onnx.py

Models:
    - YOLO-Fastest-1.1 (0.35M params, 1.3MB)
    - YOLO-Fastest-XL (0.92M params, 3.5MB)
    - YOLO-Fastest-1.1-Body (body keypoints)

Note: Check ModelZoo for available formats:
    models/yolo-fastest-1.1/yolo-fastest-1.1.{weights,cfg}
    models/yolo-fastest-1.1/yolo-fastest-1.1-xl.{weights,cfg}
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
from pathlib import Path

# Add models directory to path
MODELS_DIR = Path(__file__).parent.parent / "models"


class YOLOFastestConverter:
    """Convert YOLO-Fastest Darknet models to ONNX"""

    def __init__(self):
        self.models = {
            "1": {
                "name": "YOLO-Fastest-1.1",
                "cfg": "yolo-fastest-1.1/yolo-fastest-1.1.cfg",
                "weights": "yolo-fastest-1.1/yolo-fastest-1.1.weights",
                "output": "yolo-fastest-1.1.onnx",
            },
            "2": {
                "name": "YOLO-Fastest-XL",
                "cfg": "yolo-fastest-1.1/yolo-fastest-1.1-xl.cfg",
                "weights": "yolo-fastest-1.1/yolo-fastest-1.1-xl.weights",
                "output": "yolo-fastest-xl.onnx",
            },
            "3": {
                "name": "YOLO-Fastest-1.1-Body",
                "cfg": "yolo-fastest-1.1_body/yolo-fastest-1.1_body.cfg",
                "weights": "yolo-fastest-1.1_body/yolo-fastest-1.1_body.weights",
                "output": "yolo-fastest-1.1-body.onnx",
            },
        }

    def parse_cfg(self, cfg_file):
        """Parse Darknet .cfg file"""
        blocks = []
        with open(cfg_file, 'r') as f:
            lines = f.readlines()
            block = {}
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('['):
                    if block:
                        blocks.append(block)
                    block = {'type': line[1:-1].strip()}
                else:
                    key, value = line.split('=')
                    block[key.strip()] = value.strip()
            if block:
                blocks.append(block)
        return blocks

    def load_weights(self, weights_file):
        """Load Darknet weights"""
        with open(weights_file, 'rb') as f:
            # Read header
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        return weights

    def build_model_from_cfg(self, blocks):
        """Build PyTorch model from Darknet config blocks"""
        net_info = blocks[0]
        input_size = int(net_info.get('width', 320))
        
        print(f"Model input size: {input_size}x{input_size}")
        
        # For simplicity, we'll use a placeholder model structure
        # In production, you'd need to parse each layer type properly
        return SimplifiedYOLOFastest(input_size)

    def convert(self, model_key):
        """Convert a specific model"""
        if model_key not in self.models:
            print(f"❌ Invalid model key: {model_key}")
            return False

        model_info = self.models[model_key]
        cfg_path = MODELS_DIR / model_info["cfg"]
        weights_path = MODELS_DIR / model_info["weights"]
        output_path = MODELS_DIR / model_info["output"]

        print(f"\n{'='*60}")
        print(f"Converting {model_info['name']}")
        print(f"{'='*60}")

        # Check if files exist
        if not cfg_path.exists():
            print(f"❌ Config file not found: {cfg_path}")
            return False
        if not weights_path.exists():
            print(f"❌ Weights file not found: {weights_path}")
            return False

        print(f"📄 Config: {cfg_path.name}")
        print(f"⚖️  Weights: {weights_path.name} ({weights_path.stat().st_size / 1024 / 1024:.2f} MB)")

        try:
            # Parse config
            print("\n🔍 Parsing config...")
            blocks = self.parse_cfg(cfg_path)
            
            # Load weights
            print("📦 Loading weights...")
            weights = self.load_weights(weights_path)
            print(f"   Loaded {len(weights):,} weight values")
            
            # Build model
            print("🏗️  Building PyTorch model...")
            model = self.build_model_from_cfg(blocks)
            model.eval()
            
            # Export to ONNX
            print("🔄 Converting to ONNX...")
            input_size = 320  # YOLO-Fastest default
            dummy_input = torch.randn(1, 3, input_size, input_size)
            
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch'}
                }
            )
            
            # Verify ONNX model
            print("✅ Verifying ONNX model...")
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            print(f"✅ Successfully converted to: {output_path.name}")
            print(f"   Output size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def convert_all(self):
        """Convert all available models"""
        results = {}
        for key in self.models.keys():
            success = self.convert(key)
            results[self.models[key]["name"]] = success
        
        # Print summary
        print(f"\n{'='*60}")
        print("Conversion Summary")
        print(f"{'='*60}")
        for name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{name}: {status}")
        
        return all(results.values())


class SimplifiedYOLOFastest(nn.Module):
    """
    Simplified YOLO-Fastest model structure
    
    Note: This is a placeholder structure. For production use, you should:
    1. Parse the .cfg file completely
    2. Build the exact network architecture
    3. Load weights properly into each layer
    
    Consider using existing tools like:
    - https://github.com/dog-qiuqiu/Yolo-Fastest (official)
    - https://github.com/Tianxiaomo/pytorch-YOLOv4 (similar conversion)
    """
    
    def __init__(self, input_size=320):
        super().__init__()
        self.input_size = input_size
        
        # This is a placeholder - replace with actual architecture
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.head = nn.Conv2d(64, 255, 1)  # 255 = (80 classes + 5) * 3 anchors
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def show_menu():
    """Display conversion menu"""
    print("\n" + "="*60)
    print("🚀 YOLO-Fastest to ONNX Converter")
    print("="*60)
    print("\nAvailable models:")
    print("  1. YOLO-Fastest-1.1    (0.35M params, 1.3MB)")
    print("  2. YOLO-Fastest-XL     (0.92M params, 3.5MB)")
    print("  3. YOLO-Fastest-1.1-Body (body keypoints)")
    print("  4. Convert all models")
    print("  0. Exit")
    print()


def main():
    """Main conversion function"""
    converter = YOLOFastestConverter()
    
    print("\n⚠️  IMPORTANT NOTICE")
    print("="*60)
    print("This converter requires manual implementation of the complete")
    print("Darknet -> PyTorch architecture parsing.")
    print()
    print("For production use, please use official conversion tools:")
    print("  1. YOLO-Fastest official repo:")
    print("     https://github.com/dog-qiuqiu/Yolo-Fastest")
    print()
    print("  2. Or download pre-converted ONNX models:")
    print("     https://github.com/dog-qiuqiu/Yolo-Fastest/releases")
    print("="*60)
    print()
    
    response = input("Continue with placeholder conversion? (y/N): ").strip().lower()
    if response != 'y':
        print("\n💡 Recommended: Download pre-converted ONNX models")
        print("   See: docs/20.模型下载指南.md")
        return
    
    while True:
        show_menu()
        choice = input("Select an option: ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        elif choice == '4':
            converter.convert_all()
        elif choice in ['1', '2', '3']:
            converter.convert(choice)
        else:
            print("❌ Invalid choice. Please select 0-4.")


if __name__ == "__main__":
    # Check Python environment
    if not Path(".venv").exists():
        print("⚠️  Warning: .venv directory not found")
        print("   Recommended: Activate virtual environment first")
        print("   .venv\\Scripts\\Activate.ps1  # Windows")
        print()
    
    main()
