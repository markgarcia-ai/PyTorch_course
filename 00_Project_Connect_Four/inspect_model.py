#!/usr/bin/env python
"""
Utility script to inspect .pth model files.
Usage: python inspect_model.py models/trained_model.pth
"""

import sys
import torch
from pathlib import Path

def inspect_model(model_path):
    """Inspect a PyTorch .pth model file."""
    
    print("=" * 60)
    print(f"INSPECTING MODEL: {model_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(model_path).exists():
        print(f"❌ Error: File not found: {model_path}")
        return
    
    # Get file size
    file_size = Path(model_path).stat().st_size
    print(f"\n📁 File Info:")
    print(f"   Path: {model_path}")
    print(f"   Size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    # Load the state dict
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"   ✅ File loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Display state dict info
    print(f"\n🔧 Model Structure:")
    print(f"   Number of layers: {len(state_dict)}")
    
    print(f"\n📊 Layer Details:")
    total_params = 0
    for i, (key, value) in enumerate(state_dict.items(), 1):
        param_count = value.numel()
        total_params += param_count
        print(f"   {i}. {key:30s} | Shape: {str(list(value.shape)):20s} | Params: {param_count:,}")
    
    print(f"\n📈 Summary:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Model Memory: {total_params * 4 / 1024:.2f} KB (float32)")
    
    # Try to load into actual model
    print(f"\n🧠 Loading into Model Architecture:")
    try:
        from models import ConnectFourNet
        model = ConnectFourNet()
        model.load_state_dict(state_dict)
        print(f"   ✅ Model loaded successfully!")
        
        # Show model architecture
        print(f"\n🏗️  Model Architecture:")
        print(model)
        
        # Test forward pass
        print(f"\n🧪 Testing Forward Pass:")
        test_input = torch.randn(1, 2, 4, 4)
        with torch.no_grad():
            output = model(test_input)
        print(f"   Input shape:  {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"   ✅ Model works correctly!")
        
    except Exception as e:
        print(f"   ⚠️  Could not load into model: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_model.pth>")
        print("\nExample:")
        print("  python inspect_model.py models/trained_model.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inspect_model(model_path)

