import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Mock natten
mock_natten = MagicMock()
sys.modules["natten"] = mock_natten
# Need to mock NeighborhoodAttention2D specifically as a class
class MockNeighborhoodAttention2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
mock_natten.NeighborhoodAttention2D = MockNeighborhoodAttention2D
# Also mock hasattr(natten, "context")
mock_natten.context = True

from ac_kan import ACKAN
from MedViTv3 import MedViTv3, LFPv3, GFPv3, LocalityFeedForwardV3

def test_ackan():
    print("Testing ACKAN...")
    channels = 64
    groups = 4
    x = torch.randn(2, channels, 16, 16)
    model = ACKAN(channels, groups=groups)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Output shape mismatch"
    print("ACKAN forward pass successful.")

def test_medvitv3_integration():
    print("\nTesting MedViTv3 integration...")
    model = MedViTv3(use_ackan=True, num_classes=10)
    
    # Check if ACKAN is present in LFP
    print("Checking LFP layers...")
    lfp_found_ackan = False
    for module in model.modules():
        if isinstance(module, LFPv3):
            # Check FFN
            ffn = module.ffn
            # Check layers in FFN
            for layer in ffn.net:
                if isinstance(layer, ACKAN):
                    lfp_found_ackan = True
                    break
        if lfp_found_ackan:
            break
    
    if lfp_found_ackan:
        print("SUCCESS: ACKAN found in LFPv3.")
    else:
        print("FAILURE: ACKAN NOT found in LFPv3.")

    # Check if ACKAN is present in GFP
    print("Checking GFP layers...")
    gfp_found_ackan = False
    for module in model.modules():
        if isinstance(module, GFPv3):
            # Check possible locations
            # In GFPv3, kan is an attribute on self if present
            if hasattr(module, 'kan') and isinstance(module.kan, ACKAN):
                gfp_found_ackan = True
                break
    
    if not gfp_found_ackan:
        print("SUCCESS: ACKAN NOT found in GFPv3.")
    else:
        print("FAILURE: ACKAN found in GFPv3 (Should be disabled).")

    # Forward pass
    x = torch.randn(1, 3, 224, 224)
    print("Running MedViTv3 forward pass...")
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
        print("MedViTv3 forward pass successful.")
    except Exception as e:
        print(f"MedViTv3 forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ackan()
    test_medvitv3_integration()
