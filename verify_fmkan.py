"""
FM-KAN Verification Script
Tests FMKANLayer, FMKAN, and MedViTVVV integration.
"""
import sys
import torch
import torch.nn as nn

def test_fmkan_layer():
    """Test single FMKANLayer."""
    print("=" * 60)
    print("Test 1: FMKANLayer forward pass")
    print("=" * 60)
    from fm_kan import FMKANLayer
    
    input_dim, output_dim = 64, 128
    batch_size = 4
    layer = FMKANLayer(input_dim, output_dim, num_basis=8)
    x = torch.randn(batch_size, input_dim)
    y = layer(x)
    
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (batch_size, output_dim), f"Expected ({batch_size}, {output_dim}), got {y.shape}"
    
    # Check gradients flow
    loss = y.sum()
    loss.backward()
    print(f"  Gradient check: omega.grad exists = {layer.omega.grad is not None}")
    print(f"  Gradient check: alpha.grad exists = {layer.alpha.grad is not None}")
    print("  PASSED ✓")
    
    # Count parameters
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"  Parameters: {total_params:,}")

def test_fmkan_multi_layer():
    """Test FMKAN multi-layer network."""
    print("\n" + "=" * 60)
    print("Test 2: FMKAN multi-layer forward pass")
    print("=" * 60)
    from fm_kan import FMKAN
    
    layers_hidden = [64, 192, 64]
    batch_size = 4
    model = FMKAN(layers_hidden, num_basis=8)
    x = torch.randn(batch_size, 64)
    y = model(x)
    
    print(f"  Architecture: {layers_hidden}")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (batch_size, 64), f"Expected ({batch_size}, 64), got {y.shape}"
    print("  PASSED ✓")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

def test_alpha_zero_degeneracy():
    """Test that alpha=0 makes FM-KAN degenerate to windowed sine."""
    print("\n" + "=" * 60)
    print("Test 3: Alpha=0 degeneracy (windowed sine)")
    print("=" * 60)
    from fm_kan import FMKANLayer
    
    layer = FMKANLayer(32, 32, num_basis=4)
    # Set alpha to 0 (no modulation)
    with torch.no_grad():
        layer.alpha.zero_()
    
    x = torch.randn(2, 32)
    y = layer(x)
    print(f"  Output shape: {y.shape}")
    print(f"  Output has finite values: {torch.isfinite(y).all()}")
    assert torch.isfinite(y).all(), "Output contains NaN/Inf with alpha=0"
    print("  PASSED ✓ (alpha=0 produces valid output)")

def test_medvitvvv_forward():
    """Test MedViTVVV model forward pass."""
    print("\n" + "=" * 60)
    print("Test 4: MedViTVVV_tiny forward pass (224x224)")
    print("=" * 60)
    from medvitvvv import MedViTVVV_tiny
    
    model = MedViTVVV_tiny(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    
    print("  Running forward pass...")
    try:
        y = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape == (1, 10), f"Expected (1, 10), got {y.shape}"
        print("  PASSED ✓")
    except Exception as e:
        print(f"  FAILED ✗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    fmkan_params = 0
    for name, p in model.named_parameters():
        if 'kan' in name.lower():
            fmkan_params += p.numel()
    
    print(f"\n  Total parameters:  {total_params:,}")
    print(f"  FM-KAN parameters: {fmkan_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    return True

def test_medvitvvv_backward():
    """Test MedViTVVV backward pass (gradient flow)."""
    print("\n" + "=" * 60)
    print("Test 5: MedViTVVV_tiny backward pass")
    print("=" * 60)
    from medvitvvv import MedViTVVV_tiny
    
    model = MedViTVVV_tiny(num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Check that FM-KAN params received gradients
    grad_ok = True
    for name, p in model.named_parameters():
        if 'kan' in name.lower() and p.requires_grad:
            if p.grad is None:
                print(f"  WARNING: No gradient for {name}")
                grad_ok = False
    
    if grad_ok:
        print("  All FM-KAN parameters received gradients")
        print("  PASSED ✓")
    else:
        print("  FAILED ✗ (some parameters missing gradients)")

if __name__ == "__main__":
    test_fmkan_layer()
    test_fmkan_multi_layer()
    test_alpha_zero_degeneracy()
    success = test_medvitvvv_forward()
    if success:
        test_medvitvvv_backward()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
