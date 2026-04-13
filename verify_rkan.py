import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from rkan import RationalKANLayer, RKAN

def test_rkan_layer():
    print("Testing RationalKANLayer...")
    batch_size = 4
    input_dim = 8
    output_dim = 16

    layer = RationalKANLayer(input_dim, output_dim, degree_p=5, degree_q=4)
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    y = layer(x)
    print(f"Forward output shape: {y.shape}")
    assert y.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {y.shape}"

    # Backward pass
    loss = y.sum()
    loss.backward()
    print("Backward pass successful.")

    # Check gradients
    assert layer.numerator_coeffs.grad is not None, "numerator_coeffs grad is None"
    assert layer.denominator_coeffs.grad is not None, "denominator_coeffs grad is None"
    assert layer.w.grad is not None, "w grad is None"
    print("Gradients computed successfully.")

def test_rkan_model():
    print("\nTesting RKAN model container...")
    batch_size = 4
    layers_hidden = [8, 16, 32]

    model = RKAN(layers_hidden)
    x = torch.randn(batch_size, layers_hidden[0])

    # Forward pass
    y = model(x)
    print(f"Model output shape: {y.shape}")
    assert y.shape == (batch_size, layers_hidden[-1]), f"Expected {(batch_size, layers_hidden[-1])}, got {y.shape}"

    # Backward pass
    loss = y.sum()
    loss.backward()
    print("Model backward pass successful.")

def test_rkan_param_count():
    print("\nTesting RKAN parameter count...")
    model = RKAN([64, 192, 64], degree_p=5, degree_q=4)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    assert trainable_params > 0, "No trainable parameters found"
    assert trainable_params == total_params, "Some parameters are not trainable"

if __name__ == "__main__":
    try:
        test_rkan_layer()
        test_rkan_model()
        test_rkan_param_count()
        print("\nAll verifications passed!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
