import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from orkan import OrthogonalRationalKANLayer, ORKAN

def test_orkan_layer():
    print("Testing OrthogonalRationalKANLayer...")
    batch_size = 4
    input_dim = 8
    output_dim = 16
    
    layer = OrthogonalRationalKANLayer(input_dim, output_dim, degree_m=5, degree_n=4)
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
    assert layer.alpha_coeffs.grad is not None, "alpha_coeffs grad is None"
    assert layer.beta_coeffs.grad is not None, "beta_coeffs grad is None"
    assert layer.w.grad is not None, "w grad is None"
    print("Gradients computed successfully.")

def test_orkan_model():
    print("\nTesting ORKAN model container...")
    batch_size = 4
    layers_hidden = [8, 16, 32]
    
    model = ORKAN(layers_hidden)
    x = torch.randn(batch_size, layers_hidden[0])
    
    # Forward pass
    y = model(x)
    print(f"Model output shape: {y.shape}")
    assert y.shape == (batch_size, layers_hidden[-1]), f"Expected {(batch_size, layers_hidden[-1])}, got {y.shape}"
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    print("Model backward pass successful.")

if __name__ == "__main__":
    try:
        test_orkan_layer()
        test_orkan_model()
        print("\nAll verifications passed!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        exit(1)
