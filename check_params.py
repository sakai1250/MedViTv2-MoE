import torch
import torch.nn as nn
import time
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from MedViTv3 import MedViTv3_tiny, MedViTv3_small, MedViTv3_base, MedViTv3_large

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops(model, input_shape=(1, 3, 224, 224)):
    if not HAS_FVCORE:
        return "N/A"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    inputs = torch.randn(input_shape).to(device)
    
    try:
        flops = FlopCountAnalysis(model, inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total()
    except Exception as e:
        return None

def measure_fps(model, input_shape=(1, 3, 224, 224), warmup=20, runs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    inputs = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = runs / elapsed_time
    return fps

def main():
    print(f"{'Model':<25} | {'Params':<12} | {'Size (M)':<10} | {'FLOPs (G)':<10} | {'FPS':<10}")
    print("-" * 85)

    models_to_check = [
        ('MedViT_tiny', MedViT_tiny),
        ('MedViT_small', MedViT_small),
        ('MedViT_base', MedViT_base),
        ('MedViT_large', MedViT_large),
        ('MedViTv3_tiny', MedViTv3_tiny),
        ('MedViTv3_small', MedViTv3_small),
        ('MedViTv3_base', MedViTv3_base),
        ('MedViTv3_large', MedViTv3_large),
    ]

    for name, model_fn in models_to_check:
        try:
            # Instantiate with default args (usually MLP or FasterKAN depending on defaults)
            model = model_fn(num_classes=1000)
            model.eval()
            params = count_parameters(model)
            flops = get_flops(model)
            fps = measure_fps(model)
            flops_str = f"{flops/1e9:.2f} G" if isinstance(flops, (int, float)) else str(flops)
            print(f"{name:<25} | {params:<12,} | {params/1e6:.2f} M   | {flops_str:<10} | {fps:.2f}")
        except Exception as e:
            print(f"{name:<25} | Error: {e}")

    print("-" * 85)
    print("Ablation Study (MedViTv3_tiny):")
    
    # WavKAN
    try:
        model = MedViTv3_tiny(num_classes=1000, use_wavkan=True, use_kan=True)
        model.eval()
        params = count_parameters(model)
        flops = get_flops(model)
        fps = measure_fps(model)
        flops_str = f"{flops/1e9:.2f} G" if isinstance(flops, (int, float)) else str(flops)
        print(f"{'  With WavKAN':<25} | {params:<12,} | {params/1e6:.2f} M   | {flops_str:<10} | {fps:.2f}")
    except Exception as e: 
        print(f"  With WavKAN Error: {e}")

    # MLP (No KAN)
    try:
        model = MedViTv3_tiny(num_classes=1000, use_kan=False)
        model.eval()
        params = count_parameters(model)
        flops = get_flops(model)
        fps = measure_fps(model)
        flops_str = f"{flops/1e9:.2f} G" if isinstance(flops, (int, float)) else str(flops)
        print(f"{'  With MLP (No KAN)':<25} | {params:<12,} | {params/1e6:.2f} M   | {flops_str:<10} | {fps:.2f}")
    except Exception as e: 
        print(f"  With MLP Error: {e}")

if __name__ == '__main__':
    main()