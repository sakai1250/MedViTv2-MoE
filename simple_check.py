
import sys
print("Script starting...")
sys.stdout.flush()

import torch
print("Imported torch")
sys.stdout.flush()

from MedViT import MedViT_tiny
print("Imported MedViT")
sys.stdout.flush()

from medvitvv import MedViTVV_tiny, MedViTVV_small, MedViTVV_base
print("Imported MedViTVV classes")
sys.stdout.flush()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Defining models...")
models = [
    ('MedViT_tiny', MedViT_tiny),
    ('MedViTVV_tiny', MedViTVV_tiny),
    ('MedViTVV_small', MedViTVV_small),
    ('MedViTVV_base', MedViTVV_base),
]

for name, cls in models:
    print(f"Checking {name}...")
    sys.stdout.flush()
    try:
        model = cls(num_classes=1000)
        params = count_parameters(model)
        
        # Calculate FLOPs using fvcore
        # Move to CPU for FLOPs to avoid GPU OOM/hangs if possible, or GPU if needed.
        # fvcore works on CPU too for FLOPs.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        flops_str = "N/A"
        try:
             from fvcore.nn import FlopCountAnalysis
             flops = FlopCountAnalysis(model, dummy_input)
             flops.unsupported_ops_warnings(False)
             flops.uncalled_modules_warnings(False)
             flops_val = flops.total()
             flops_str = f"{flops_val/1e9:.2f} G"
        except ImportError:
             flops_str = "fvcore not installed"
        except Exception as e:
             flops_str = f"Error: {e}"

        print(f"{name}: {params:<12} params ({params/1e6:.2f} M) | FLOPs: {flops_str}")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error checking {name}: {e}")
        sys.stdout.flush()
