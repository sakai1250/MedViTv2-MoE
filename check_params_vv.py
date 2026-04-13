
import torch
from fvcore.nn import FlopCountAnalysis
from MedViT import MedViT_tiny

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_stats(model, name):
    print(f"Analyzing {name}...")
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
    
    params = count_parameters(model)
    try:
        flops = FlopCountAnalysis(model, input_tensor).total()
        flops_str = f"{flops/1e9:.2f}G"
    except Exception as e:
        flops_str = f"Error: {e}"
        
    print(f"RESULT: {name} | Params: {params/1e6:.2f}M | FLOPs: {flops_str}")

print("--- Starting Analysis ---")

# Original MedViT_tiny
try:
    print("\nInstantiating Original MedViT_tiny...")
    model_orig = MedViT_tiny(use_orkan=False)
    get_stats(model_orig, "MedViT_tiny (Original)")
except Exception as e:
    print(f"Failed to check Original: {e}")

# MedViTVV_tiny (ORKAN)
try:
    print("\nInstantiating MedViTVV_tiny (ORKAN)...")
    # Note: MedViT_tiny accepts kwargs which are passed to MedViT init
    model_orkan = MedViT_tiny(use_orkan=True)
    get_stats(model_orkan, "MedViTVV_tiny (ORKAN)")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed to check ORKAN: {e}")
