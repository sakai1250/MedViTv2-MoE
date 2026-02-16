
import torch
import copy
from MedViT_origin import MedViT_tiny, MedViT_small, MedViT_base
from fvcore.nn import FlopCountAnalysis

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops(model, input_shape=(1, 3, 224, 224)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    inputs = torch.randn(input_shape).to(device)
    try:
        flops = FlopCountAnalysis(model, inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total()
    except Exception as e:
        return f"Error: {e}"

def main():
    print("MedViT_origin Sensitivity Analysis")
    print("==================================")
    
    # Define configurations
    # Baseline sr_ratios for Tiny/Small/Base are usually [8, 4, 2, 1]
    sr_ratio_configs = {
        "Baseline": [8, 4, 2, 1],
        "Reduced_Half": [4, 2, 1, 1],
        "No_Reduction": [1, 1, 1, 1]
    }
    
    base_models = {
        "MedViT_tiny": MedViT_tiny,
        #"MedViT_small": MedViT_small
    }
    
    results = []

    print(f"{'Model':<15} | {'KAN':<6} | {'SR Ratios':<15} | {'Params (M)':<10} | {'GFLOPs':<10}")
    print("-" * 70)

    for model_name, model_fn in base_models.items():
        for kan_config in [True, False]:
            for sr_name, sr_ratios in sr_ratio_configs.items():
                try:
                    # Instantiate model
                    model = model_fn(num_classes=1000, sr_ratios=sr_ratios, use_kan=kan_config)
                    model.eval()
                    
                    params = count_parameters(model)
                    flops = get_flops(model)
                    
                    flops_str = f"{flops/1e9:.2f}" if isinstance(flops, (int, float)) else "N/A"
                    
                    print(f"{model_name:<15} | {str(kan_config):<6} | {str(sr_ratios):<15} | {params/1e6:<10.2f} | {flops_str:<10}")
                    
                    results.append({
                        "model": model_name,
                        "use_kan": kan_config,
                        "sr_ratios": sr_ratios,
                        "params": params,
                        "flops": flops
                    })
                    
                except Exception as e:
                    print(f"{model_name} | {kan_config} | {sr_ratios} | Error: {e}")

    print("\n")
    print("Generating Training Commands for Accuracy Sensitivity Analysis...")
    print("===============================================================")
    
    training_script = "run_sensitivity_training.sh"
    with open(training_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Dataset: using 'pneumoniamnist' as a quick example. Change as needed.\n")
        f.write("DATASET='pneumoniamnist'\n\n")
        
        for res in results:
            model_name = res["model"]
            use_kan = res["use_kan"]
            sr_ratios = str(res["sr_ratios"]).replace(" ", "") # Remove spaces for safer arg passing if needed, though ' "[...]" ' works
            
            cmd = f"python main_origin.py --model_name '{model_name}' --dataset $DATASET --pretrained True --use_kan {use_kan} --sr_ratios '{sr_ratios}'"
            
            description = f"# {model_name}, KAN={use_kan}, SR={sr_ratios}"
            print(description)
            print(cmd)
            f.write(f"{description}\n")
            f.write(f"echo \"Running: {description}\"\n")
            f.write(f"{cmd}\n\n")
            
    print(f"\nTraining commands saved to {training_script}")

if __name__ == "__main__":
    main()
