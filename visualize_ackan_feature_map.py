
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import medmnist
from torchvision import transforms
from MedViTv3 import MedViTv3_tiny

def load_model(weight_path, num_classes, use_ackan=True, use_kmp_glu=True):
    # Instantiate the model with the same configuration as training
    # Based on logs, breastmnist training used use_kmp_glu=True and ACKAN
    model = MedViTv3_tiny(
        num_classes=num_classes,
        use_ackan=use_ackan,
        use_kmp_glu=use_kmp_glu,
        use_wavkan=False # Assuming WavKAN was not used designated by 'ACKAN' name in logs usually implies ACKAN only or overriding
    )
    
    # Load weights
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Handle state dict (sometimes it's in 'state_dict' key, sometimes direct)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Load state dict
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load status: {msg}")
    
    model.eval()
    return model

def get_dataset(dataset_name):
    # Map dataset name to MedMNIST class
    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Preprocessing
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Load test dataset
    dataset = DataClass(split='test', transform=data_transform, download=True, size=224, mmap_mode='r', as_rgb=True)
    return dataset

def visualize_feature_maps(args):
    dataset_name = args.dataset
    weight_path = args.weight_path
    output_dir = "Fig/FeatureMaps"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Visualizing feature maps for {dataset_name} using weights {weight_path}")
    
    # Get dataset info
    info = medmnist.INFO[dataset_name]
    num_classes = len(info['label'])
    
    # Load model
    model = load_model(weight_path, num_classes, use_ackan=True, use_kmp_glu=True)
    
    # Hook to capture features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks on ACKAN layers
    hooks = []
    
    # Find LFP blocks
    lfp_indices = []
    for i, layer in enumerate(model.features):
        if hasattr(layer, 'ffn') and layer.ffn.__class__.__name__ == 'LocalityFeedForwardV3':
            lfp_indices.append(i)
            
    print(f"Found LFP blocks at indices: {lfp_indices}")
    
    # Hook first and middle LFP blocks
    target_layers = [lfp_indices[0], lfp_indices[len(lfp_indices)//2]] 
    
    for layer_idx in target_layers:
        ffn_net = model.features[layer_idx].ffn.net
        ackan_layer_idx = -1
        for i, m in enumerate(ffn_net):
            if 'ACKAN' in m.__class__.__name__:
                ackan_layer_idx = i
                break
        
        if ackan_layer_idx != -1:
            layer_name = f"Stage_{layer_idx}_ACKAN"
            h = ffn_net[ackan_layer_idx].register_forward_hook(get_activation(layer_name))
            hooks.append(h)
            print(f"Hooked {layer_name}")
        else:
            print(f"Warning: No ACKAN layer found in LFP block {layer_idx}")

    # Load dataset
    dataset = get_dataset(dataset_name)
    
    # Loop over 5 samples
    for sample_idx in range(5):
        print(f"Processing sample {sample_idx}...")
        image, label = dataset[sample_idx]
        image_input = image.unsqueeze(0)
    
        # Forward pass
        output = model(image_input)
        
        # Plotting
        for name, act in activation.items():
            if sample_idx == 0:
                h, w = act.shape[2], act.shape[3]
                print(f"[ACKAN] Layer: {name}, Resolution: {h}x{w}")

            # Create stage-specific directory
            stage_dir = os.path.join(output_dir, name)
            os.makedirs(stage_dir, exist_ok=True)
            
            # act: (1, C, H, W)
            act = act.cpu().squeeze(0) # (C, H, W)
            num_channels = act.shape[0]
            
            # Plot mean activation (heat map)
            mean_act = torch.mean(act, dim=0).numpy()
            
            # Min-Max Scaling to 0-1
            m_min, m_max = mean_act.min(), mean_act.max()
            if m_max > m_min:
                mean_act = (mean_act - m_min) / (m_max - m_min)
            else:
                mean_act = np.zeros_like(mean_act)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            # Show original image
            img_show = image.permute(1, 2, 0).numpy()
            img_show = (img_show * 0.5 + 0.5).clip(0, 1) # Un-normalize
            if img_show.shape[2] == 1:
                plt.imshow(img_show[:,:,0], cmap='gray')
            else:
                plt.imshow(img_show)
            plt.title(f"Sample {sample_idx} (Label: {label})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mean_act, cmap='viridis', vmin=0, vmax=1)
            plt.title(f"{name} Mean Output\n(Scaled 0-1)")
            plt.axis('off')
            plt.colorbar()
            
            save_path = os.path.join(stage_dir, f"{dataset_name}_sample{sample_idx}_mean.png")
            plt.savefig(save_path)
            print(f"Saved {save_path}")
            plt.close()
            
            # Plot top 16 channels
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f"{name} First 16 Channels - Sample {sample_idx} (Scaled 0-1)")
            for i in range(min(16, num_channels)):
                ax = axes[i//4, i%4]
                ch_img = act[i].numpy()
                
                # Scale each channel individually to 0-1 for visibility
                c_min, c_max = ch_img.min(), ch_img.max()
                if c_max > c_min:
                    ch_img = (ch_img - c_min) / (c_max - c_min)
                else:
                    ch_img = np.zeros_like(ch_img)
                    
                ax.imshow(ch_img, cmap='viridis', vmin=0, vmax=1)
                ax.axis('off')
            
            save_path_grid = os.path.join(stage_dir, f"{dataset_name}_sample{sample_idx}_grid.png")
            plt.savefig(save_path_grid)
            print(f"Saved {save_path_grid}")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breastmnist', help='Dataset name (breastmnist, retinamnist)')
    parser.add_argument('--weight_path', type=str, default='MedViTv3_tiny_breastmnist_best.pth', help='Path to weight file')
    args = parser.parse_args()
    
    visualize_feature_maps(args)
