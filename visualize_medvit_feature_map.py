
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import medmnist
from torchvision import transforms
from MedViT import MedViT_tiny

def load_model(weight_path, num_classes):
    # Instantiate MedViT_tiny
    model = MedViT_tiny(
        num_classes=num_classes,
        use_kmp_glu=True # Assuming standard config based on typical usage, adjust if needed
    )
    
    # Load weights
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Handle state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix
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
    output_dir = "Fig/FeatureMaps_def"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Visualizing feature maps for {dataset_name} using weights {weight_path}")
    
    # Get dataset info
    info = medmnist.INFO[dataset_name]
    num_classes = len(info['label'])
    
    # Load model
    model = load_model(weight_path, num_classes)
    
    # Hook to capture features
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    hooks = []
    
    # Find LFP blocks
    lfp_indices = []
    for i, layer in enumerate(model.features):
        # In MedViT_tiny, LFP is the class name in MedViT.py
        if layer.__class__.__name__ == 'LFP':
            lfp_indices.append(i)
            
    print(f"Found LFP blocks at indices: {lfp_indices}")
    
    # Hook first and middle LFP blocks
    target_layers = [lfp_indices[0], lfp_indices[len(lfp_indices)//2]] 
    
    for layer_idx in target_layers:
        # Access the Depth-wise Convolution in LocalityFeedForward
        # Structure: layer(LFP) -> conv(LocalityFeedForward) -> conv(Sequential) -> Index 3 (Conv2d 3x3 depthwise)
        # Note: LocalityFeedForward structure:
        # 0: Conv 1x1
        # 1: BN
        # 2: Act
        # 3: Conv 3x3 dw (if not wo_dp_conv and not dp_first) - check checks.
        
        lff = model.features[layer_idx].conv
        lff_seq = lff.conv
        
        # Verify if index 3 is Conv2d with groups > 1 (depthwise)
        dw_conv_idx = -1
        for i, m in enumerate(lff_seq):
             # Check for depthwise convolution: groups == in_channels
             if isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.groups > 1:
                 dw_conv_idx = i
                 break
        
        if dw_conv_idx != -1:
            # The activation is typically 2 layers after conv (Conv -> BN -> Act)
            # But let's find the next activation layer dynamically
            act_idx = -1
            for i in range(dw_conv_idx + 1, len(lff_seq)):
                m = lff_seq[i]
                if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish)) or 'swish' in m.__class__.__name__.lower():
                    act_idx = i
                    break
            
            if act_idx != -1:
                layer_name = f"Stage_{layer_idx}_Conv_Act"
                h = lff_seq[act_idx].register_forward_hook(get_activation(layer_name))
                hooks.append(h)
                print(f"Hooked Activation at index {act_idx} (after Conv at {dw_conv_idx})")
            else:
                # Fallback to conv if no activation found (should not happen in standard blocks)
                print(f"Warning: No activation found after depth-wise conv at {dw_conv_idx}, hooking conv instead")
                layer_name = f"Stage_{layer_idx}_Conv"
                h = lff_seq[dw_conv_idx].register_forward_hook(get_activation(layer_name))
                hooks.append(h)
                
        else:
            print(f"Warning: No Depth-wise Conv layer found in LFP block {layer_idx}")

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
                print(f"[MedViT] Layer: {name}, Resolution: {h}x{w}")

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
            plt.title(f"{name} Mean Output\n(ReLU Output, Scaled 0-1)")
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
    parser.add_argument('--dataset', type=str, default='breastmnist', help='Dataset name')
    parser.add_argument('--weight_path', type=str, default='MedViT_tiny_breastmnist.pth', help='Path to weight file')
    args = parser.parse_args()
    
    visualize_feature_maps(args)
