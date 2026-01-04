import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from distutils.util import strtobool

# Import MedViTv3 models
from MedViTv3 import MedViTv3_tiny, MedViTv3_small, MedViTv3_base, MedViTv3_large

class GradCAM:
    """
    Simple implementation of Grad-CAM.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        # forward hook to capture activations
        self.target_layer.register_forward_hook(self.save_activation)
        # backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, we take the first element
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward pass
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            # If no class index is specified, use the predicted class
            class_idx = output.argmax(dim=1).item()
            
        # 2. Backward pass
        self.model.zero_grad()
        
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        # Compute gradients
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 3. Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients (weights)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU (we are only interested in features that have a positive influence)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.detach().cpu().numpy()[0, 0]
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-7)
        
        return cam, class_idx, output

def show_cam_on_image(img, mask):
    """
    Overlay the heatmap on the image.
    img: numpy array [H, W, 3], range [0, 1]
    mask: numpy array [H, W], range [0, 1]
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Convert RGB to BGR for OpenCV if needed, but here we assume input img is RGB
    # heatmap is BGR from applyColorMap, so we convert it to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading model: {args.model_name}")
    model_classes = {
        'MedViTv3_tiny': MedViTv3_tiny,
        'MedViTv3_small': MedViTv3_small,
        'MedViTv3_base': MedViTv3_base,
        'MedViTv3_large': MedViTv3_large
    }
    
    if args.model_name not in model_classes:
        raise ValueError(f"Unknown model name: {args.model_name}")
        
    model = model_classes[args.model_name](
        num_classes=args.num_classes,
        use_wavkan=args.use_wavkan,
        use_kan=args.use_kan,
        use_kmp_glu=args.use_kmp_glu
    ).to(device)

    # Load checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # Handle state dict keys if necessary (e.g., 'model' key or direct state_dict)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No checkpoint found or provided. Using random initialization.")

    # 2. Prepare Grad-CAM
    # Target the last normalization layer before global pooling
    target_layer = model.norm
    grad_cam = GradCAM(model, target_layer)

    # 3. Load and Preprocess Image
    image_path = args.image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    raw_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 4. Run Grad-CAM
    print("Generating Grad-CAM...")
    mask, class_idx, output = grad_cam(input_tensor, class_idx=args.target_class)
    
    print(f"Predicted class: {output.argmax(dim=1).item()}")
    print(f"Visualizing class: {class_idx}")

    # 5. Visualization
    # Resize raw image to 224x224 for overlay
    img_resized = np.array(raw_image.resize((224, 224))) / 255.0
    mask_resized = cv2.resize(mask, (224, 224))
    
    vis_image = show_cam_on_image(img_resized, mask_resized)
    
    save_path = "gradcam_result.png"
    Image.fromarray(vis_image).save(save_path)
    print(f"Saved Grad-CAM result to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='MedViTv3_tiny')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to .pth file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--target_class', type=int, default=None, help='Class index to visualize (default: predicted class)')
    
    # Model architecture args
    parser.add_argument('--use_wavkan', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--use_kan', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--use_kmp_glu', type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    main(args)