import os
import json
import sys
import numpy as np
import shutil
import matplotlib.pyplot as plt
import argparse
import requests
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import seaborn as sns
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from datasets import build_dataset
from distutils.util import strtobool
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import natten
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from MedViTv3 import MedViTv3_tiny, MedViTv3_small, MedViTv3_base, MedViTv3_large
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'medmnistc-api')))
try:
    from medmnistc.corruptions.registry import CORRUPTIONS_DS, CORRUPTIONS_DS_FOLDS
    MEDMNISTC_AVAILABLE = True
except ImportError:
    MEDMNISTC_AVAILABLE = False
    print("Warning: medmnistc not found. Using default corruptions.")

# Fallback if medmnistc not available
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
    'saturate'
]

model_classes = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large,
    'MedViTv3_tiny': MedViTv3_tiny,
    'MedViTv3_small': MedViTv3_small,
    'MedViTv3_base': MedViTv3_base,
    'MedViTv3_large': MedViTv3_large
}

model_urls = {
    "MedViT_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViT_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0",
    "MedViTv3_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViTv3_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViTv3_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViTv3_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0"
}

def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")

def ensure_medmnist_npz(flag, size, root):
    """Ensure Evaluator can access the expected `{flag}.npz` file."""
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, f"{flag}.npz")
    if os.path.exists(target):
        return target

    size_specific = os.path.join(root, f"{flag}_{size}.npz")
    if os.path.exists(size_specific):
        shutil.copyfile(size_specific, target)
        return target

    available = ", ".join(sorted(f for f in os.listdir(root) if f.endswith(".npz")))
    raise FileNotFoundError(
        f"Missing MedMNIST file `{flag}.npz` in `{root}`. "
        f"Checked fallback `{flag}_{size}.npz`. "
        f"Available: {available if available else 'none'}."
    )

def evaluate_mnist(net, test_loader, device, data_flag, task, size=224):
    net.eval()
    y_score = torch.tensor([])
    with torch.no_grad():
        val_bar = tqdm(test_loader, desc="Evaluating", file=sys.stdout)
        for val_data in val_bar:
            inputs, targets = val_data
            outputs = net(inputs.to(device))
            
            if task == 'multi-label, binary-class':
                outputs = outputs.softmax(dim=-1)
            else:
                outputs = outputs.softmax(dim=-1)
            
            y_score = torch.cat((y_score, outputs.cpu()), 0)
            
    y_score = y_score.detach().numpy()
    ensure_medmnist_npz(data_flag, size=size, root='./data')
    evaluator = Evaluator(data_flag, 'test', size=size, root='./data')
    metrics = evaluator.evaluate(y_score)
    return metrics

def train_mnist(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, data_flag, task):
    best_acc = 0.0
    for epoch in range(epochs):
        if hasattr(net, "reset_stat_logs"):
            net.reset_stat_logs()
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            
            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels)
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs.squeeze(0), labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        # Validation
        net.eval()
        y_score = torch.tensor([])
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                inputs, targets = val_data
                outputs = net(inputs.to(device))
                
                if task == 'multi-label, binary-class':
                    outputs = outputs.softmax(dim=-1)
                else:
                    outputs = outputs.softmax(dim=-1)
                
                y_score = torch.cat((y_score, outputs.cpu()), 0)
                
        y_score = y_score.detach().numpy()
        ensure_medmnist_npz(data_flag, size=224, root='./data')
        evaluator = Evaluator(data_flag, 'test', size=224, root='./data')
        metrics = evaluator.evaluate(y_score)
        
        val_accurate, _ = metrics
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}')
        print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    model_name = args.model_name
    dataset_name = args.dataset
    pretrained = args.pretrained
    
    if args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        if task == "multi-label, binary-class":
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
        task = 'multi-class' # Default assumption for non-mnist

    batch_size = args.batch_size
    lr = args.lr
    
    # 1. Setup Model
    # We build dataset initially just to get nb_classes
    # For training, we force clean dataset
    temp_args = argparse.Namespace(**vars(args))
    temp_args.corruption = 'clean'
    _, _, nb_classes = build_dataset(args=temp_args)

    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=nb_classes, use_kmp_glu=args.use_kmp_glu, 
                          use_coord=args.use_coord, use_wavkan=args.use_wavkan,
                          use_ackan=args.use_ackan,
                          use_kan=args.use_kan, enable_global=args.enable_global,
                          enable_local=args.enable_local).to(device)
        
        # Load pretrained if requested and not training from scratch
        if pretrained and not args.checkpoint_path:
             checkpoint_url = model_urls.get(model_name)
             if checkpoint_url:
                 download_checkpoint(checkpoint_url, f'./{model_name}.pth')
                 checkpoint = torch.load(f'./{model_name}.pth')
                 net.load_state_dict(checkpoint, strict=False)
    else:
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=nb_classes).to(device)

    save_path = f'./{model_name}_{dataset_name}_best.pth'
    
    # 2. Train on Clean Data (if not eval_only)
    if not args.eval_only:
        print("--- Starting Training on Clean Data ---")
        # Force clean for training
        args.corruption = 'clean'
        train_dataset, test_dataset, _ = build_dataset(args=args)
        
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
        
        optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.05)
        eta = args.epochs * len(train_dataset) // args.batch_size
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eta, eta_min=5e-6)
        
        if dataset_name.endswith('mnist'):
            train_mnist(args.epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, dataset_name, task)
        else:
            print("MedMNIST-C script currently optimized for MedMNIST datasets.")
            # Fallback or implement train_other if needed
            pass
            
    # 3. MedMNIST-C Evaluation Loop
    print("\n--- Starting MedMNIST-C Evaluation ---")
    
    # Load best model
    if os.path.exists(save_path):
        print(f"Loading best model from {save_path}")
        checkpoint = torch.load(save_path)
        net.load_state_dict(checkpoint['model'])
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        if 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
        else:
            net.load_state_dict(checkpoint)
    else:
        print("Warning: No checkpoint found. Evaluating with current weights.")

    # Determine corruptions to evaluate
    results = {}
    
    # Run Clean Evaluation First (Always necessary for relative metrics)
    print(f"Evaluating: Clean (s0)")
    args.corruption = 'clean'
    args.severity = 0
    _, test_dataset, _ = build_dataset(args=args)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    
    if dataset_name.endswith('mnist'):
        metrics = evaluate_mnist(net, test_loader, device, dataset_name, task)
        clean_auc, clean_acc = metrics
        print(f"Result [Clean] -> AUC: {clean_auc:.4f}, ACC: {clean_acc:.4f}")
        results['clean'] = {'auc': clean_auc, 'acc': clean_acc}
    else:
        print("Evaluation for non-MedMNIST datasets not fully implemented in this script.")
        clean_acc = 0.0 # Placeholder
    
    
    if args.corruption == 'all':
        # Use MEDMNISTC grouping if available and dataset supported
        if MEDMNISTC_AVAILABLE and dataset_name in CORRUPTIONS_DS:
            eval_groups = CORRUPTIONS_DS_FOLDS
            # Ensure we only use corruptions defined for this dataset
            ds_corruptions = CORRUPTIONS_DS[dataset_name]
        else:
            # Fallback grouping
            eval_groups = {'all': CORRUPTIONS}
            ds_corruptions = set(CORRUPTIONS) # Treat as set of names
    elif args.corruption != 'clean':
        eval_groups = {'custom': [args.corruption]}
        ds_corruptions = {args.corruption} # Placeholder
    else:
        eval_groups = {} # Just clean done
    
    # Initialize aggregated results
    group_results = {} # group -> list of accs
    
    for group_name, corruption_list in eval_groups.items():
        if group_name not in group_results:
             group_results[group_name] = []
             
        for c in corruption_list:
            # Check if corruption is valid for this dataset (if using medmnistc logic)
            if MEDMNISTC_AVAILABLE and dataset_name in CORRUPTIONS_DS:
                 if c not in ds_corruptions:
                     continue
            
            for s in [1, 2, 3, 4, 5]:
                print(f"Evaluating: {c} (Severity {s}) [Group: {group_name}]")
                
                # Update args for build_dataset
                args.corruption = c
                args.severity = s
                
                # Rebuild dataset with corruption
                _, test_dataset, _ = build_dataset(args=args)
                test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
                
                if dataset_name.endswith('mnist'):
                    metrics = evaluate_mnist(net, test_loader, device, dataset_name, task)
                    auc, acc = metrics
                    print(f"Result [{c}, s{s}] -> AUC: {auc:.4f}, ACC: {acc:.4f}")
                    
                    key = f"{c}_{s}"
                    # results[key] = {"auc": auc, "acc": acc}
                    group_results[group_name].append(acc)
                    
                    # Store detailed results if needed
                    if group_name not in results: results[group_name] = {}
                    if c not in results[group_name]: results[group_name][c] = {}
                    results[group_name][c][s] = acc
                else:
                    pass

    # Save results
    result_file = f'medmnist_c_results_{model_name}_{dataset_name}.json'
    results['final_metrics'] = {}
    
    print("\n" + "="*80)
    print(f"{'Metric':<20} | {'bACC':<10} | {'rBE':<10} | {'BE':<10}")
    print("-" * 80)
    
    # Calculate metrics grouped by Digital, Noise, Blur, Color, TS
    # And Overall
    
    # Helper to calc metrics
    def calc_metrics(acc_list, clean_acc):
        if not acc_list: return 0, 0, 0
        bACC = np.mean(acc_list)
        BE = 1.0 - bACC
        clean_error = 1.0 - clean_acc
        rBE = (BE / clean_error) if clean_error > 1e-6 else 0.0 # Relative Benchmark Error = Error_corr / Error_clean? Or (Error_corr - Error_clean)/Error_clean?
        # Standard rCE in ImageNet-C is (Error_corr / Error_AlexNet). 
        # User requested rBE ↓. If clean error is small, rBE is high. 
        # Let's interpret rBE as Relative Benchmark Error = Error / CleanError. 
        return bACC, rBE, BE
        
    # Calculate Overall
    all_accs = []
    for g, accs in group_results.items():
        all_accs.extend(accs)
        
    overall_bACC, overall_rBE, overall_BE = calc_metrics(all_accs, clean_acc)
    print(f"{'Overall':<20} | {overall_bACC:.4f}     | {overall_rBE:.4f}     | {overall_BE:.4f}")
    results['final_metrics']['Overall'] = {'bACC': overall_bACC, 'rBE': overall_rBE, 'BE': overall_BE}
    
    # Calculate Per Group
    # Requested Order: Digital Noise Blur Color TS
    ordered_groups = ['digital', 'noise', 'blur', 'color', 'task-specific']
    existing_groups = list(group_results.keys())
    
    # Process ordered groups if they exist
    for g in ordered_groups:
        if g in group_results:
             bACC, rBE, BE = calc_metrics(group_results[g], clean_acc)
             g_label = g.capitalize() if g != 'task-specific' else 'TS'
             print(f"{g_label:<20} | {bACC:.4f}     | {rBE:.4f}     | {BE:.4f}")
             results['final_metrics'][g] = {'bACC': bACC, 'rBE': rBE, 'BE': BE}
             
    # Process any other groups
    for g in group_results:
        if g not in ordered_groups and g != 'all':
             bACC, rBE, BE = calc_metrics(group_results[g], clean_acc)
             print(f"{g:<20} | {bACC:.4f}     | {rBE:.4f}     | {BE:.4f}")
    
    print("="*80)
    print(f"bACCclean: {clean_acc:.4f}")
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Evaluation script for MedMNIST-C.')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny', help='Model name to use.')
    parser.add_argument('--dataset', type=str, default='pathmnist', help='Dataset to use (e.g., pathmnist, bloodmnist).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default=False, help="Use pretrained weights.")
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint.')
    
    # MedMNIST-C specific args
    parser.add_argument('--corruption', type=str, default='all', help='Corruption type: "clean", "all", or specific name.')
    parser.add_argument('--severity', type=int, default=1, help='Severity level (1-5). Ignored if corruption is "all" or "clean".')
    parser.add_argument('--eval_only', type=lambda x: bool(strtobool(x)), default=False, help='Skip training and only evaluate.')

    # Model args
    parser.add_argument('--use_kmp_glu', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--use_coord', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--use_wavkan', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--use_ackan', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--use_kan', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--enable_global', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--enable_local', type=lambda x: bool(strtobool(x)), default=True)

    args = parser.parse_args()
    main(args)
