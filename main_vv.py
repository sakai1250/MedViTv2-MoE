
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
# import natten
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from MedViTv3 import MedViTv3_tiny, MedViTv3_small, MedViTv3_base, MedViTv3_large
# from medvitvv import MedViTVV_tiny, MedViTVV_small, MedViTVV_base, MedViTVV_large
from medvitvvv import MedViTVVV_tiny, MedViTVVV_small, MedViTVVV_base, MedViTVVV_large
from datetime import datetime



model_classes = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large,
    'MedViTv3_tiny': MedViTv3_tiny,
    'MedViTv3_small': MedViTv3_small,
    'MedViTv3_base': MedViTv3_base,
    'MedViTv3_large': MedViTv3_large,
    'MedViTVV_tiny': MedViT_tiny, # Alias to MedViT, distinguish via args
    'MedViTVV_small': MedViT_small,
    'MedViTVV_base': MedViT_base,
    'MedViTVV_large': MedViT_large,
    'MedViTVVV_tiny': MedViTVVV_tiny,
    'MedViTVVV_small': MedViTVVV_small,
    'MedViTVVV_base': MedViTVVV_base,
    'MedViTVVV_large': MedViTVVV_large
}

model_urls = {
    "MedViT_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViT_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0",
    "MedViTv3_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViTv3_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViTv3_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViTv3_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0",
    "MedViTVV_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0", 
    # Using MedViT_tiny checkpoint for MedViTVV_tiny as a base (strict=False will handle missing keys)
    "MedViTVV_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViTVV_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViTVV_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0",
    "MedViTVVV_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViTVVV_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViTVVV_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViTVVV_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0"
}

def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")

# Define the MNIST training routine
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
    # raise FileNotFoundError(
    #     f"Missing MedMNIST file `{flag}.npz` in `{root}`. "
    #     f"Checked fallback `{flag}_{size}.npz`. "
    #     f"Available: {available if available else 'none'}."
    # )
    print(f"Warning: Missing MedMNIST file `{flag}.npz` in `{root}`. Proceeding without specific file check (Evaluator might download).")


def train_mnist(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, data_flag, task, args):
    best_acc = 0.0
    alpha_history, gap_history, var_history = [], [], []
    gating_history = []
    
    # ... (training loop unchanged)
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

            # Pole regularization for RKAN
            if args.pole_reg_lambda > 0 and hasattr(net, 'pole_regularization'):
                pole_loss = net.pole_regularization()
                loss = loss + args.pole_reg_lambda * pole_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        epoch_stats = net.collect_stat_logs() if hasattr(net, "collect_stat_logs") else None
        record = {"epoch": epoch + 1}
        if epoch_stats:
            alpha_history.append(epoch_stats["alpha_mean"])
            gap_history.append(epoch_stats["gap_mean"])
            var_history.append(epoch_stats["var_mean"])
            record.update({
                "alpha_mean": epoch_stats["alpha_mean"],
                "gap_mean": epoch_stats["gap_mean"],
                "var_mean": epoch_stats["var_mean"],
                "count": epoch_stats.get("count"),
            })
            if "per_layer" in epoch_stats:
                record["per_layer"] = epoch_stats["per_layer"]
            print(f'  gating stats -> alpha: {epoch_stats["alpha_mean"]:.4f}, gap: {epoch_stats["gap_mean"]:.4f}, var: {epoch_stats["var_mean"]:.4f}')
        else:
            alpha_history.append(None)
            gap_history.append(None)
            var_history.append(None)
        gating_history.append(record)

        net.eval()
        y_score = torch.tensor([])
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                inputs, targets = val_data
                outputs = net(inputs.to(device))
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                
                y_score = torch.cat((y_score, outputs.cpu()), 0)
                
        y_score = y_score.detach().numpy()
        ensure_medmnist_npz(data_flag, size=224, root='./data')
        try:
            evaluator = Evaluator(data_flag, 'test', size=224, root='./data')
            metrics = evaluator.evaluate(y_score)
            
            val_accurate, _ = metrics
            print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}')
        except Exception as e:
            print(f"Evaluation error: {e}")
            val_accurate = 0
            metrics = [0, 0]

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
    
    # Determine filename
    if args.use_fmkan:
        out_filename = f'{data_flag}_fmkan.txt'
        kan_type = "FMKAN"
    elif args.use_drkan:
        out_filename = f'{data_flag}_drkan.txt'
        kan_type = "DRKAN"
    elif args.use_rkan:
        out_filename = f'{data_flag}_rkan.txt'
        kan_type = "RKAN"
    elif args.use_okan:
        out_filename = f'{data_flag}_okan.txt'
        kan_type = "OKAN"
    elif args.use_orkan:
        out_filename = f'{data_flag}_orkan.txt'
        kan_type = "ORKAN"
    elif args.use_ackan:
        out_filename = f'{data_flag}_ackan.txt'
        kan_type = "ACKAN"
    else:
        out_filename = f'{data_flag}.txt'
        kan_type = "KAN" if args.use_kan else "MLP"
        
    # Formatting
    header = f"{args.model_name} {kan_type} --enable_local {args.enable_local} --enable_global {args.enable_global}"
    
    with open(out_filename, 'a') as f:
        now = datetime.now().strftime("%Y/%m/%d %H:%M")
        f.write(f"{header}\n")
        f.write(f"{now}\n")
        f.write(f"testacc       {val_accurate * 100:.2f}%\n")
        f.write(f"auc            {metrics[0] * 100:.2f}%\n")
        f.write(f"acc      {metrics[1] * 100:.2f}%\n")
        f.write("\n")

    # 実行時刻をフォルダ名に
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    log_dir = f'log_files/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)

    if any(value is not None for value in alpha_history + gap_history + var_history):
        epochs_axis = np.arange(1, len(alpha_history) + 1)
        alpha_vals = np.array([np.nan if v is None else v for v in alpha_history], dtype=float)
        gap_vals = np.array([np.nan if v is None else v for v in gap_history], dtype=float)
        var_vals = np.array([np.nan if v is None else v for v in var_history], dtype=float)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_axis, alpha_vals, label='alpha')
        plt.plot(epochs_axis, gap_vals, label='gap')
        plt.plot(epochs_axis, var_vals, label='var')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Epoch-wise alpha / gap / var')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{log_dir}/alpha_var_gap.png')
        plt.close()

        with open(f'{log_dir}/alpha_var_gap.json', 'w') as stats_file:
            json.dump(gating_history, stats_file, indent=2)

    # 混同行列の可視化と保存
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.tight_layout()
    # plt.savefig(f'{log_dir}/conf_matrix.png')
    # plt.close()
    
# Define the non-MNIST training routine
def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))
    return specificity

def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()  # Sum of all diagonal elements (TP for all classes)
    total_sum = conf_matrix.sum()  # Sum of all elements in the matrix
    return tp_tn_sum / total_sum

def train_other(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, args):
    best_acc = 0.0
    alpha_history, gap_history, var_history = [], [], []
    gating_history = []
    
    # ... (training loop unchanged)
    for epoch in range(epochs):
        if hasattr(net, "reset_stat_logs"):
            net.reset_stat_logs()
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        # Training Loop
        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))

            # Pole regularization for RKAN
            if args.pole_reg_lambda > 0 and hasattr(net, 'pole_regularization'):
                pole_loss = net.pole_regularization()
                loss = loss + args.pole_reg_lambda * pole_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        epoch_stats = net.collect_stat_logs() if hasattr(net, "collect_stat_logs") else None
        record = {"epoch": epoch + 1}
        if epoch_stats:
            alpha_history.append(epoch_stats["alpha_mean"])
            gap_history.append(epoch_stats["gap_mean"])
            var_history.append(epoch_stats["var_mean"])
            record.update({
                "alpha_mean": epoch_stats["alpha_mean"],
                "gap_mean": epoch_stats["gap_mean"],
                "var_mean": epoch_stats["var_mean"],
                "count": epoch_stats.get("count"),
            })
            if "per_layer" in epoch_stats:
                record["per_layer"] = epoch_stats["per_layer"]
            print(f'  gating stats -> alpha: {epoch_stats["alpha_mean"]:.4f}, gap: {epoch_stats["gap_mean"]:.4f}, var: {epoch_stats["var_mean"]:.4f}')
        else:
            alpha_history.append(None)
            gap_history.append(None)
            var_history.append(None)
        gating_history.append(record)

        # Validation Loop
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Store raw probabilities/logits for AUC
        acc = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # Raw outputs (logits)
                probs = torch.softmax(outputs, dim=1)  # Convert to probabilities
                
                predict_y = torch.max(probs, dim=1)[1]  # Predicted class

                # Collect predictions, labels, and probabilities
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                # all_probs.extend(probs.cpu().numpy())

                # Calculate accuracy
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        # Calculate metrics
        val_accurate = acc / len(test_loader.dataset)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')  # Sensitivity
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Confusion Matrix for multi-class
        conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = specificity_per_class(conf_matrix)  # List of specificities per class
        avg_specificity = sum(specificity) / len(specificity)  # Average specificity

        # Overall Accuracy calculation
        overall_acc = overall_accuracy(conf_matrix)

        # One-hot encode the labels for AUC calculation
        n_classes = len(conf_matrix)
        all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))

        try:
            # Compute AUC for multi-class
            # auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
            auc = 0.0
        except ValueError:
            auc = float('nan')  # Handle edge case where AUC can't be computed

        # Print metrics
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} '
              f'val_accuracy: {val_accurate:.4f} precision: {precision:.4f} '
              f'recall: {recall:.4f} specificity: {avg_specificity:.4f} '
              f'f1_score: {f1:.4f} auc: {auc:.4f} overall_accuracy: {overall_acc:.4f}')
        
        print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        
        # Save best model
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
    
    # Determine filename
    data_flag = args.dataset # Or pass explicit if different
    if args.use_fmkan:
        out_filename = f'{data_flag}_fmkan.txt'
        kan_type = "FMKAN"
    elif args.use_drkan:
        out_filename = f'{data_flag}_drkan.txt'
        kan_type = "DRKAN"
    elif args.use_rkan:
        out_filename = f'{data_flag}_rkan.txt'
        kan_type = "RKAN"
    elif args.use_okan:
        out_filename = f'{data_flag}_okan.txt'
        kan_type = "OKAN"
    elif args.use_orkan:
        out_filename = f'{data_flag}_orkan.txt'
        kan_type = "ORKAN"
    elif args.use_ackan:
        out_filename = f'{data_flag}_ackan.txt'
        kan_type = "ACKAN"
    else:
        out_filename = f'{data_flag}.txt'
        kan_type = "KAN" if args.use_kan else "MLP"
        
    # Formatting
    header = f"{args.model_name} {kan_type} --enable_local {args.enable_local} --enable_global {args.enable_global}"
    
    # 追記モードでファイルオープン
    with open(out_filename, 'a') as f:
        now = datetime.now().strftime("%Y/%m/%d %H:%M")
        f.write(f"{header}\n")
        f.write(f"{now}\n")
        f.write(f"testacc       {val_accurate * 100:.2f}%\n")
        f.write(f"auc            {auc * 100:.2f}%\n")
        f.write(f"precision      {precision * 100:.2f}%\n")
        f.write(f"recall         {recall * 100:.2f}%\n")
        f.write(f"specificity    {avg_specificity * 100:.2f}%\n")
        f.write(f"f1_score       {f1 * 100:.2f}%\n")
        f.write(f"overall_acc    {overall_acc * 100:.2f}%\n")
        f.write("\n")

        # 実行時刻をフォルダ名に
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        log_dir = f'log_files/{timestamp}'
        os.makedirs(log_dir, exist_ok=True)

        if any(value is not None for value in alpha_history + gap_history + var_history):
            epochs_axis = np.arange(1, len(alpha_history) + 1)
            alpha_vals = np.array([np.nan if v is None else v for v in alpha_history], dtype=float)
            gap_vals = np.array([np.nan if v is None else v for v in gap_history], dtype=float)
            var_vals = np.array([np.nan if v is None else v for v in var_history], dtype=float)

            plt.figure(figsize=(8, 5))
            plt.plot(epochs_axis, alpha_vals, label='alpha')
            plt.plot(epochs_axis, gap_vals, label='gap')
            plt.plot(epochs_axis, var_vals, label='var')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Epoch-wise alpha / gap / var')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{log_dir}/alpha_var_gap.png')
            plt.close()

            with open(f'{log_dir}/alpha_var_gap.json', 'w') as stats_file:
                json.dump(gating_history, stats_file, indent=2)

        # 混同行列の可視化と保存
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        # plt.title('Confusion Matrix')
        # plt.xlabel('Predicted Label')
        # plt.ylabel('True Label')
        # plt.tight_layout()
        # plt.savefig(f'{log_dir}/conf_matrix.png')
        # plt.close()

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
    model_class = model_classes.get(model_name)

    # if not model_class:
    #     raise ValueError(f"Model {model_name} is not recognized. Available models: {list(model_classes.keys())}")

    batch_size = args.batch_size
    lr = args.lr
    
    train_dataset, test_dataset, nb_classes = build_dataset(args=args)
    val_num = len(test_dataset)
    train_num = len(train_dataset)
    
    # scheduler max iteration
    eta = args.epochs * train_num // args.batch_size

    # Select model
    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=nb_classes, use_kmp_glu=args.use_kmp_glu, 
                          use_coord=args.use_coord, use_wavkan=args.use_wavkan,
                          use_checkpoint=args.use_checkpoint,
                          use_ackan=args.use_ackan,
                          use_kan=args.use_kan, enable_global=args.enable_global,
                          enable_local=args.enable_local, 
                          use_orkan=args.use_orkan,
                          use_okan=args.use_okan,
                          use_rkan=args.use_rkan,
                          use_drkan=args.use_drkan).cuda()
        if pretrained:
            checkpoint_path = args.checkpoint_path
            if not os.path.exists(checkpoint_path):
                checkpoint_url = model_urls.get(model_name)
                if not checkpoint_url:
                    # fallback to medvit_tiny for vv_tiny if not found
                    if "MedViTVV" in model_name:
                         base_name = model_name.replace("MedViTVV", "MedViT")
                         checkpoint_url = model_urls.get(base_name)
                    
                    if not checkpoint_url:
                         raise ValueError(f"Checkpoint URL for model {model_name} not found.")
                download_checkpoint(checkpoint_url, f'./{model_name}.pth')
                checkpoint_path = f'./{model_name}.pth'

            checkpoint = torch.load(checkpoint_path)
            state_dict = net.state_dict()
            for k in ['proj_head.0.weight', 'proj_head.0.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            net.load_state_dict(checkpoint, strict=False)
    else:
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=nb_classes).cuda()

    
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eta, eta_min=5e-6)
    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    
    print(train_dataset)
    print("===================")
    print(test_dataset)

    epochs = args.epochs
    best_acc = 0.0
    save_path = f'./{model_name}_{dataset_name}.pth'
    train_steps = len(train_loader)

    if dataset_name.endswith('mnist'):
        
        train_mnist(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path, dataset_name, task, args)
    else:
        train_other(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for MedViT models.')
    parser.add_argument('--model_name', type=str, default='MedViTVV_tiny', help='Model name to use.')
    #tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, breastmnist, bloodmnist,
    #organamnist, organcmnist, organsmnist'
    parser.add_argument('--dataset', type=str, default='PAD', help='Dataset to use.')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default=False, help="Whether to use pretrained weights (True/False).")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/MedViT_tiny.pth', help='Path to the checkpoint file.')
    parser.add_argument('--use_kmp_glu', type=lambda x: bool(strtobool(x)), default=False, help='Enable KAN-MLP parallel GLU fusion in GFP blocks (default: False).')
    parser.add_argument('--use_coord', type=lambda x: bool(strtobool(x)), default=False, help='Enable Coordinate Attention in MHCA (default: False).')
    parser.add_argument('--use_wavkan', type=lambda x: bool(strtobool(x)), default=False, help='Use WavKAN instead of FasterKAN (default: False).')
    parser.add_argument('--use_ackan', type=lambda x: bool(strtobool(x)), default=False, help='Use AC-KAN (default: False).')
    parser.add_argument('--use_kan', type=lambda x: bool(strtobool(x)), default=True, help='Use KAN in FFN (default: True). If False, uses MLP.')
    parser.add_argument('--enable_global', type=lambda x: bool(strtobool(x)), default=True, help='Enable Global Branch in GFP (default: True).')
    parser.add_argument('--enable_local', type=lambda x: bool(strtobool(x)), default=True, help='Enable Local Branch in GFP (default: True).')
    parser.add_argument('--use_checkpoint', type=lambda x: bool(strtobool(x)), default=False, help='Enable gradient checkpointing (default: False).')
    parser.add_argument('--use_orkan', type=lambda x: bool(strtobool(x)), default=False, help='Use Orthogonal Rational KAN (default: False).')
    parser.add_argument('--use_okan', type=lambda x: bool(strtobool(x)), default=False, help='Use Orthogonal KAN without Rational denominator (default: False).')
    parser.add_argument('--use_fmkan', type=lambda x: bool(strtobool(x)), default=False, help='Use FM-KAN (Frequency Modulation KAN) (default: False).')
    parser.add_argument('--use_rkan', type=lambda x: bool(strtobool(x)), default=False, help='Use RationalKAN (Rational Function KAN) (default: False).')
    parser.add_argument('--use_drkan', type=lambda x: bool(strtobool(x)), default=False, help='Use DR-KAN (Dynamic Rational KAN) (default: False).')
    parser.add_argument('--pole_reg_lambda', type=float, default=0.0, help='Pole regularization lambda for RKAN/DR-KAN (default: 0.0, disabled).')

    args = parser.parse_args()
    main(args)
