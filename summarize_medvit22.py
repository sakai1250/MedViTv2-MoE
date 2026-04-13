import re
import csv
from collections import defaultdict

datasets = ["breastmnist", "retinamnist", "pneumoniamnist", "dermamnist"]
target_model = "MedViT22_tiny"

# {(dataset, aggregation_mode): {"acc": [values], "auc": [values]}}
results = defaultdict(lambda: {"acc": [], "auc": []})

for dataset in datasets:
    filepath = f"{dataset}.txt"
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # MedViT22_tiny kan_aggregation=sum seed=42
        match = re.match(r"(MedViT22_\w+)\s+kan_aggregation=(\w+)\s+seed=(\d+)", line)
        if match:
            model = match.group(1)
            agg_mode = match.group(2)
            seed = match.group(3)

            if model == target_model:
                # 次の数行からtestacc, aucを取得
                acc_val = None
                auc_val = None
                for j in range(i + 1, min(i + 6, len(lines))):
                    l = lines[j].strip()
                    m_acc = re.match(r"testacc\s+([\d.]+)%", l)
                    m_auc = re.match(r"auc\s+([\d.]+)%", l)
                    if m_acc:
                        acc_val = float(m_acc.group(1))
                    if m_auc:
                        auc_val = float(m_auc.group(1))

                if acc_val is not None and auc_val is not None:
                    results[(dataset, agg_mode)]["acc"].append(acc_val)
                    results[(dataset, agg_mode)]["auc"].append(auc_val)
        i += 1

# CSV出力
output_file = "medvit22_tiny_summary.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "aggregation_mode", "acc_mean", "auc_mean", "acc_std", "auc_std", "n_seeds"])

    for dataset in datasets:
        for agg_mode in ["sum", "mean", "attention"]:
            key = (dataset, agg_mode)
            if key in results:
                acc_list = results[key]["acc"]
                auc_list = results[key]["auc"]
                n = len(acc_list)
                acc_mean = sum(acc_list) / n
                auc_mean = sum(auc_list) / n
                acc_std = (sum((x - acc_mean) ** 2 for x in acc_list) / n) ** 0.5
                auc_std = (sum((x - auc_mean) ** 2 for x in auc_list) / n) ** 0.5
                writer.writerow([
                    dataset, agg_mode,
                    f"{acc_mean:.2f}", f"{auc_mean:.2f}",
                    f"{acc_std:.2f}", f"{auc_std:.2f}",
                    n
                ])

print(f"Saved to {output_file}")

# 表示
print(f"\n{'Dataset':<18} {'Agg Mode':<12} {'ACC Mean':>10} {'AUC Mean':>10} {'ACC Std':>10} {'AUC Std':>10} {'N':>3}")
print("-" * 75)
for dataset in datasets:
    for agg_mode in ["sum", "mean", "attention"]:
        key = (dataset, agg_mode)
        if key in results:
            acc_list = results[key]["acc"]
            auc_list = results[key]["auc"]
            n = len(acc_list)
            acc_mean = sum(acc_list) / n
            auc_mean = sum(auc_list) / n
            acc_std = (sum((x - acc_mean) ** 2 for x in acc_list) / n) ** 0.5
            auc_std = (sum((x - auc_mean) ** 2 for x in auc_list) / n) ** 0.5
            print(f"{dataset:<18} {agg_mode:<12} {acc_mean:>9.2f}% {auc_mean:>9.2f}% {acc_std:>9.2f}% {auc_std:>9.2f}% {n:>3}")
