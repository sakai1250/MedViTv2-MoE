#!/bin/bash

# Datasets to process (defaulting to a small dataset for testing)
datasets=(
    "breastmnist"
    "retinamnist"
)

run_ablation() {
    local dataset=$1
    local gpu_id=$2
    local output_file="${dataset}_ablation_result.md"

    echo "Running Gabor-KAN Ablation for $dataset on GPU $gpu_id"

    # Write header for Markdown table
    echo "## Gabor-KAN Ablation Study Results ($dataset)" > "$output_file"
    echo "| ACKAN Stages | Test Acc | AUC |" >> "$output_file"
    echo "| :--- | :---: | :---: |" >> "$output_file"

    mkdir -p checkpoints_ablation

    extract_and_log() {
        local setting_name=$1
        # Extract the last metrics from {dataset}.txt
        local testacc=$(tail -n 5 "${dataset}.txt" | grep 'testacc' | awk '{print $2}')
        local auc=$(tail -n 5 "${dataset}.txt" | grep 'auc' | awk '{print $2}')
        # Fallback if empty
        if [[ -z "$testacc" ]]; then testacc="N/A"; fi
        if [[ -z "$auc" ]]; then auc="N/A"; fi
        
        echo "| $setting_name | $testacc | $auc |" >> "$output_file"
    }

    echo "train_gaborkan.sh"
    bash train_gaborkan.sh

    echo "=== 1. All Stages [0, 1, 2, 3] ==="
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True --enable_local False --enable_global False --ackan_stages 0 1 2 3 --epochs 100
    extract_and_log "All Stages [0, 1, 2, 3]"

    echo "=== 2. Shallow Stages Only [0, 1] ==="
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True --enable_local False --enable_global False --ackan_stages 0 1 --epochs 100
    extract_and_log "Shallow Stages [0, 1]"

    echo "=== 3. Deep Stages Only [2, 3] ==="
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True --enable_local False --enable_global False --ackan_stages 2 3 --epochs 100
    extract_and_log "Deep Stages [2, 3]"

    echo "=== 4. No ACKAN (Standard Conv/MLP) ==="
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan False --enable_local False --enable_global False --epochs 100
    extract_and_log "No ACKAN (Baseline)"

    echo "Ablation completed for $dataset. Results saved to $output_file"
    cat "$output_file"
}

# ==========================================
# Execution
# ==========================================
for dataset in "${datasets[@]}"; do
    run_ablation "$dataset" 1
done
