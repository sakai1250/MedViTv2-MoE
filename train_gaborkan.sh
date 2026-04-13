#!/bin/bash

# --dataset [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
#            pneumoniamnist, retinamnist, breastmnist, bloodmnist,
#            organamnist, organcmnist, organsmnist, Kvasir, CPN,
#            Fetal, PAD, ISIC2018]

# Datasets to process データセットサイズ順(昇順)
datasets=(
    "breastmnist"
    "retinamnist"
    # "pneumoniamnist"
    # "dermamnist"
    # "bloodmnist"
    # "organcmnist"
    # "organsmnist"
    # "organamnist"
    # "pathmnist"
    # "octmnist"
    # "chestmnist"
    # "tissuemnist"
)

# Function to run ablation experiments
run_ablation() {
    local dataset=$1
    local gpu_id=$2
    local output_file="${dataset}_gaborkan.txt"

    echo "Running ablation for $dataset on GPU $gpu_id"

    # ACKAN with enable_local=False, enable_global=False
    echo "MedViTv3_tiny ACKAN --enable_local False --enable_global False" >> "$output_file"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        --model_name 'MedViTv3_tiny' \
        --dataset "$dataset" \
        --pretrained True \
        --use_ackan True \
        --enable_local False \
        --enable_global False \
        --save_name "${dataset}_MedViTv3_tiny_gaborkan_no_local_no_global"
}

# ==========================================
# Execution on GPU 1
# ==========================================
for dataset in "${datasets[@]}"; do
    run_ablation "$dataset" 1
done
