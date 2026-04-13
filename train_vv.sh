#!/bin/bash

# --dataset [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
#            pneumoniamnist, retinamnist, breastmnist, bloodmnist,
#            organamnist, organcmnist, organsmnist, Kvasir, CPN,
#            Fetal, PAD, ISIC2018]

# Datasets to process
datasets=(
    # "bloodmnist"
    "breastmnist"
    "retinamnist"
    # "pneumoniamnist"
    # "dermamnist"
    # "organcmnist"
    # "organsmnist"
    # "organamnist"
    # "pathmnist"
    # "octmnist"
    # "chestmnist"
    # "tissuemnist"
)

# Function to run ablation experiments
run_experiment() {
    local dataset=$1
    local gpu_id=$2
    echo "Running MedViTVV experiment for $dataset on GPU $gpu_id"
    # Output handled by main_vv.py
    # output_file="${dataset}_orkan.txt"
    # Using --pretrained True to load MedViT weights as initialization where possible
    # Using batch_size 24 (Standard) with FastORKAN + Checkpointing
    CUDA_VISIBLE_DEVICES=$gpu_id python3 main_vv.py --model_name 'MedViT_tiny' --dataset "$dataset" --pretrained True --epochs 100 --use_checkpoint True --use_orkan True
}

# ==========================================
# Execution on GPU 1
# ==========================================
for dataset in "${datasets[@]}"; do
    run_experiment "$dataset" 1
done
