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

# Function to run Gabor-RKAN experiments
# LFP: ACKAN (ComplexGaborConv2d), GFP: RKAN (Rational KAN)
run_experiment() {
    local dataset=$1
    local gpu_id=$2
    echo "Running MedViT with Gabor-RKAN experiment for $dataset on GPU $gpu_id" >> "${dataset}_gabor_rkan.txt"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 main_vv.py --model_name 'MedViT_tiny' --dataset "$dataset" --pretrained True --epochs 100 --use_checkpoint True --use_ackan True --use_rkan True
}

# ==========================================
# Execution on GPU 1
# ==========================================
for dataset in "${datasets[@]}"; do
    run_experiment "$dataset" 1
done
