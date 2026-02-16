#!/bin/bash

# --dataset [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
#            pneumoniamnist, retinamnist, breastmnist, bloodmnist,
#            organamnist, organcmnist, organsmnist, Kvasir, CPN,
#            Fetal, PAD, ISIC2018]

# Datasets to process (Same as train.sh)
datasets=(
    "breastmnist"
    "retinamnist"
    "pneumoniamnist"
    "dermamnist"
    "bloodmnist"
    # "organcmnist"
    # "organsmnist"
    # "organamnist"
    # "pathmnist"
    # "octmnist"
    # "chestmnist"
    # "tissuemnist"
)

# Function to run ablation experiments (Evaluation on MedMNIST-C)
run_ablation_c() {
    local dataset=$1
    local gpu_id=$2
    local output_file="${dataset}-c.txt" # Output to different file
    local model_name="MedViTv3_tiny"
    
    # Checkpoint path assumed from train.sh behavior
    local checkpoint="./${model_name}_${dataset}_best.pth"

    echo "Running MedMNIST-C evaluation for $dataset on GPU $gpu_id"
    echo "Checkpoint: $checkpoint"

    # Ensure checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "Error: Checkpoint $checkpoint not found!"
        return
    fi

    # Config matching train.sh active run
    # MedViTv3_tiny ACKAN --enable_local False --enable_global False
    echo "MedViTv3_tiny ACKAN --enable_local False --enable_global False (MedMNIST-C)" >> "$output_file"
    CUDA_VISIBLE_DEVICES=$gpu_id python main-c.py \
        --model_name "$model_name" \
        --dataset "$dataset" \
        --pretrained True \
        --use_ackan True \
        --enable_local False \
        --enable_global False \
        --eval_only True \
        --corruption all \
        --checkpoint_path "$checkpoint" >> "$output_file"

    # # Baseline (WavKAN) - Uncomment if needed and if checkpoint matches
    # echo "MedViTv3_tiny Baseline (WavKAN) (MedMNIST-C)" >> "$output_file"
    # CUDA_VISIBLE_DEVICES=$gpu_id python main-c.py \
    #     --model_name "$model_name" \
    #     --dataset "$dataset" \
    #     --pretrained True \
    #     --use_ackan True \
    #     --eval_only True \
    #     --corruption all \
    #     --checkpoint_path "$checkpoint" >> "$output_file"

    # # No KAN (MLP)
    # # ... (Add other configs if needed, ensuring checkpoint matches)
}

# ==========================================
# Execution on GPU 1
# ==========================================
for dataset in "${datasets[@]}"; do
    run_ablation_c "$dataset" 1
done
