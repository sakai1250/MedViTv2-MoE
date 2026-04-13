#!/bin/bash

# MedViT22: KAN Node Aggregation Ablation Study
# Compare three aggregation modes:
#   sum:       Original KAN (Kolmogorov-Arnold representation theorem)
#   mean:      Average aggregation (Altarabichi 2024)
#   attention: Attention-weighted aggregation (Proposed)
#
# 3 aggregation modes × 4 datasets × 3 seeds = 36 experiments

datasets=(
    "breastmnist"
    "retinamnist"
    "pneumoniamnist"
    "dermamnist"
    # "bloodmnist"
    # "organcmnist"
    # "organsmnist"
    # "organamnist"
    # "pathmnist"
    # "octmnist"
    # "chestmnist"
    # "tissuemnist"
)

aggregation_modes=("sum" "mean" "attention")

run_aggregation_ablation() {
    local dataset=$1
    local gpu_id=$2
    local output_file="${dataset}_medvit22.txt"

    echo "Running MedViT22 KAN Aggregation Ablation for $dataset on GPU $gpu_id"

    for seed in 42 0 1; do
        for agg_mode in "${aggregation_modes[@]}"; do
            label="MedViT22_tiny kan_aggregation=$agg_mode seed=$seed"
            echo "$label" >> "$output_file"
            echo ">>> $label"
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                --model_name 'MedViT22_tiny' \
                --dataset "$dataset" \
                --pretrained True \
                --kan_aggregation "$agg_mode" \
                --seed $seed

            label="MedViT22_small kan_aggregation=$agg_mode seed=$seed"
            echo "$label" >> "$output_file"
            echo ">>> $label"
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                --model_name 'MedViT22_small' \
                --dataset "$dataset" \
                --pretrained True \
                --kan_aggregation "$agg_mode" \
                --seed $seed
        done
    done
}

# ==========================================
# Execution on GPU 1
# ==========================================
for dataset in "${datasets[@]}"; do
    run_aggregation_ablation "$dataset" 1
done

# # ==========================================
# # Execution on GPU 0
# # ==========================================
# for dataset in "${datasets[@]}"; do
#     run_aggregation_ablation "$dataset" 0
# done
