#!/bin/bash

# MedViT2.5 Ablation Study
# Ablation targets:
#   --parallel_gfp         True/False  : GFP の並列ブランチ (v3スタイル) ON/OFF
#   --external_lfp_residual True/False : LFP の FFN 残差を外部管理 (二重残差解消) ON/OFF
#
# 2x2 の4条件を全データセットで実施

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

run_ablation() {
    local dataset=$1
    local gpu_id=$2
    local output_file="${dataset}.txt"

    echo "Running MedViT2.5 ablation for $dataset on GPU $gpu_id"

    for seed in 42 0 1; do
        for parallel in "True" "False"; do
            for ext_res in "True" "False"; do
                label="MedViT25_tiny parallel_gfp=$parallel external_lfp_residual=$ext_res seed=$seed"
                echo "$label" >> "$output_file"
                CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                    --model_name 'MedViT25_tiny' \
                    --dataset "$dataset" \
                    --pretrained True \
                    --parallel_gfp $parallel \
                    --external_lfp_residual $ext_res \
                    --seed $seed

                label="MedViT25_small parallel_gfp=$parallel external_lfp_residual=$ext_res seed=$seed"
                echo "$label" >> "$output_file"
                CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                    --model_name 'MedViT25_small' \
                    --dataset "$dataset" \
                    --pretrained True \
                    --parallel_gfp $parallel \
                    --external_lfp_residual $ext_res \
                    --seed $seed
            done
        done
    done
}

# ==========================================
# Execution on GPU 1
# ==========================================
for dataset in "${datasets[@]}"; do
    run_ablation "$dataset" 1
done

# # ==========================================
# # Execution on GPU 0
# # ==========================================
# for dataset in "${datasets[@]}"; do
#     run_ablation "$dataset" 0
# done
