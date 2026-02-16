#!/bin/bash

# --dataset [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
#            pneumoniamnist, retinamnist, breastmnist, bloodmnist,
#            organamnist, organcmnist, organsmnist, Kvasir, CPN,
#            Fetal, PAD, ISIC2018]

# MedMNIST v2 データセットサイズ順 (降順)
# 順位	データセット名	合計サンプル数	(内訳: Train / Val / Test)
# 1	TissueMNIST	236,386	(165,466 / 23,640 / 47,280)
# 2	ChestMNIST	112,120	(78,468 / 11,219 / 22,433)
# 3	OCTMNIST	109,309	(97,477 / 10,832 / 1,000)
# 4	PathMNIST	107,180	(89,996 / 10,004 / 7,180)
# 5	OrganAMNIST	58,850	(34,581 / 6,491 / 17,778)
# 6	OrganSMNIST	25,211	(13,932 / 2,452 / 8,827)
# 7	OrganCMNIST	23,660	(13,000 / 2,392 / 8,268)
# 8	BloodMNIST	17,092	(11,959 / 1,712 / 3,421)
# 9	DermaMNIST	10,015	(7,007 / 1,003 / 2,005)
# 10	PneumoniaMNIST	5,856	(4,708 / 524 / 624)
# 12	RetinaMNIST	1,600	(1,080 / 120 / 400)
# 13	BreastMNIST	780	(546 / 78 / 156)

# Datasets to process データセットサイズ順(昇順)
datasets=(
    # "breastmnist"
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

# Function to run ablation experiments
run_ablation() {
    local dataset=$1
    local gpu_id=$2
    local output_file="${dataset}.txt"

    echo "Running ablation for $dataset on GPU $gpu_id"

    # ACKAN
    # echo "MedViTv3_tiny ACKAN" >> "$output_file"
    # CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True
    echo "MedViTv3_tiny ACKAN --enable_local False --enable_global False" >> "$output_file"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True --enable_local False --enable_global False

    # # Baseline (WavKAN)
    # echo "MedViTv3_tiny Baseline (WavKAN)" >> "$output_file"
    # CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True

    # # No KAN (MLP)
    # echo "MedViTv3_tiny No KAN (MLP)" >> "$output_file"
    # CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True --use_kan False

    # # No Global Branch
    # echo "MedViTv3_tiny No Global Branch" >> "$output_file"
    # CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True --enable_global False

    # # No Local Branch
    # echo "MedViTv3_tiny No Local Branch" >> "$output_file"
    # CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name 'MedViTv3_tiny' --dataset "$dataset" --pretrained True --use_ackan True --enable_local False
}

# ==========================================
# Execution on GPU 1
# ==========================================
# for dataset in "${datasets[@]}"; do
#     run_ablation "$dataset" 1
# done

# ==========================================
# Execution on GPU 0
# ==========================================
for dataset in "${datasets[@]}"; do
    run_ablation "$dataset" 0
done
