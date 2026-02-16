#!/bin/bash

# Dataset: using 'pneumoniamnist' as a quick example. Change as needed.
DATASET='breastmnist'

# MedViT_tiny, KAN=True, SR=[8,4,2,1]
echo "Running: # MedViT_tiny, KAN=True, SR=[8,4,2,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan True --sr_ratios '[8,4,2,1]'

# MedViT_tiny, KAN=True, SR=[4,2,1,1]
echo "Running: # MedViT_tiny, KAN=True, SR=[4,2,1,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan True --sr_ratios '[4,2,1,1]'

# MedViT_tiny, KAN=True, SR=[2,2,1,1]
echo "Running: # MedViT_tiny, KAN=True, SR=[2,2,1,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan True --sr_ratios '[2,2,1,1]'

# MedViT_tiny, KAN=True, SR=[1,1,1,1]
echo "Running: # MedViT_tiny, KAN=True, SR=[1,1,1,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan True --sr_ratios '[1,1,1,1]'

# MedViT_tiny, KAN=False, SR=[8,4,2,1]
echo "Running: # MedViT_tiny, KAN=False, SR=[8,4,2,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan False --sr_ratios '[8,4,2,1]'

# MedViT_tiny, KAN=False, SR=[4,2,1,1]
echo "Running: # MedViT_tiny, KAN=False, SR=[4,2,1,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan False --sr_ratios '[4,2,1,1]'

# MedViT_tiny, KAN=False, SR=[2,2,1,1]
echo "Running: # MedViT_tiny, KAN=False, SR=[2,2,1,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan False --sr_ratios '[2,2,1,1]'

# MedViT_tiny, KAN=False, SR=[1,1,1,1]
echo "Running: # MedViT_tiny, KAN=False, SR=[1,1,1,1]"
python main_origin.py --model_name 'MedViT_tiny' --dataset $DATASET --pretrained True --use_kan False --sr_ratios '[1,1,1,1]'

