# --dataset [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
#            pneumoniamnist, retinamnist, breastmnist, bloodmnist,
#            organamnist, organcmnist, organsmnist, Kvasir, CPN,
#            Fetal, PAD, ISIC2018]
# python main.py --model_name 'MedViT_tiny' --dataset 'breastmnist' --pretrained True
# echo "--use_kmp_glu True alpha=1.0" >> breastmnist.txt
# python main.py --model_name 'MedViT_tiny' --dataset 'breastmnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_small' --dataset 'breastmnist' --pretrained True
echo "--use_kmp_glu True alpha=1.0 small" >> breastmnist.txt
python main.py --model_name 'MedViT_small' --dataset 'breastmnist' --pretrained True --use_kmp_glu True

# python main.py --model_name 'MedViT_tiny' --dataset 'retinamnist' --pretrained True
# echo "--use_kmp_glu True alpha=1.0" >> retinamnist.txt
# python main.py --model_name 'MedViT_tiny' --dataset 'retinamnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_small' --dataset 'retinamnist' --pretrained True
echo "--use_kmp_glu True alpha=1.0 small" >> retinamnist.txt
python main.py --model_name 'MedViT_small' --dataset 'retinamnist' --pretrained True --use_kmp_glu True

# python main.py --model_name 'MedViT_tiny' --dataset 'pneumoniamnist' --pretrained True
# echo "--use_kmp_glu True alpha=1.0" >> pneumoniamnist.txt
# python main.py --model_name 'MedViT_tiny' --dataset 'pneumoniamnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_small' --dataset 'pneumoniamnist' --pretrained True
echo "--use_kmp_glu True alpha=1.0 small" >> pneumoniamnist.txt
python main.py --model_name 'MedViT_small' --dataset 'pneumoniamnist' --pretrained True --use_kmp_glu True

# python main.py --model_name 'MedViT_tiny' --dataset 'dermamnist' --pretrained True
# echo "--use_kmp_glu True alpha=1.0" >> dermamnist.txt
# python main.py --model_name 'MedViT_tiny' --dataset 'dermamnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_small' --dataset 'dermamnist' --pretrained True
echo "--use_kmp_glu True alpha=1.0 small" >> dermamnist.txt
python main.py --model_name 'MedViT_small' --dataset 'dermamnist' --pretrained True --use_kmp_glu True

# python main.py --model_name 'MedViT_tiny' --dataset 'bloodmnist' --pretrained True
# python main.py --model_name 'MedViT_tiny' --dataset 'bloodmnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_small' --dataset 'bloodmnist' --pretrained True
# python main.py --model_name 'MedViT_small' --dataset 'bloodmnist' --pretrained True --use_kmp_glu True



# python main.py --model_name 'MedViT_tiny' --dataset 'tissuemnist' --pretrained True
# python main.py --model_name 'MedViT_tiny' --dataset 'pathmnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'chestmnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'octmnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'organamnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'organsmnist' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'Kvasir' --pretrained True --use_kmp_glu True


# python main.py --model_name 'MedViT_tiny' --dataset 'CPN' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'Fetal' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'PAD' --pretrained True --use_kmp_glu True
# python main.py --model_name 'MedViT_tiny' --dataset 'ISIC2018' --pretrained True --use_kmp_glu True
