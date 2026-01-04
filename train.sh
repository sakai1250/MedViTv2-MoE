# --dataset [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
#            pneumoniamnist, retinamnist, breastmnist, bloodmnist,
#            organamnist, organcmnist, organsmnist, Kvasir, CPN,
#            Fetal, PAD, ISIC2018]
# python main.py --model_name 'MedViT_tiny' --dataset 'breastmnist' --pretrained True


# BreastMNIST Ablation
echo "MedViTv3_tiny Baseline (WavKAN)" >> breastmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'breastmnist' --pretrained True --use_ackan True

echo "MedViTv3_tiny No KAN (MLP)" >> breastmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'breastmnist' --pretrained True --use_ackan True --use_kan False

echo "MedViTv3_tiny No Global Branch" >> breastmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'breastmnist' --pretrained True --use_ackan True --enable_global False

echo "MedViTv3_tiny No Local Branch" >> breastmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'breastmnist' --pretrained True --use_ackan True --enable_local False

# RetinaMNIST Ablation
echo "MedViTv3_tiny Baseline (WavKAN)" >> retinamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'retinamnist' --pretrained True --use_ackan True
echo "MedViTv3_tiny No KAN (MLP)" >> retinamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'retinamnist' --pretrained True --use_ackan True --use_kan False

echo "MedViTv3_tiny No Global Branch" >> retinamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'retinamnist' --pretrained True --use_ackan True --enable_global False

echo "MedViTv3_tiny No Local Branch" >> retinamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'retinamnist' --pretrained True --use_ackan True --enable_local False

# echo "--use_kmp_glu True tiny3" >> breastmnist.txt
# python main.py --model_name 'MedViTv3_tiny' --dataset 'breastmnist' --pretrained True --use_wavkan True
#echo "--use_kmp_glu True tiny" >> breastmnist.txt
#python main.py --model_name 'MedViT_tiny' --dataset 'breastmnist' --pretrained True



# python main.py --model_name 'MedViT_small' --dataset 'breastmnist' --pretrained True
# echo "--use_kmp_glu True small" >> breastmnist.txt
# python main.py --model_name 'MedViT_small' --dataset 'breastmnist' --pretrained True --use_kmp_glu True

# python main.py --model_name 'MedViT_tiny' --dataset 'retinamnist' --pretrained True

# echo "--use_kmp_glu True tiny3" >> retinamnist.txt
# python main.py --model_name 'MedViTv3_tiny' --dataset 'retinamnist' --pretrained True --use_wavkan True
#echo "--use_kmp_glu True tiny" >> retinamnist.txt
#python main.py --model_name 'MedViT_tiny' --dataset 'retinamnist' --pretrained True


# python main.py --model_name 'MedViT_small' --dataset 'retinamnist' --pretrained True
# echo "--use_kmp_glu True small" >> retinamnist.txt
# python main.py --model_name 'MedViT_small' --dataset 'retinamnist' --pretrained True --use_kmp_glu True

# PneumoniaMNIST Ablation
echo "MedViTv3_tiny Baseline (WavKAN)" >> pneumoniamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'pneumoniamnist' --pretrained True --use_ackan True
echo "MedViTv3_tiny No KAN (MLP)" >> pneumoniamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'pneumoniamnist' --pretrained True --use_ackan True --use_kan False
echo "MedViTv3_tiny No Global Branch" >> pneumoniamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'pneumoniamnist' --pretrained True --use_ackan True --enable_global False
echo "MedViTv3_tiny No Local Branch" >> pneumoniamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'pneumoniamnist' --pretrained True --use_ackan True --enable_local False

# echo "--use_kmp_glu True tiny3" >> pneumoniamnist.txt
# python main.py --model_name 'MedViTv3_tiny' --dataset 'pneumoniamnist' --pretrained True --use_wavkan True
# echo "--use_kmp_glu True tiny" >> pneumoniamnist.txt
# python main.py --model_name 'MedViT_tiny' --dataset 'pneumoniamnist' --pretrained True
# # python main.py --model_name 'MedViT_tiny' --dataset 'pneumoniamnist' --pretrained True
# # echo "--use_kmp_glu True alpha=1.0" >> pneumoniamnist.txt
# # python main.py --model_name 'MedViT_tiny' --dataset 'pneumoniamnist' --pretrained True --use_kmp_glu True
# # python main.py --model_name 'MedViT_small' --dataset 'pneumoniamnist' --pretrained True
# # echo "--use_kmp_glu True small" >> pneumoniamnist.txt
# # python main.py --model_name 'MedViT_small' --dataset 'pneumoniamnist' --pretrained True --use_kmp_glu True

# # Dermamnist Ablation
echo "MedViTv3_tiny Baseline (WavKAN)" >> dermamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'dermamnist' --pretrained True --use_ackan True
echo "MedViTv3_tiny No KAN (MLP)" >> dermamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'dermamnist' --pretrained True --use_ackan True --use_kan False
echo "MedViTv3_tiny No Global Branch" >> dermamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'dermamnist' --pretrained True --use_ackan True --enable_global False
echo "MedViTv3_tiny No Local Branch" >> dermamnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'dermamnist' --pretrained True --use_ackan True --enable_local False
# # python main.py --model_name 'MedViT_tiny' --dataset 'dermamnist' --pretrained True
# # echo "--use_kmp_glu True alpha=1.0" >> dermamnist.txt
# # python main.py --model_name 'MedViT_tiny' --dataset 'dermamnist' --pretrained True --use_kmp_glu True
# # python main.py --model_name 'MedViT_small' --dataset 'dermamnist' --pretrained True
# # echo "--use_kmp_glu True small" >> dermamnist.txt
# # python main.py --model_name 'MedViT_small' --dataset 'dermamnist' --pretrained True --use_kmp_glu True
# echo "--use_kmp_glu True tiny3" >> dermamnist.txt
# python main.py --model_name 'MedViTv3_tiny' --dataset 'dermamnist' --pretrained True --use_wavkan True
# echo "--use_kmp_glu True tiny" >> dermamnist.txt
# python main.py --model_name 'MedViT_tiny' --dataset 'dermamnist' --pretrained True

# # BloodMNIST Ablation
echo "MedViTv3_tiny Baseline (WavKAN)" >> bloodmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'bloodmnist' --pretrained True --use_ackan True
echo "MedViTv3_tiny No KAN (MLP)" >> bloodmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'bloodmnist' --pretrained True --use_ackan True --use_kan False
echo "MedViTv3_tiny No Global Branch" >> bloodmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'bloodmnist' --pretrained True --use_ackan True --enable_global False
echo "MedViTv3_tiny No Local Branch" >> bloodmnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'bloodmnist' --pretrained True --use_ackan True --enable_local False

# # python main.py --model_name 'MedViT_tiny' --dataset 'bloodmnist' --pretrained True
# # python main.py --model_name 'MedViT_tiny' --dataset 'bloodmnist' --pretrained True --use_kmp_glu True
# # python main.py --model_name 'MedViT_small' --dataset 'bloodmnist' --pretrained True
# # python main.py --model_name 'MedViT_small' --dataset 'bloodmnist' --pretrained True --use_kmp_glu True
# echo "--use_kmp_glu True tiny3" >> bloodmnist.txt
# python main.py --model_name 'MedViTv3_tiny' --dataset 'bloodmnist' --pretrained True --use_wavkan True
# echo "--use_kmp_glu True tiny" >> bloodmnist.txt
# python main.py --model_name 'MedViT_tiny' --dataset 'bloodmnist' --pretrained True

# TissueMNIST Ablation
echo "MedViTv3_tiny Baseline (WavKAN)" >> tissuemnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'tissuemnist' --pretrained True --use_ackan True
echo "MedViTv3_tiny No KAN (MLP)" >> tissuemnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'tissuemnist' --pretrained True --use_ackan True --use_kan False
echo "MedViTv3_tiny No Global Branch" >> tissuemnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'tissuemnist' --pretrained True --use_ackan True --enable_global False
echo "MedViTv3_tiny No Local Branch" >> tissuemnist.txt
python main.py --model_name 'MedViTv3_tiny' --dataset 'tissuemnist' --pretrained True --use_ackan True --enable_local False

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
