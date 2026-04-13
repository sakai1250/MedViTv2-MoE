import medmnist
from medmnist import INFO
info = INFO['breastmnist']
DataClass = getattr(medmnist, info['python_class'])
dataset = DataClass(split='train', download=True, size=224)
print("Shape of first image:", dataset.imgs[0].shape)
