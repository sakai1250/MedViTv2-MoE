
print("Starting debug_import.py...")
import sys
print("Importing torch...")
import torch
print("Importing torch.nn...")
import torch.nn as nn
print("Importing time...")
import time
print("Importing fvcore...")
try:
    from fvcore.nn import FlopCountAnalysis
    print("fvcore imported.")
except ImportError:
    print("fvcore not found.")

print("Importing MedViT_tiny...")
try:
    from MedViT import MedViT_tiny
    print("MedViT_tiny imported.")
except Exception as e:
    print(f"Failed to import MedViT_tiny: {e}")

print("Importing MedViTVV_tiny...")
try:
    from medvitvv import MedViTVV_tiny
    print("MedViTVV_tiny imported.")
except Exception as e:
    print(f"Failed to import MedViTVV_tiny: {e}")

print("Imports done.")
