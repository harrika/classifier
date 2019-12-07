# ========================================================================
# RUBIC
# python train.py data_directory
# python train.py data_dir --save_dir save_directory
# python train.py data_dir --arch "vgg13"
# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# python train.py data_dir --gpu
# =======================================================================
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict  
from PIL import Image
import argparse
import base

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", default="./flowers")
parser.add_argument("--arch", choices=["vgg", "dense"], default="vgg")
parser.add_argument("--hidden_units", default=4096)
parser.add_argument("--dropout", default=0.5)
parser.add_argument("--learning_rate", default=0.001)  
parser.add_argument("--gpu", default="gpu")
parser.add_argument("--epochs", default=2)
parser.add_argument("--every", default=42)
parser.add_argument("--save_dir", default="checkpoint.pth")
args = parser.parse_args()

rootdir = args.data_dir
arch = args.arch
layer1 = int(args.hidden_units)
dropout = float(args.dropout)
lr = float(args.learning_rate)
gpu = args.gpu
epochs = int(args.epochs)
every = int(args.every)
path = args.save_dir

def Main():
  trainloded, validloded, testloded  = base.transloader(rootdir)
  model, optimizer ,criterion = base.network_construct(arch, layer1, dropout, lr)
  base.deeplearn(model, trainloded, epochs, every, criterion, optimizer, gpu)   
  base.savechkpnt(model, arch, layer1, path)
  print("checkpoint saved at ", path)

if __name__ == "__main__":
	Main()
