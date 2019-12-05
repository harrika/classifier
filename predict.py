# ===========================================================
# RUBIC
# python predict.py /path/to/image checkpoint

# Return top KK most likely classes 
# python predict.py input checkpoint --top_k 3

#Use a mapping of categories to real names 
# python predict.py input checkpoint --category_names cat_to_name.json

#Use GPU for inference 
# python predict.py input checkpoint --gpu
# ===========================================================
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict  
import json
import PIL
from PIL import Image
import argparse
import base

parser = argparse.ArgumentParser()
#parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('input', default='./flowers/test/12/image_04014.jpg')
parser.add_argument('checkpoint', default='./checkpoint.pth')
parser.add_argument('--top_k', default=5, dest="top_k")
parser.add_argument('--category_names', default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu")

args = parser.parse_args()
path = args.checkpoint
image_path = args.input
topk = int(args.top_k)
dev = args.gpu

def main():
    model=base.loadchkpnt(path)
    with open('cat_to_name.json', 'r') as jfil:
    	cat_to_name = json.load(jfil)
    probs = base.predict(image_path, model, topk, dev)
    names = [cat_to_name[str(ix + 1)] for ix in np.array(probs[1][0])]
    probability = np.array(probs[0][0])
    i=0
    while i < topk:
        print("{} has probability: {}".format(names[i], probability[i]))
        i += 1

if __name__== "__main__":
	main()

