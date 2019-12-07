# AI Programming with Python Project

This is the command line section of hte Immage classifier project
It is mainlycomprises of three  files

base.py  - contains helper modules and functions

predict.py - for predicting an image
 -- example usage -- 
python predict.py "test_flowers.jpg" "checkpoint.pth" --gpu gpu

train.py - for training a model and saving its checkpoint
---example usage ---
python train.py ./flowers --arch "vgg16"


