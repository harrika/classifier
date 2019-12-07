import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict  
from PIL import Image
from torchvision import transforms


def imagetrans(images='flowers'):
    data_dir = images
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])
                                         ]),
    'transforms': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])
                                         ])
    }
    image_datasets = {
    'trainset': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
    'validset': datasets.ImageFolder(valid_dir, transform=data_transforms['transforms']),
    'testset': datasets.ImageFolder(test_dir, transform=data_transforms['transforms'])
    }

    return image_datasets['trainset'], image_datasets['validset'], image_datasets['testset']
    
traindat, validat, testdat = imagetrans('flowers')

def transloader(images):    
    dataloaders = {
    'trainloader': torch.utils.data.DataLoader(traindat, batch_size=64, shuffle=True),
    'validloader': torch.utils.data.DataLoader(validat, batch_size=32),
    'testloader': torch.utils.data.DataLoader(testdat, batch_size=32)
    }    
    return dataloaders['trainloader'], dataloaders['validloader'], dataloaders['testloader']

trainloded, validloded, testloded = transloader('flowers')

def modense():
    model=models.densenet121(pretrained=True)
def movgg():
    model=models.vgg16(pretrained=True)

def network_construct(arch='vgg', layer1=4096, dropout=0.5, lr = 0.001):    
    if arch == 'vgg':        
        inlayer = 25088
        model = models.vgg16(pretrained=True)
    else:
        inlayer = 1024
        model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False        
         
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inlayer,layer1)),
                          ('relu1', nn.ReLU()),
                          ('d_out1',nn.Dropout(dropout)),
                          ('fc2', nn.Linear(layer1, 1024)),
                          ('relu2', nn.ReLU()),
                          ('d_out2',nn.Dropout(dropout)),
                          ('fc3', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))        
        
    model.classifier = classifier
    criterion = nn.NLLLoss()    
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    #model.cuda()        
    return model, optimizer ,criterion

def validation(mm, op, cr):
    mm.eval()
    val_lost = 0
    val_accuracy=0
    for ii, (inputsv,labelsv) in enumerate(validloded):
        op.zero_grad()
        inputsv, labelsv = inputsv.to('cuda') , labelsv.to('cuda')
        mm.to('cuda')
        with torch.no_grad():    
            outputs = mm.forward(inputsv)
            val_lost = cr(outputs,labelsv)
            ps = torch.exp(outputs).data
            equality = (labelsv.data == ps.max(1)[1])
            #val_accuracy += equality.type_as(torch.FloatTensor()).mean()
            val_accuracy += equality.type(torch.FloatTensor).mean()

    val_lost = val_lost / len(validloded)
    val_accuracy = val_accuracy /len(validloded)
    mm.train()
    return val_lost, val_accuracy

def deeplearn(model, trainloader, epochs, every, criterion, optimizer, device='cpu'):
    print_every = every
    steps = 0
    model.to('cuda')
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloded):
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()    
            if steps % print_every == 0:                
                val_lost, val_accuracy = validation(model, optimizer, criterion)                 
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training loss: {:.3f}".format(running_loss/print_every),
                  "Test loss {:.3f}".format(val_lost),
                  "Accuracy: {:.3f}".format(val_accuracy))                
                running_loss = 0

            
def savechkpnt(model, arch="vgg", layer1=4096, path="checkpoint.pth"):
    model.class_to_idx = traindat.class_to_idx    
    model.cpu
    torch.save({'arch':arch,
            'layer1':layer1,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            path)

def loadchkpnt(path):
    checkpoint = torch.load(path)  
    arch = checkpoint['arch']
    layer1 = checkpoint['layer1']      
    model,optimizer,criterion = network_construct(arch=arch, layer1=layer1)   
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model    

def process_image(image):
    img = Image.open(image)   
    img_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_image = img_trans(img)
    img_procd = np.array(tensor_image)
    img_procd = img_procd.transpose((0, 2, 1))
    return img_procd

def predict(image_path, model, topk=5, gpu="gpu"):
    ''' Predict the class of an image using a trained deep learning model'''
    
    gpu =='gpu' and model.to('cuda')
    img = process_image(image_path)    
    img = img.unsqueeze_(0)
    img = img.float()
    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img)  
        
    probability = torch.exp(output.data,dim=1)
    return probability.topk(topk)

