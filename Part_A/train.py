import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torch.nn.functional as func
# import wandb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

!wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip -O nature_12K.zip
!unzip -q nature_12K.zip
!rm nature_12K.zip


dataset = torchvision.datasets.ImageFolder(root='inaturalist_12K/train', transform=transforms)
from torchvision import transforms
transform = transforms.Compose([transforms.Resize((200,200)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])])

train_dataset = ImageFolder(root='inaturalist_12K/train', transform=transform)
test_dataset = ImageFolder(root='inaturalist_12K/val', transform=transform)
# test_dataset = ImageFolder(root='inaturalist_12K/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the size of the validation set (20%)
validation_size = 0.2

# Split the dataset into training and validation sets with shuffling
train_data, val_data = train_test_split(list(range(len(train_dataset))), test_size=validation_size, shuffle=True, random_state=42)
#Data Loader

train_subset = Subset(train_dataset, train_data)
val_subset  = Subset(train_dataset, val_data)

# train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_subset, batch_size=32, shuffle=True)

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
# print(len(train_dataset))
# print(len(test_dataset))
# print(len(train_data))
# print(len(val_data))

# dataiter = iter(train_loader)
# data = next(dataiter)
# inputs, targets = data
# print(inputs.shape, targets.shape)

def DataLoaders(aug,batch_size):
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    return train_loader,val_loader

    import math
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def choose_optimizer(algo,epochs,eta,n_filters,filter_size,activationFun,denseLay,aug,batch_size,drop_out,norm):
    if algo == 'adam':
        Network=ConvNetwork(n_filters,filter_size,activationFun,denseLay,drop_out,norm).to(device)
        Algo = torch.optim.Adam(Network.parameters(), lr=eta)
        train_model(epochs,Network,Algo,activationFun,aug,batch_size)
    if algo == 'nadam':
        Network=ConvNetwork(n_filters,filter_size,activationFun,denseLay,drop_out,norm).to(device)
        Algo = torch.optim.NAdam(Network.parameters(), lr=eta)
        train_model(epochs,Network,Algo,activationFun,aug,batch_size)
    if algo == 'sgd':
        Network=ConvNetwork(n_filters,filter_size,activationFun,denseLay,drop_out,norm).to(device)
        Algo = torch.optim.SGD(Network.parameters(), lr=eta)
        train_model(epochs,Network,Algo,activationFun,aug,batch_size)


def compute_size(input_size, kernel_size,stride):
    padding=0
    output_size = math.floor((input_size - kernel_size + 2 * padding) / stride) + 1
#     print(output_size)
    return output_size



class ConvNetwork(nn.Module):
    def __init__(self,n_filters,filter_size,activationFun,denseLay,drop_out,norm):
        super(ConvNetwork, self).__init__()
        self.drop_out = drop_out
        self.norm = norm
        self.conv1 = nn.Conv2d(3,n_filters,filter_size,stride=1)
        self.norm1 = nn.BatchNorm2d(n_filters)
        self.size_after_conv1 = compute_size(200, filter_size, stride=1)
        self.pool1 = nn.MaxPool2d(filter_size,stride=2)
        self.size_after_pool1 = compute_size(self.size_after_conv1, filter_size, stride=2)

        self.conv2 = nn.Conv2d(n_filters, n_filters*2,filter_size,stride=1)
        self.norm2 = nn.BatchNorm2d(n_filters*2)
        self.size_after_conv2 = compute_size(self.size_after_pool1, filter_size, stride=1)
        self.pool2 = nn.MaxPool2d(filter_size,stride=2)
        self.size_after_pool2 = compute_size(self.size_after_conv2, filter_size, stride=2)

        self.conv3 = nn.Conv2d(n_filters*2,n_filters*4,filter_size,stride=1)
        self.norm3 = nn.BatchNorm2d(n_filters*4)
        self.size_after_conv3 = compute_size(self.size_after_pool2, filter_size, stride=1)
        self.pool3= nn.MaxPool2d(filter_size,stride=2)
        self.size_after_pool3 = compute_size(self.size_after_conv3, filter_size, stride=2)

        self.conv4 = nn.Conv2d(n_filters*4,n_filters*8,filter_size,stride=1)
        self.norm4 = nn.BatchNorm2d(n_filters*8)
        self.size_after_conv4 = compute_size(self.size_after_pool3, filter_size, stride=1)
        self.pool4 = nn.MaxPool2d(filter_size,stride=2)
        self.size_after_pool4 = compute_size(self.size_after_conv4, filter_size, stride=2)


        self.conv5 = nn.Conv2d(n_filters*8,n_filters*16,filter_size,stride=1)
        self.norm5 = nn.BatchNorm2d(n_filters*16)
        self.size_after_conv5 = compute_size(self.size_after_pool4, filter_size, stride=1)
        self.pool5 = nn.MaxPool2d(filter_size,stride=2)
        self.size_after_pool5 = compute_size(self.size_after_conv5, filter_size, stride=2)

        self.drop_out = nn.Dropout(p=drop_out)

        # Fully connected layers
        self.fc1 = nn.Linear(n_filters*16 * (self.size_after_pool5)*(self.size_after_pool5), denseLay)
        self.normFC1 = nn.BatchNorm1d(denseLay)
        self.opLay = nn.Linear(denseLay, 10)

    def _get_activation_function(self, activation):
        # Return the appropriate activation function based on the input argument
        if activation == 'relu':
            return func.relu
        elif activation == 'gelu':
            return func.gelu
        elif activation == 'silu':
            return func.silu
        else:
            return func.mish


    def forward(self, img,activationFun):
        # Define activation function based on the input argument
        activation_fn = self._get_activation_function(activationFun)

        # Apply convolutional layers followed by activation and pooling
        if(self.norm == 'true'):
            img = self.pool1(activation_fn(self.norm1(self.conv1(img))))
        else:
            img = self.pool1(activation_fn(self.conv1(img)))
        if(self.norm == 'true'):
            img = self.pool2(activation_fn(self.norm2(self.conv2(img))))
        else:
            img = self.pool2(activation_fn(self.conv2(img)))
        if(self.norm == 'true'):
            img = self.pool3(activation_fn(self.norm3(self.conv3(img))))
        else:
            img = self.pool3(activation_fn(self.conv3(img)))
        if(self.norm=='true'):
            img = self.pool4(activation_fn(self.norm4(self.conv4(img))))
        else:
            img = self.pool4(activation_fn(self.conv4(img)))
        if(self.norm == 'true'):
            img = self.pool5(activation_fn(self.norm5(self.conv5(img))))
        else:
            img = self.pool5(activation_fn(self.conv5(img)))

        #reshaping
        img=img.reshape(img.shape[0],-1)
        if(self.norm == 'true'):
            img = activation_fn(self.normFC1(self.fc1(img)))
        else:
            img = activation_fn(self.fc1(img))
        img = self.drop_out(img)
        img = self.opLay(img)
        return img

# learning_rate=0.0001
# epochs=10
# n_filters = 128
# filter_size = 3
# denseLay = 64
# activationFun='relu'



LossFun = nn.CrossEntropyLoss()

def train_model(epochs,Network,Algo,activationFun,aug,batch_size):
    train_loader,val_loader = DataLoaders(aug,batch_size)
    for epoch in range(epochs):
        for i,(images,labels) in enumerate(tqdm(train_loader)):

            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = Network(images,activationFun)
            loss = LossFun(outputs, labels)

            # Backward and optimize
            Network.zero_grad()
            loss.backward()
            Algo.step()
        test_acc,test_loss=calculate_accuracy(Network,test_loader,activationFun)
#         train_acc,train_loss = calculate_accuracy(Network,train_loader,activationFun)
#         Validation_acc,val_loss=calculate_accuracy(Network, val_loader,activationFun)

#         wandb.log({"Validation_acc" : Validation_acc})
#         wandb.log({"val_loss" : val_loss})
#         wandb.log({"train_acc" : train_acc})
#         wandb.log({"train_loss" : train_loss})
        print(test_acc,test_loss)
#         print(train_acc,train_loss)
#         print(Validation_acc,val_loss)


#accuracy
def calculate_accuracy(model, data_loader,activationFun):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    num_loss=0
    total_length = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images=images.to(device=device)
            labels=labels.to(device=device)
            outputs = model(images,activationFun)
            loss=LossFun(outputs,labels)
            num_loss+=loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_length += predicted.size(0)
    accuracy = correct / total
    loss=num_loss/total_length
    model.train()
    return accuracy,loss


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')

    parser.add_argument('-wp', '--wandb_project', type=str, default='DL_assignment_2',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-n', '--neurons', type= int, default=64, choices = [128,256],help='Choice of neurons in dense layer')
    
    parser.add_argument('-nF', '--numFilters', type= int, default=32, choices = [32,64,128],help='Choice of number of filters')

    parser.add_argument('-sF', '--sizeFilter', type= int, default=3, choices = [3,5],help='Choice of kernel size')

    parser.add_argument('-aF', '--activFun', type= str, default='relu', choices = ['relu','gelu','silu','mish'],help='Choice of activation function')
    
    parser.add_argument('-opt', '--optimizer', type= str, default='nadam', choices = ['adam','nadam'],help='Choice of optimizer')

    parser.add_argument('-bS', '--batchSize', type= int, default=64, choices = [32,64,128],help='Choice of batch size')

    parser.add_argument('-d', '--dropOut', type= float, default=0.4, choices = [0,0.2,0.4],help='Choice of drop out probability')

    parser.add_argument('-nE', '--epochs', type= int, default=10, choices = [5,10],help='Choice of epochs')

    parser.add_argument('-lR', '--learnRate', type =float, default=1e-4, choices = [1e-3,1e-4],help='Choice of learnRate')

    parser.add_argument('-bN', '--batchNorm', type= str, default='yes', choices = ['yes','no'],help='Choice of batch normalization')

    parser.add_argument('-ag', '--aug', type= str, default='no', choices = ['yes','no'],help='Choice of augumentation')

    parser.add_argument('-o', '--org', type= int, default=1, choices = [0,1],help='Choice of filter organization')

    return parser.parse_args()

args = parse_arguments()
wandb.init(project=args.wandb_project)

wandb.run.name=f'activation {args.activFun}opt{args.optimizer}batchNorm{args.batchNorm}'
choose_optimizer(args.optimizer,args.epochs,args.learnRate,args.numFilters,args.sizeFilter,args.activFun,args.neurons,args.aug,args.batchSize,args.dropOut,args.batchNorm)
                 
