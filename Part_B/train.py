import torchvision.models as models
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

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])])

train_dataset = ImageFolder(root='inaturalist_12K/train', transform=transform)
# test_dataset = ImageFolder(root='/kaggle/input/inaturalist-dataset/inaturalist_12K/val', transform=transform)


# Define the size of the validation set (20%)
validation_size = 0.2

# Split the dataset into training and validation sets with shuffling
train_data, val_data = train_test_split(list(range(len(train_dataset))), test_size=validation_size, shuffle=True, random_state=42)
#Data Loader

train_subset = Subset(train_dataset, train_data)
val_subset  = Subset(train_dataset, val_data)

def DataLoaders(aug,batch_size):
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    return train_loader,val_loader


def RESNET50(k,NUM_OF_CLASSES): #this function returns the model by freezing first k layers only
    model = models.resnet50(pretrained=True)

    params = list(model.parameters())
    for param in params[:k]:
        param.requires_grad = False #freezing

    num_ftrs = model.fc.in_features

    model.fc = torch.nn.Linear(num_ftrs, NUM_OF_CLASSES)

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # integrating with gpu

def train_model(epochs,activationFun,aug,batch_size,eta):
    LossFun = nn.CrossEntropyLoss()
    train_loader,val_loader = DataLoaders(aug,batch_size)
    Network = RESNET50(50,10).to(device)
    Algo = torch.optim.Adam(Network.parameters(), lr=eta)
    for epoch in range(epochs):
        for i,(images,labels) in enumerate(tqdm(train_loader)):

            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = Network(images)
            loss = LossFun(outputs, labels)

            # Backward and optimize
            Network.zero_grad()
            loss.backward()
            Algo.step()
        train_acc,train_loss = calculate_accuracy(Network,train_loader,activationFun)
        Validation_acc,val_loss=calculate_accuracy(Network, val_loader,activationFun)
        # wandb.log({"Validation_acc" : Validation_acc})
        # wandb.log({"val_loss" : val_loss})
        # wandb.log({"train_acc" : train_acc})
        # wandb.log({"train_loss" : train_loss})
        print(train_acc,train_loss)
        print(Validation_acc,val_loss)


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
            outputs = model(images)
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

    parser.add_argument('-wp', '--wandb_project', type=str, default='DL_assignment_2_B',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-bS', '--batchSize', type= int, default=32, choices = [32,64],help='Choice of batch size')
    
    parser.add_argument('-e', '--epochs', type= int, default=20, choices = [10,20,30],help='Number of epochs')

    parser.add_argument('-lE', '--learnRate', type= float, default=1e-4, choices = [1e-3,1e-4],help='Learning rates')

    parser.add_argument('-ag', '--aug', type= str, default='no', choices = ['yes','no'],help='Augumentation choices')
    
    parser.add_argument('-s', '--strategy', type= int, default=1, choices = [0,1,2],help='Choice of strategies')

    return parser.parse_args()

args = parse_arguments()
wandb.init(project=args.wandb_project)

wandb.run.name=f'strategy {args.strategy}'
train_model(args.epochs,'relu',args.aug,args.batchSize,args.learnRate)

