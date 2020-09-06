import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict


import json
import argparse


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 512, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 6, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')

arguments = parser.parse_args()

# Image data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms for the training, validation, and testing sets
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
image_datasets = [datasets.ImageFolder(train_dir, transform=training_transforms), 
                  datasets.ImageFolder(valid_dir, transform=validation_transforms),
                  datasets.ImageFolder(test_dir, transform=testing_transforms)]


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]


# Build and train the neural network (Transfer Learning)
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
else:
    model = models.vgg13(pretrained=True)
    
print(model)

# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, arguments.hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(arguments.hidden_units, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

# Loss function (since the output is LogSoftmax, we use NLLLoss)
criterion = nn.NLLLoss()

# Gradient descent optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
    
# Train the classifier

def train_classifier(model, optimizer, criterion, arguments.epochs, train_loader, validate_loader, arguments.gpu):
        train_loader=dataloaders[0]
        validate_loader = dataloaders[1]
        epochs = 6
        steps = 0
        print_every = 40

        model.to('cuda')

        for e in range(epochs):
        
            model.train()
    
            running_loss = 0
    
            for images, labels in iter(dataloaders[0]):
        
                steps += 1
        
                images, labels = images.to('cuda'), labels.to('cuda')
        
                optimizer.zero_grad()
        
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
                if steps % print_every == 0:
                
                    model.eval()
                
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        validation_loss, accuracy = validation(model, dataloaders[1], criterion)
            
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(dataloaders[1])),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders[1])))
            
                    running_loss = 0
                    model.train()
                    
train_classifier()

# TODO: Do validation on the test set
test_loader= dataloaders[2]
def test_accuracy(model, test_loader):
# Do validation on the test set
    model.eval()
    model.to('cuda')

    with torch.no_grad():
    
        accuracy = 0
    
        for images, labels in iter(dataloaders[2]):
    
            images, labels = images.to('cuda'), labels.to('cuda')
    
            output = model.forward(images)

            probabilities = torch.exp(output)
        
            equality = (labels.data == probabilities.max(dim=1)[1])
        
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(dataloaders[2]))) 
        
