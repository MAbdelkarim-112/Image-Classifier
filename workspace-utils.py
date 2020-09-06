import argparse

import torch
from torchvision import transforms, datasets

import json

# TODO: Save the checkpoint 
def save_checkpoint(model):
    

    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoint = {'input_size': 1024,
              'output_size': 102,
              'arch': 'vgg16',
              'classifier' : classifier,
              'learning_rate': 0.001,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
             }

    torch.save(checkpoint, 'checkpoint.pth')
    
save_checkpoint(model)
        
# TODO: Write a function that loads a checkpoint and rebuilds the model
    def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    #learning_rate = checkpoint['learning_rate']
    #model = getattr(torchvision.models, checkpoint['structure'])(pretrained = True)
    #model.epochs = checkpoint['epochs']
    #model.load_state_dict(checkpoint['state_dict'])
    #model.class_to_idx = checkpoint['class_to_idx']
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = checkpoint['classifier']

        model.load_state_dict(checkpoint['state_dict'])
    return model

# Load class_to_name json file 
def load_json(json_file):
    
    with open(json_file, 'r') as f:
        flower_to_name = json.load(f)
        return flower_to_name