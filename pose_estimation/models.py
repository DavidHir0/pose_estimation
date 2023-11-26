import torch
from torch.hub import load

loaded_model = None

def load_resnet50():
    global loaded_model

    if loaded_model is None:
        loaded_model = load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    
    return loaded_model
