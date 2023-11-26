import torch
from torch.hub import load

loaded_model = None

def load_resnet50():
    global loaded_model

    if loaded_model is None:
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
    
    return loaded_model
