import torch
from torch.hub import load

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)