import torch
import numpy as np
import random
from tqdm import tqdm
from pose_estimation.deep_cut_model import DeconvHeadModel
import matplotlib.pyplot as plt
from pose_estimation.data_pp.loaders import static_loader
from pose_estimation.data_pp.utils import create_normalized_gaussian_maps, normalize_input
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(pred, target):
    return -torch.sum(torch.log(pred + 1e-15) * target) / (pred.shape[0] * pred.shape[1])

safe = "simple_architecture/end_results/5e_0.1lr_0.1plat"

dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

dataloaders =  static_loader(dirname, batch_size=16, cuda=True)

model = DeconvHeadModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)#, weight_decay=0.0001

criterion = cross_entropy

# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2,3], gamma=0.1)
scheduler =  lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

# std = 0.015 # for 224x224 maps radius of 10px in original image

#std = 0.025 # for 224x224 maps radius of 15px in original image

std = 0.04 # for 112x112 maps, 50% in 10px range, 99% in 30px range

# std = 0.02 # for 112x112 maps 50% in 5px range, 99% in 14px range

epochs = 5

#input size of network
size = 224

#seed for reproducabilty
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

loss, MSE, val_loss, val_MSE = model.train_model(device, dataloaders["train"], optimizer, criterion, validate_loader=dataloaders["validation"], std=std, scheduler=scheduler, num_epochs=epochs, size=size)

model.save_model(f"results/{safe}/model_parameters.pt")

np.save(f'results/{safe}/training_loss.npy', loss)
np.save(f'results/{safe}/training_MSE.npy', MSE)
np.save(f'results/{safe}/validation_loss.npy', val_loss)
np.save(f'results/{safe}/validation_MSE.npy', val_MSE)