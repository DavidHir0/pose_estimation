import torch
import numpy as np
from tqdm import tqdm
from pose_estimation.models import KpDetector
import matplotlib.pyplot as plt
from pose_estimation.data_pp.loaders import static_loader
from pose_estimation.data_pp.utils import create_normalized_gaussian_maps, normalize_input
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(pred, target):
    return -torch.sum(torch.log(pred + 1e-15) * target) / (pred.shape[0] * pred.shape[1])

safe = "cropped_9_10_epochs/2"

dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

dataloaders =  static_loader(dirname, batch_size=16, cuda=True)

model = KpDetector()

optimizer = optim.SGD(model.parameters(), lr=0.1)#, weight_decay=0.0001

criterion = cross_entropy
rgs = 0

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)

loss, MSE, val_loss, val_MSE = model.train_kp_model(device, dataloaders["train"], optimizer, criterion, validate_loader=dataloaders["validation"], std=1, reg_strength=rgs, scheduler=scheduler, num_epochs=10)

model.save_model(f"results/{safe}/model_parameters.pt")

np.save(f'results/{safe}/training_loss.npy', loss)
np.save(f'results/{safe}/training_MSE.npy', MSE)
np.save(f'results/{safe}/validation_loss.npy', val_loss)
np.save(f'results/{safe}/validation_MSE.npy', val_MSE)