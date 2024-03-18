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

dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = static_loader(dirname, batch_size=16, cuda=True)
count = 0
for batch in dataloaders['test']:
    pose = batch.pose_matrix
    count += pose.shape[0] * pose.shape[1]


print(count)