import torch
import numpy as np
from tqdm import tqdm
from pose_estimation.deep_cut_model import DeconvHeadModel
import matplotlib.pyplot as plt
from pose_estimation.data_pp.loaders import static_loader
from pose_estimation.data_pp.utils import create_normalized_gaussian_maps, normalize_input
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import pose_estimation.utils as utils
import random


# load and safe path
load = "results/simple_architecture/end_results/normal"
safe = "results/simple_architecture/end_results/normal"

# path to dataset
dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = static_loader(dirname, batch_size=16, cuda=True)

std = 0.04

criterion = utils.individual_cross_entropy

# load models
model = DeconvHeadModel()
model.load_state_dict(torch.load(f"{load}/model_parameters.pt"))
model.eval()
model.to(device)

#seed for reproducabilty
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# initialize variables
MSE = []
fitted_MSE = []
fitted_gaussian = []
loss = []
running_loss = 0


# evaluation loop
with torch.no_grad():
    with tqdm(dataloaders["test"], unit="batch") as tepoch:

        tepoch.set_description(f"Evaluation: ")

        for batch in tepoch:

            # get image and gt-positions
            images, label_pose = batch.image.to(device), batch.pose_matrix.to(device).clone()

            # normalize
            norm_images = normalize_input(images, size=224)

            # predict
            pred_score_maps = model(norm_images)

            # scale for original image
            
            height_q = images.shape[1] / pred_score_maps.shape[2]
            width_q = images.shape[2] / pred_score_maps.shape[3]

            # for mse scale to network input size
            h_scale = norm_images.shape[-2] / pred_score_maps.shape[2]
            w_scale = norm_images.shape[-1] / pred_score_maps.shape[3]


            # create gt_heatmap
            label_mask = create_normalized_gaussian_maps(label_pose.clone(), pred_score_maps.shape[2], pred_score_maps.shape[3], std, quotients=(width_q, height_q))
            label_mask = label_mask.to(device)

            # calculate loss
            current_loss = criterion(pred_score_maps, label_mask)
            loss += current_loss.tolist()
            running_loss += torch.sum(current_loss).item()

            

            # get MSE and fitted gaussinas parameters

            MSE, fitted_MSE, fitted_gaussian = utils.get_MSE_var(pred_score_maps.clone(), label_pose.clone(), MSE, fitted_MSE, fitted_gaussian, (w_scale, h_scale))



            tepoch.set_postfix(loss=(f"{running_loss/len(loss):.3f}, {sum(MSE)/len(MSE):.3f}"))
            
# save loss and count for loss
np.array(loss)
np.array(MSE)
np.array(fitted_MSE)
np.array(fitted_gaussian)

np.save(f"{safe}/loss.npy", loss)

np.save(f"{safe}/MSE.npy", MSE)       
np.save(f"{safe}/fitted_MSE.npy", fitted_MSE)   
np.save(f"{safe}/fitted_gaussian.npy", fitted_gaussian)   



