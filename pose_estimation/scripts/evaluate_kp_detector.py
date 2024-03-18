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
def KL_divergence(pred, target):
    return torch.sum(torch.log((target + 1e-15) / (pred + 1e-15)) * target) / (pred.shape[0] * pred.shape[1])

dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = static_loader(dirname, batch_size=16, cuda=True)

load = "cropped_6_wd/wd_0.0001"
safe = "cropped_6_wd/wd_0.0001/new_data_test/t=0.4/s2-d1"


model = KpDetector()
model.load_state_dict(torch.load(f'results/{load}/model_parameters.pt'))
model.eval()

# training_loss = np.load(f'results/{load}/training_loss.npy')
# temp = np.load(f'results/{load}/temperature.npy')

# criterion = KL_divergence
criterion = cross_entropy

# preds = model.evaluate_kp_model(device, dataloaders["test"], criterion, temperature=1.0, std=0.04)
# np.save(f"results/{safe}/pred_pairs.npy", preds)

#evaluate_loss, best_pred, best_images, best_pose, worst_pred, worst_images, worst_pose, MSE, ece, MSE_list, var_list = model.evaluate_kp_model(device, dataloaders["test"], criterion, temperature=temp, std=0.04)


t = 0.4
t_arr = [0.4, 1.0]
# for t in np.arange(0.1, 0.4, 0.1):
    #     print(t)
for t in t_arr:
    count, outlier_count, point_count, MSE, MSE_count, loss, likelihood, MSE_list, likelihood_list = model.evaluate_kp_model(device, dataloaders["test"], criterion, temperature=t, std=0.04)
    np.save(f"results/{safe}/temp={t}_outlier_count.npy", outlier_count)
    np.save(f"results/{safe}/temp={t}_bin_count.npy", count)
    np.save(f"results/{safe}/temp={t}_data_point_count.npy", point_count)
    np.save(f"results/{safe}/temp={t}_MSE.npy", MSE)
    np.save(f"results/{safe}/temp={t}_MSE_count.npy", MSE_count)
    np.save(f"results/{safe}/temp={t}_ce_loss.npy", loss)
    np.save(f"results/{safe}/temp={t}_likelihood.npy", likelihood)
    np.save(f"results/{safe}/temp={t}_MSE_list.npy", MSE_list)
    np.save(f"results/{safe}/temp={t}_likelihood_list.npy", likelihood_list)






dirname = "data/rat7m/s3-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = static_loader(dirname, batch_size=16, cuda=True)

load = "cropped_6_wd/wd_0.0001"
safe = "cropped_6_wd/wd_0.0001/new_data_test/t=0.4/s3-d1"


model = KpDetector()
model.load_state_dict(torch.load(f'results/{load}/model_parameters.pt'))
model.eval()

criterion = cross_entropy


for t in t_arr:
    count, outlier_count, point_count, MSE, MSE_count, loss, likelihood, MSE_list, likelihood_list = model.evaluate_kp_model(device, dataloaders["test"], criterion, temperature=t, std=0.04)
    np.save(f"results/{safe}/temp={t}_outlier_count.npy", outlier_count)
    np.save(f"results/{safe}/temp={t}_bin_count.npy", count)
    np.save(f"results/{safe}/temp={t}_data_point_count.npy", point_count)
    np.save(f"results/{safe}/temp={t}_MSE.npy", MSE)
    np.save(f"results/{safe}/temp={t}_MSE_count.npy", MSE_count)
    np.save(f"results/{safe}/temp={t}_ce_loss.npy", loss)
    np.save(f"results/{safe}/temp={t}_likelihood.npy", likelihood)
    np.save(f"results/{safe}/temp={t}_MSE_list.npy", MSE_list)
    np.save(f"results/{safe}/temp={t}_likelihood_list.npy", likelihood_list)




























# for t in np.arange(0.6, 1.6, 0.2):
#     count, outlier_count, point_count = model.evaluate_kp_model(device, dataloaders["test"], criterion, temperature=t, std=0.04)
#     np.save(f"results/{safe}/{t}outlier_count.npy", outlier_count)
#     np.save(f"results/{safe}/{t}bin_count.npy", count)
#     np.save(f"results/{safe}/{t}data_point_count.npy", point_count)




# for t in np.arange(1.0, 2.1, 0.1):
#     z = t
#     ece, outlier_count,  bin_accuracy, bin_confidence, bin_count = model.evaluate_kp_model(device, dataloaders["test"], criterion, temperature=t, std=0.04)
#     bins = np.vstack((bin_accuracy, bin_confidence, bin_count))
#     np.save(f"results/{safe}/bins_{z}.npy", bins)
#     np.save(f"results/{safe}/ECE_{z}.npy", ece)
#     np.save(f"results/{safe}/outlier_count_{z}.npy", outlier_count)
    # , best_pred, best_images, best_pose, worst_pred, worst_images, worst_pose
    # np.save(f"results/{safe}/best_prediction_{z}.npy", best_pred)
    # np.save(f"results/{safe}/best_images_{z}.npy", best_images)
    # np.save(f"results/{safe}/best_pose_{z}.npy", best_pose)
    # np.save(f"results/{safe}/worst_prediction_{z}.npy", worst_pred)
    # np.save(f"results/{safe}/worst_images_{z}.npy", worst_images)
    # np.save(f"results/{safe}/worst_pose_{z}.npy", worst_pose)

# np.save(f"results/{safe}/evaluate_loss.npy", evaluate_loss)
# np.save(f"results/{safe}/best_prediction.npy", best_pred)
# np.save(f"results/{safe}/best_images.npy", best_images)
# np.save(f"results/{safe}/best_pose.npy", best_pose)
# np.save(f"results/{safe}/worst_prediction.npy", worst_pred)
# np.save(f"results/{safe}/worst_images.npy", worst_images)
# np.save(f"results/{safe}/worst_pose.npy", worst_pose)
# np.save(f"results/{safe}/MSE.npy", MSE)
# t = "1"
# np.save(f"results/{safe}/ECE.npy", ece)
# np.save(f"results/{safe}/outlier_count.npy", outlier_count)
# np.save(f"results/{safe}/all_MSE.npy", MSE_list)
# np.save(f"results/{safe}/all_var.npy", var_list)
print(f"{safe}")
