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
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error

def find_max_value_coordinates(matrix, quotients):
    batch_size, num_keypoints, height, width = matrix.shape
    max_coords = torch.zeros(batch_size, num_keypoints, 2, dtype=torch.float32)

    for i in range(batch_size):
        for j in range(num_keypoints):
            max_value, max_index = torch.max(matrix[i][j].view(-1), dim=0)
            max_y = max_index // width # line
            max_x = max_index % width  #collumn
            max_x = max_x * quotients[0]
            max_y = max_y * quotients[1]
            max_coords[i, j] = torch.tensor([max_x, max_y])

    return max_coords


def gaussian_2d(xy, mu_x, mu_y, sigma_x, sigma_y, rho):
    x, y = xy[0], xy[1]
    z = (x - mu_x) ** 2 / sigma_x ** 2 - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y) + (y - mu_y) ** 2 / sigma_y ** 2
    exponent = -1 / (2 * (1 - rho ** 2)) * z
    constant = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(np.maximum(1 - rho ** 2, 1e-10)))  
    # return (constant * np.exp(exponent, where=(exponent <= 0))).ravel()
    return (constant * np.exp(exponent)).ravel()

def cross_entropy(pred, target):
    return -torch.sum(torch.log(pred + 1e-15) * target) / (pred.shape[0] * pred.shape[1])


def get_MSE_var(pred, label, MSE, fitted_MSE, fitted_gaussian, count, quotients = (1,1)):

    # create mesh gird
    width, height = pred.shape[2:]
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    
    max_idx = find_max_value_coordinates(pred, (1,1)).numpy()

    # make numpy arrays
    label = label.cpu().numpy()
    pred = pred.detach().cpu().numpy()
    

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):

            score_map = pred[i][j]
            pose = label[i][j]
            pred_mean = max_idx[i][j]

            initial_guess = (pred_mean[0], pred_mean[1], 1,1,0)

            score_map_flat = score_map.ravel()


            # cruve fit
            try:
                popt, pcov = opt.curve_fit(gaussian_2d, (X, Y), score_map_flat, p0=initial_guess, maxfev=5000)
            except RuntimeError:
                print(f"Curve fit failed for data at index ({i}, {j}). Skipping...")
                continue
            
            # rescale to original size
            pose[0] /= quotients[0]
            pose[1] /= quotients[1]

            popt[0] *= quotients[0]
            popt[1] *= quotients[1]

            pred_mean[0] *= quotients[0]
            pred_mean[1] *= quotients[1]

            
            
            current_fitted_MSE = mean_squared_error(pose, popt[:2])
            current_MSE = mean_squared_error(pose, pred_mean)

                  

            MSE.append(current_MSE)
            fitted_MSE.append(current_fitted_MSE)
            fitted_gaussian.append(popt)

            count += 1

    return MSE, fitted_MSE, fitted_gaussian, count

            


