import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
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

def gaussian_2d(xy, mu_x, mu_y, sigma_x, sigma_y, rho):
    x, y = xy[0], xy[1]
    z = (x - mu_x) ** 2 / sigma_x ** 2 - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y) + (y - mu_y) ** 2 / sigma_y ** 2
    exponent = -1 / (2 * (1 - rho ** 2)) * z
    constant = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(np.maximum(1 - rho ** 2, 1e-10)))  
    return (constant * np.exp(exponent)).ravel()

def gaussian_2d2(xy, mu_x, mu_y, sigma_x, sigma_y, rho):
    x, y = xy[0], xy[1]
    z = (x - mu_x) ** 2 / sigma_x ** 2 - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y) + (y - mu_y) ** 2 / sigma_y ** 2
    exponent = -1 / (2 * (1 - rho ** 2)) * z
    constant = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(np.maximum(1 - rho ** 2, 1e-10)))  
    return (constant * np.exp(exponent, where=(exponent <= 0))).ravel()


def find_max_value_coordinates(matrix):
    max_value = float('-inf')
    max_coords = None
    width = matrix.shape[0]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value > max_value:
                max_value = value
                max_coords = (i, j)

    matrix = torch.tensor(matrix)
    max_value, max_index = torch.max(matrix.view(-1), dim=0)
    max_y = max_index // width
    max_x = max_index % width
    return (max_x, max_y)


dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

dataloaders =  static_loader(dirname, batch_size=16, cuda=True)

for batch in dataloaders["train"]:
    image = batch.image[0].cpu().numpy()
    poses = batch.pose_matrix[0].cpu().numpy()

    #'gaussian'
    map = create_normalized_gaussian_maps(batch.pose_matrix, image.shape[0], image.shape[1], 1)
    print(map.sum())
    width, height = (image.shape[0], image.shape[0])
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    map2 = gaussian_2d((X,Y), poses[0][0], poses[0][1], 1, 1, 0).reshape(X.shape)
    print(map2.sum())
    map2 = gaussian_2d2((X,Y), poses[0][0], poses[0][1], 2, 2, 0).reshape(X.shape)
    print(map2.sum())
    
    # image
    normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
    normalized_image = normalized_image.astype(np.uint8)
    plt.imshow(normalized_image)
    plt.scatter(poses[0, 0], poses[0, 1], marker='.', color='red')
    plt.scatter(50, 10, marker='.', color='red')
    plt.savefig("test")
    plt.close()

    #heat
    print(poses[0])
    point = find_max_value_coordinates(map.cpu().numpy()[0][0])
    print(point)
    plt.imshow(map.cpu().numpy()[0][0], cmap='viridis')
    plt.scatter(point[0], point[1], color='red')
    plt.savefig("test1")
    plt.close()

    point = find_max_value_coordinates(map2)
    print(point)
    plt.imshow(map2, cmap='viridis')
    # plt.scatter(point[0], point[1], color='red', marker='.')
    plt.savefig("test2")

    print(gaussian_2d((point[0], point[1]), poses[0][0], poses[0][1], 1, 1, 0))
    print(gaussian_2d((poses[0][0], poses[0][1]), poses[0][0], poses[0][1], 10, 10, 0))
    print(gaussian_2d2((poses[0][0], poses[0][1]), poses[0][0], poses[0][1], 10, 10, 0))

    break








