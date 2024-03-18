import torch
from torch import nn
from torch.hub import load
from pose_estimation.data_pp.utils import create_normalized_gaussian_maps, normalize_input
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error
from torchmetrics.classification import MulticlassCalibrationError
import scipy.stats as st
from torch import optim
from pose_estimation.data_pp.loaders import static_loader
import matplotlib.pyplot as plt



dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

dataloaders =  static_loader(dirname, batch_size=16, cuda=True)

for i, batch in enumerate(dataloaders['train']):
    pose = batch.pose_matrix
    im = batch.image
    image = im.cpu().numpy()
    # print(im.shape)
    # pose[0][0][0] = 50
    # print(pose[0][0])
    # pose[0][0][1] = 100
    im = normalize_input(im, size=(448,448))
    
    # print(im.shape)
    
    pose /= 4
    map = create_normalized_gaussian_maps(pose, 112, 112, 0.02)
    break
    print(i, im.shape)

plt.imshow(map[0][0], cmap='viridis')
plt.savefig("test.png")

plt.close()
contour_level = 0.0565
contour_level = 0.001

# Threshold the heatmap to create a binary mask
binary_mask = (map[0][0].numpy() >= contour_level).astype(np.uint8)

# Count the number of pixels in the binary mask
print(torch.sum(map[0][0][binary_mask]))
area_pixels = np.sum(binary_mask)
print(area_pixels)
print(map[0][0][0][0]>contour_level)


image = image[0]
print(image.shape)
normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
normalized_image = normalized_image.astype(np.uint8)
plt.imshow(normalized_image)
pose = pose.cpu().numpy()
pose *= 4
plt.scatter(pose[0][0][0], pose[0][0][1], color='blue')
plt.scatter(50, 100, color='red')
plt.savefig("test1.png")