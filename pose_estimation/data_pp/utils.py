import matplotlib.pyplot as plt
import torchvision
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image


def make_kp_maps(joints, size_x, size_y, radius):
    batch_size, num_joints, _ = joints.shape
    
    y_coords, x_coords = torch.meshgrid(torch.arange(size_y), torch.arange(size_x))
    
    kp_maps = []
    for frame in joints:
        frame_kp_maps = []
        for joint in frame:
            center_x = joint[0].cpu()
            center_y = joint[1].cpu()
            
            distances = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
            
            mask = (distances <= radius ** 2).float()
            frame_kp_maps.append(mask.unsqueeze(0))

        frame_kp_maps = torch.cat(frame_kp_maps, dim=0)
        kp_maps.append(frame_kp_maps)

    kp_maps_tensor = torch.stack(kp_maps, dim=0)
    return kp_maps_tensor 


def get_mean_std(dataloader, key, name):
    mean_filename = f'data/rat7m/{key}/{name}_mean.npy'
    std_filename = f'data/rat7m/{key}/{name}_std.npy'

    if os.path.exists(mean_filename) and os.path.exists(std_filename):
        mean = torch.from_numpy(np.load(mean_filename))
        std = torch.from_numpy(np.load(std_filename))
    else:
        mean = 0.0
        std = 0.0
        n = len(dataloader)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch.image
                mean += images.mean((0, 1, 2))  
                std += images.std((0, 1, 2))

            mean /= n
            std /= n

        mean = mean.cpu().numpy()
        std = std.cpu().numpy()

        np.save(mean_filename, mean)
        np.save(std_filename, std)

    return mean, std

def calculate_offset(scoremap, groundtruth):
    _, indices = torch.max(scoremap.view(scoremap.shape[0], scoremap.shape[1], -1), dim=2)
    indices = torch.stack((indices // scoremap.shape[3], indices % scoremap.shape[3]), dim=2)

    offsets = gt_coordinates - indices
    
    return offsets


def normalize_input(x, size=(224,224)):
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = x.permute(0, 3, 1, 2)
    #x = x.permute(2, 0, 1)
    preprocessed_tensor = preprocess(x)
    #preprocessed_tensor = preprocessed_tensor.permute(0, 2, 3, 1)

    return preprocessed_tensor





def create_normalized_gaussian_maps(poses, width, height, std, quotients=(1.0, 1.0)):
    batch_size, num_keypoints, _ = poses.shape
    heat_maps = torch.zeros((batch_size, num_keypoints, height, width), dtype=torch.float32)
    poses[:, :, 0] /= quotients[0] # for x
    poses[:, :, 1] /= quotients[1]  # for y
    poses = poses.cpu()

    for i in range(batch_size):
        for j in range(num_keypoints):
            center = (poses[i, j, 0], poses[i, j, 1])
            x = torch.linspace(-1, 1, width)
            y = torch.linspace(-1, 1, height)
            xv, yv = torch.meshgrid(x, y)

            # Calculate the Gaussian function
            gaussian = torch.exp(-((xv + (width / 2 - center[1]) / width * 2)**2 +
                                   (yv + (height / 2 - center[0]) / height * 2)**2) / (2 * std**2))
            gaussian_normalized = gaussian / torch.sum(gaussian)
            heat_maps[i, j, :, :] = gaussian_normalized
    return heat_maps


