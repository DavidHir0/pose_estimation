import matplotlib.pyplot as plt
import torchvision
import numpy as np
from tqdm import tqdm
import torch


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
