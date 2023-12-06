from pose_estimation.data_pp.loaders import static_loader
from propose.datasets.rat7m.Rat7mDataset import Rat7mDataset
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


def calculate_mean_std(dataloader, key, name):
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

    np.save(f'data/rat7m/{key}/{name}' + '_mean.npy', mean.cpu().numpy())
    np.save(f'data/rat7m/{key}/{name}' + '_std.npy', std.cpu().numpy())




dirname = "data/rat7m/s2-d1"

dataloaders = static_loader(dirname, batch_size=16, cuda=True)

# print(len(dataloaders["train"]))
# print(len(dataloaders["test"]))
# print(len(dataloaders["validation"]))
#calculate_mean_std(dataloaders["validation"], "s2-d1", "validation")
#print(calculate_mean_std(dataloaders["train"]))



mean = np.load('data/rat7m/s2-d1/train_mean.npy')
std = np.load('data/rat7m/s2-d1/train_std.npy')
# print(mean, std)

for i in dataloaders["train"]:
    # Assuming 'i.image' is your tensor data
    im = i.image.detach().cpu()  # Detach tensor and move to CPU if necessary
    print(i.image.shape)
    print(i.pose_matrix.shape)
    print(i.adjacency_matrix.shape)
    print(make_kp_maps(i.pose_matrix, i.image.shape[1], i.image.shape[2], 10).shape)
    heat_map = make_kp_maps(i.pose_matrix, i.image.shape[2], i.image.shape[1], 100)

    for im, pose in zip(i.image, i.pose_matrix):
        image = im.cpu().detach().numpy()

        normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
        normalized_image = normalized_image.astype(np.uint8)

        # Plot the image
        plt.imshow(heat_map[0][0], cmap='gray')  
        #plt.imshow(normalized_image, cmap='gray')  
        plt.axis('off')

        plt.scatter(pose[0, 0].cpu(), pose[0, 1].cpu(), color='red', marker='.') 

        plt.savefig(f'image_with_pose.png', dpi=300)
        plt.show() 

        break  

    break

