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

def find_max_value_coordinates(matrix, quotients):
    batch_size, num_keypoints, height, width = matrix.shape
    max_coords = torch.zeros(batch_size, num_keypoints, 2, dtype=torch.float32)

    for i in range(batch_size):
        for j in range(num_keypoints):
            max_value, max_index = torch.max(matrix[i][j].view(-1), dim=0)
            max_x = max_index // width
            max_y = max_index % width
            max_x = max_x * quotients[0]
            max_y = max_y * quotients[1]
            max_coords[i, j] = torch.tensor([max_y, max_x])

    return max_coords


dirname = "data/rat7m/s2-d1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloaders = static_loader(dirname, batch_size=16, cuda=True)

safe = "cropped_6_wd/wd_0.0001"

model = KpDetector()
model.load_state_dict(torch.load(f'results/{safe}/model_parameters.pt'))
model.eval()
model.to(device)

# criterion = KL_divergence
criterion = cross_entropy





def optimize_temperature(model, dataloader, temperature, optimizer, device, criterion, num_epochs=5):
    model.eval()
    temperature.to(device)
    for epoch in range(num_epochs):
        total_loss = 0.0
        tqdm_dataloader = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')

        for batch in tqdm_dataloader:
            images, label_pose = batch.image.to(device), batch.pose_matrix.to(device)

            norm_images = normalize_input(images)

            outputs = model(norm_images)

            height_q = images.shape[1] / outputs.shape[2] #for x
            width_q = images.shape[2] / outputs.shape[3] # for y

            # create groundtruth mask 
            # label_mask =  create_normalized_gaussian_maps(label_pose, outputs.shape[2], outputs.shape[3], 0.04, quotients=(width_q, height_q))
            # label_mask = label_mask.to(device)

            optimizer.zero_grad()

            batch_size, num_channels, height, width = outputs.size()
            x_reshaped = outputs.view(batch_size, num_channels, height * width)
            x_reshaped = x_reshaped / temperature 
            score_maps_reshaped = F.softmax(x_reshaped, dim=2)
            score_maps = score_maps_reshaped.view(batch_size, num_channels, height, width)

            nll = 0
            label_pose[:, :, 0] /= height_q
            label_pose[:, :, 1] /= width_q
            

            count = 0
            for i, poses in enumerate(label_pose):
                poses = torch.round(poses)
                for j, pose in enumerate(poses):
                    if (int(pose[0]) < score_maps.shape[2]) and (int(pose[1]) < score_maps.shape[3]):
                        nll += torch.log(score_maps[i][j][int(pose[0])][int(pose[1])])
                        count -= 1

            nll /= count
            #loss = #einfüge
            loss = nll
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tqdm_dataloader.set_postfix(loss=f'{loss.item():.4f}', temperature=f'{temperature.item():.4f}')

        average_loss = total_loss / len(dataloader)
        tqdm_dataloader.set_postfix(loss=f'{average_loss:.4f}')
        tqdm_dataloader.close()
        return temperature.cpu().detach().numpy()



initial_temperature = 1.0
temperature = nn.Parameter(torch.tensor(initial_temperature, requires_grad=True))  # Initial temperature
optimizer = optim.Adam([temperature], lr=0.01)


temp = optimize_temperature(model, dataloaders["test"], temperature, optimizer, device, KL_divergence, num_epochs=5)

np.save(f"results/{safe}/temperature.npy", temp)
print(temp)


