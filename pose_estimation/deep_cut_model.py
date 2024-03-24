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
from torchvision.transforms import transforms

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
resnet = load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

class DeconvHeadModel(nn.Module):
    def __init__(self):
        super(DeconvHeadModel, self).__init__()
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Change stride of last Block
        self.resnet[-1][0].downsample[0].stride = (1, 1)
        self.resnet[-1][0].conv2.stride = (1, 1)

        # Change to ceil_mode
        #self.resnet[3].ceil_mode = True

        # Use Dilated Convolution for all 3x3 Convolutional Layers in Last Block
        for block in self.resnet[-1]:
            for layer in block.children():
                if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
                    layer.dilation = (2, 2)
                    layer.padding = (2, 2)


        self.deconv1 = nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.final_conv = nn.Conv2d(
            in_channels=256,
            out_channels=20, 
            kernel_size=1
        )


    def forward(self, x):
        # Forward pass through the ResNet
        x = self.resnet(x)

        # Forward pass through the deconvolutional layers
        x = self.relu1(self.batchnorm1(self.deconv1(x)))
        x = self.relu2(self.batchnorm2(self.deconv2(x)))
        x = self.relu3(self.batchnorm3(self.deconv3(x)))
        x = self.final_conv(x)

        batch_size, num_channels, height, width = x.size()
        x_reshaped = x.view(batch_size, num_channels, height * width)
        score_maps_reshaped = F.softmax(x_reshaped, dim=2)
        pred_score_maps = score_maps_reshaped.view(batch_size, num_channels, height, width)

        return pred_score_maps

    
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)


    def train_model(self, device, train_loader, optimizer, criterion, validate_loader=None, std=0.05, scheduler=None, num_epochs=10, size=224, data_augmentation=[False, False, False]):
        self.to(device)
        self.train()
        training_loss = []
        training_MSE = []
        validation_loss = []
        validation_MSE = []
        not_improved_count = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            total_samples = 0
            MSE = 0
            batch_number = len(train_loader)

            for param_group in optimizer.param_groups:
                print("Current learning rate:", param_group['lr'])

            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch_Train [{epoch+1}/{num_epochs}]")
                
                for batch in tepoch:
                    images, label_pose = batch.image.to(device), batch.pose_matrix.to(device)

                    # data augmentation [rotation, flip, color jitter]

                    # flip
                    if data_augmentation[1] is True:
                        hflipper = transforms.RandomHorizontalFlip(p=0.5)
                        images = hflipper(images)

                    # colorjitter
                    if data_augmentation[2] is True:
                        images = images.permute(0, 3, 1, 2)
                        color_jitter = transforms.ColorJitter(brightness=.1, hue=.1)
                        images = color_jitter(images)
                        images = images.permute(0,2,3,1)
                        

                    
                    #rotation 
                    if data_augmentation[0] is True:
                        rotation = transforms.RandomRotation(degrees=(-30, 30))
                        images = rotation(images)


                    # Normalize images and 
                    norm_images = normalize_input(images, size=size)


                    optimizer.zero_grad()

                    #get prediction
                    pred_score_maps = self(norm_images)
                    
                    # get scale factors
                    height_q = images.shape[1] / pred_score_maps.shape[2] #for y
                    width_q = images.shape[2] / pred_score_maps.shape[3] # for x

                    # for mse scale to network input size
                    h_scale = norm_images.shape[-2] / pred_score_maps.shape[2]
                    w_scale = norm_images.shape[-1] / pred_score_maps.shape[3]
                    
                    # get COORDS of mean pred
                    
                    pred_coords = self.find_max_value_coordinates(pred_score_maps.clone().detach(), (width_q, height_q)).to(device)
                    
                    # get MSE
                    current_MSE = self.MSE(pred_coords.clone(), label_pose.clone().detach(), (w_scale, h_scale)).item()
                    MSE += current_MSE

                    # create groundtruth mask 
                    label_mask =  create_normalized_gaussian_maps(batch.pose_matrix, pred_score_maps.shape[2], pred_score_maps.shape[3], std, quotients=(width_q, height_q))
                    label_mask = label_mask.to(device)

                    #loss
                    loss = criterion(pred_score_maps, label_mask)
                   

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)

                    tepoch.set_postfix(loss=(loss.item(), current_MSE))

                epoch_loss = running_loss / total_samples
                epoch_MSE = MSE / batch_number
                

                print('\n')
                tqdm.write(f"Epoch Loss: {epoch_loss:.4f}, MSE: {epoch_MSE:.4f}")
                print('\n')
            training_MSE.append(epoch_MSE)
            training_loss.append(epoch_loss)

            

            #Validate
            if validate_loader != None:
                # reset losses for validation
                running_loss = 0.0
                total_samples = 0
                MSE = 0
                batch_number = len(validate_loader)

                with tqdm(validate_loader, unit="batch") as tepoch:
                    tepoch.set_description(f"Epoch_Validate [{epoch+1}/{num_epochs}]")
                    with torch.no_grad():
                        
                        for batch in tepoch:
                            images, label_pose = batch.image.to(device), batch.pose_matrix.to(device)

                            # Normalize images and 
                            norm_images = normalize_input(images, size=size)

                            pred_score_maps = self(norm_images)
                            
                            # get scale factors
                            height_q = images.shape[1] / pred_score_maps.shape[2] #for y
                            width_q = images.shape[2] / pred_score_maps.shape[3] # for x

                            # for mse scale to network input size
                            h_scale = norm_images.shape[-2] / pred_score_maps.shape[2]
                            w_scale = norm_images.shape[-1] / pred_score_maps.shape[3]


                            # get COORDS
                            pred_coords = self.find_max_value_coordinates(pred_score_maps.clone(), (width_q, height_q)).to(device)

                            # get MSE
                            current_MSE = self.MSE(pred_coords.clone(), label_pose.clone().detach(), (w_scale, h_scale)).item()
                            MSE += current_MSE

                            # create groundtruth mask 
                            label_mask =  create_normalized_gaussian_maps(batch.pose_matrix, pred_score_maps.shape[2], pred_score_maps.shape[3], std, quotients=(width_q, height_q))
                            label_mask = label_mask.to(device)

                            #loss
                            loss = criterion(pred_score_maps, label_mask)
                           
                            running_loss += loss.item() * images.size(0)
                            total_samples += images.size(0)

                            tepoch.set_postfix(loss=(loss.item(), current_MSE))

                        epoch_val_loss = running_loss / total_samples
                        epoch_val_MSE = MSE / batch_number
                
                # tepoch.set_postfix(loss=f"Epoch Val Loss: {epoch_val_loss:.4f}, MSE Val: {epoch_val_MSE:.4f}")
                print('\n')
                tqdm.write(f"Epoch Loss: {epoch_val_loss:.4f}, MSE: {epoch_val_MSE:.4f}")
                print('\n')

            if scheduler:
                # plateou scheduler
                scheduler.step(epoch_val_loss)

            validation_MSE.append(epoch_val_MSE)
            validation_loss.append(epoch_val_loss)

            # check if loss didnt decerase
            if epoch > 0 and validation_loss[-1] >= validation_loss[-2]:
                not_improved_count += 1
            else:
                not_improved_count = 0
            
            if not_improved_count >= 3:
                print("Validation loss hasn't improved for 3 consecutive epochs. Stopping training.")
                return training_loss, training_MSE, validation_loss, validation_MSE
        
        print('Training complete.')
        return training_loss, training_MSE, validation_loss, validation_MSE


    def find_max_value_coordinates(self, matrix, quotients):
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

    def MSE(self, tensor1, tensor2, scale=(1,1)):
        tensor1[:,:,0] /= scale[0]
        tensor1[:,:,1] /= scale[1]
        tensor2[:,:,0] /= scale[0]
        tensor2[:,:,1] /= scale[1]

        squared_diff = (tensor1 - tensor2)**2
        mse = torch.mean(squared_diff)
        return mse