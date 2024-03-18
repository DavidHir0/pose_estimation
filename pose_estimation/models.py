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

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
resnet = load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

class KpDetector(nn.Module):
    def __init__(self):
        super(KpDetector, self).__init__()
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


        # self.deconv = nn.ConvTranspose2d(
        #     in_channels=2048,
        #     out_channels=20,
        #     kernel_size=4,
        #     stride=2,
        #     padding=1
        # )
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=1024,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=20,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # self.conv_double_channels = nn.Conv2d(
        #     in_channels=512,  
        #     out_channels=1024,  
        #     kernel_size=1, 
        #     stride=1,
        #     padding=0
        # )


        # Location refinement
        # self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        # self.fc = nn.Linear(500, 40)


    def forward(self, x):
        # conv3_output = None
        # for module in self.resnet:
        #     x = module(x)
            
        #     if module == self.resnet[5]:  
        #         conv3_output = x
        # x = self.resnet(x)
        # x = self.deconv(x)
        # conv3_output = self.conv_double_channels(conv3_output)
        x = self.resnet(x)
        x = self.deconv1(x)
        # x = x + conv3_output
        x = self.deconv2(x)
        x = self.deconv3(x)
        #score_maps = F.softmax(x, dim=1)
        # batch_size, num_channels, height, width = x.size()
        
        #for t scaling removerredw!!!!!!!!!!!!!

        # x_reshaped = x.view(batch_size, num_channels, height * width)
        # score_maps_reshaped = F.softmax(x_reshaped, dim=2)
        # score_maps = score_maps_reshaped.view(batch_size, num_channels, height, width)


        #score_maps = F.sigmoid(x)
        # softmax for distribution
        
        # offsets = self.avgpool(x)
        # offsets = offsets.flatten(start_dim=1)
        # offsets = self.fc(offsets)
        # offsets = offsets.view(-1, 20, 2) 

        return x #for not plat scaling: score_maps      #, offsets


    def l1_regularization_loss(self):
        l1_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return l1_loss


    # TRAIN KEYPOINT DETECTOR
    def train_kp_model(self, device, train_loader, optimizer, criterion_score, validate_loader=None, std=0.05, reg_strength=0, criterion_offset=None, scheduler=None, num_epochs=10):
        self.to(device)
        self.train()
        training_loss = []
        training_MSE = []
        validation_loss = []
        validation_MSE = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            total_samples = 0
            MSE = 0
            batch_number = len(train_loader)

            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch_Train [{epoch+1}/{num_epochs}]")
                
                for batch in tepoch:
                    images, label_pose = batch.image.to(device), batch.pose_matrix.to(device)

                    # Normalize images and 
                    norm_images = normalize_input(images)


                    optimizer.zero_grad()

                    pred_score_maps = self(norm_images)

                    # get scale factors
                    height_q = images.shape[1] / pred_score_maps.shape[2] #for x
                    width_q = images.shape[2] / pred_score_maps.shape[3] # for y

                      # get COORDS
                    pred_coords = self.find_max_value_coordinates(pred_score_maps.clone().detach(), (width_q, height_q)).to(device)

                    # get MSE
                    current_MSE = self.MSE(pred_coords, label_pose.clone().detach()).item()
                    MSE += current_MSE

                    # create groundtruth mask 
                    label_mask =  create_normalized_gaussian_maps(batch.pose_matrix, pred_score_maps.shape[2], pred_score_maps.shape[3], std, quotients=(width_q, height_q))
                    label_mask = label_mask.to(device)

                    if criterion_offset is None:
                        loss = criterion_score(pred_score_maps, label_mask)
                    else:
                        label_offsets = calculate_offset(pred_score_maps, label_pose)
                        loss = criterion_score(pred_score_maps, label_mask) + criterion_offset(pred_offsets, label_offsets)
                    
                    # loss += self.l1_regularization_loss() * reg_strength

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)

                    tepoch.set_postfix(loss=(loss.item(), current_MSE))

                epoch_loss = running_loss / total_samples
                epoch_MSE = MSE / batch_number
                

                tqdm.write(f"Epoch Loss: {epoch_loss:.4f}, MSE: {epoch_MSE:.4f}")
            training_MSE.append(epoch_MSE)
            training_loss.append(epoch_loss)

            # reset losses for validation
            running_loss = 0.0
            total_samples = 0
            MSE = 0

            #Validate
            if validate_loader != None:
                with tqdm(validate_loader, unit="batch") as tepoch:
                    tepoch.set_description(f"Epoch_Validate [{epoch+1}/{num_epochs}]")
                    with torch.no_grad():
                        for batch in tepoch:
                            images, label_pose = batch.image.to(device), batch.pose_matrix.to(device)

                            # Normalize images and 
                            norm_images = normalize_input(images)

                            pred_score_maps = self(norm_images)

                            # get scale factors
                            height_q = images.shape[1] / pred_score_maps.shape[2] #for x
                            width_q = images.shape[2] / pred_score_maps.shape[3] # for y

                            # get COORDS
                            pred_coords = self.find_max_value_coordinates(pred_score_maps.clone(), (width_q, height_q)).to(device)

                            # get MSE
                            current_MSE = self.MSE(pred_coords, label_pose.clone().detach()).item()
                            MSE += current_MSE

                            # create groundtruth mask 
                            label_mask =  create_normalized_gaussian_maps(batch.pose_matrix, pred_score_maps.shape[2], pred_score_maps.shape[3], std, quotients=(width_q, height_q))
                            label_mask = label_mask.to(device)

                            if criterion_offset is None:
                                loss = criterion_score(pred_score_maps, label_mask)
                            else:
                                label_offsets = calculate_offset(pred_score_maps, label_pose)
                                loss = criterion_score(pred_score_maps, label_mask) + criterion_offset(pred_offsets, label_offsets)


                            running_loss += loss.item() * images.size(0)
                            total_samples += images.size(0)

                            tepoch.set_postfix(loss=(loss.item(), current_MSE))

                        epoch_val_loss = running_loss / total_samples
                        epoch_val_MSE = MSE / batch_number
                
                tepoch.set_postfix(loss=f"Epoch Val Loss: {epoch_val_loss:.4f}, MSE Val: {epoch_val_MSE:.4f}")

            if scheduler:
                scheduler.step()

            validation_MSE.append(epoch_val_MSE)
            validation_loss.append(epoch_val_loss)
                        
        print('Training complete.')
        return training_loss, training_MSE, validation_loss, validation_MSE


    def evaluate_kp_model(self, device, test_loader, criterion_score, temperature=1, criterion_offset=None, std=0.05):
        self.to(device)
        self.eval()
        test_loss = 0.0
        total_samples = 0
        # total_batches = 0
        # min_loss = float('inf')
        # max_loss = float('-inf')
        # MSE = 0
        # batch_number = len(test_loader)
        # MSE_list = []
        # var_list = []
        temperature = torch.tensor(temperature).to(device)

        # Calculate ECE
        bin_size = 0.05  #  bin size for confidence intervals
        num_bins = int(1.05 / bin_size)
        # bin_confidence = np.zeros(num_bins)
        # bin_accuracy = np.zeros(num_bins)
        bin_count = np.zeros(num_bins)
        # metric = MulticlassCalibrationError(num_classes=12544, n_bins=50, norm='l1')
        outlier_count = 0
        count = 0
        # offset_x = []
        # offset_y = []
        quantiles = np.arange(0, 1.05, 0.05)
        MSE = 0
        MSE_count = 0
        likelihood = 0
        MSE_list = []
        likelihood_list = []
        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as progress_bar:
                for batch in progress_bar:
                    images, label_pose = batch.image.to(device), batch.pose_matrix.to(device).clone()
            

                    # Normalize images and
                    norm_images = normalize_input(images)

                    outputs = self(norm_images)

                    #temperature 
                    batch_size, num_channels, height, width = outputs.size()
                    x_reshaped = outputs.view(batch_size, num_channels, height * width)
                    x_reshaped = x_reshaped / temperature 
                    score_maps_reshaped = F.softmax(x_reshaped, dim=2)
                    pred_score_maps = score_maps_reshaped.view(batch_size, num_channels, height, width)

                    
                    height_q = images.shape[1] / pred_score_maps.shape[2]
                    width_q = images.shape[2] / pred_score_maps.shape[3]
                    

                    # ECE
                    position = label_pose.clone()#.cpu().numpy()
                    position[:,:,0] /= width_q
                    position[:,:,1] /= height_q
                    
                    
                    # sample ece
                    bin_count, outlier_count, count = self.ece3(pred_score_maps.clone(), position.clone(), bin_count, outlier_count, count, quantiles)

                    MSE, MSE_count, likelihood, MSE_list, likelihood_list = self.cal_MSE(pred_score_maps.clone(), position.clone(), label_pose.clone(), MSE, MSE_count, device, width_q, height_q, likelihood, MSE_list, likelihood_list)

                    # mean = self.find_max_value_coordinates(pred_score_maps.clone(), (1, 1)).to(device)
                    
                    # offset_x.append((mean[:,:,0] - position[:,:,0]).cpu().numpy())
                    # offset_y.append((mean[:,:,1] - position[:,:,1]).cpu().numpy())
                    
                    # bin_count, outlier_count, count = self.ece2(pred_score_maps.clone(), position, bin_count, bin_size, outlier_count, count)

                    # pred_coords = self.find_max_value_coordinates(pred_score_maps.clone(), (width_q, height_q)).to(device)
                    #bin_count, bin_accuracy, bin_confidence, MSE_list, var_list = self.ece(bin_count, bin_accuracy, bin_confidence, pred_score_maps.clone().cpu().numpy(), position, bin_size, MSE_list, var_list)


                    #get coordinates of prediction
                    #pred_coords = self.find_max_value_coordinates(pred_score_maps.clone(), (width_q, height_q)).to(device)

                    # get MSE
                    # currenet_MSE = self.MSE(pred_coords, label_pose.clone()).item()
                    # MSE += currenet_MSE
                

                    # create groundtruth mask
                    label_mask = create_normalized_gaussian_maps(label_pose, pred_score_maps.shape[2], pred_score_maps.shape[3], std, quotients=(width_q, height_q))
                    label_mask = label_mask.to(device)
        
                    if criterion_offset is None:
                        loss = criterion_score(pred_score_maps, label_mask)
                    # else:
                    #     label_offsets = calculate_offset(pred_score_maps, label_pose)
                    #     loss = criterion_score(pred_score_maps, label_mask) + criterion_offset(pred_offsets, label_offsets)

                    test_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)
                    # total_batches += 1

                    # #progress_bar.update(1)

                    # if loss < min_loss:
                    #     min_loss = loss
                    #     best_pred_mask_pair = (pred_score_maps.cpu().numpy(), label_mask.cpu().numpy())
                    #     best_images = images.cpu().numpy()
                    #     best_pose = batch.pose_matrix.cpu().numpy()
                        

                    # if loss > max_loss:
                    #     max_loss = loss
                    #     worst_pred_mask_pair = (pred_score_maps.cpu().numpy(), label_mask.cpu().numpy())
                    #     worst_images = images.cpu().numpy()
                    #     worst_pose = batch.pose_matrix.cpu().numpy()

                    # progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                    # progress_bar.set_postfix(loss=f"{loss.item():.4f}, {currenet_MSE:.4f}")
                    progress_bar.set_postfix(loss=f"{MSE/MSE_count}, {likelihood/MSE_count}")
                    
                    

                    

              
        avg_test_loss = test_loss / total_samples
        # avg_MSE = MSE / batch_number
        # print(f'Evaluation complete. Average Test Loss: {avg_test_loss:.4f}')

         # clalculate ECE 
        # divisor = np.maximum(bin_count, 1e-10)
        # bin_confidence /= divisor
        # bin_accuracy /= divisor

        # probability = bin_count / np.sum(bin_count)

        # ece = np.sum(np.abs(bin_accuracy - bin_confidence) * probability)
        # ece = metric.compute()
        # print(ece, outlier_count)

        # bin_accuracy /= bin_count
        # bin_confidence /= bin_count
        # ece2 = bin_accuracy - bin_confidence
        # bin_count /= np.sum(bin_count)
        # ece2 *= bin_count
        # ece2 = np.sum(ece2)
        # print(ece2)
        MSE /= MSE_count
        likelihood /= MSE_count
        return bin_count, outlier_count, count, MSE.cpu().numpy(), MSE_count, avg_test_loss, likelihood, np.array(MSE_list), np.array(likelihood_list)
        # return ece.cpu().numpy(), outlier_count, bin_accuracy, bin_confidence, bin_count, best_pred_mask_pair, best_images, best_pose, worst_pred_mask_pair, worst_images, worst_pose

        # return avg_test_loss, best_pred_mask_pair, best_images, best_pose, worst_pred_mask_pair, worst_images, worst_pose, avg_MSE, ece, MSE_list, var_list

    # SAVE MODEL
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def find_max_value_coordinates(self, matrix, quotients):
        batch_size, num_keypoints, height, width = matrix.shape
        max_coords = torch.zeros(batch_size, num_keypoints, 2, dtype=torch.float32)

        for i in range(batch_size):
            for j in range(num_keypoints):
                max_value, max_index = torch.max(matrix[i][j].view(-1), dim=0)
                max_x = max_index // width
                max_y = max_index % width
                max_x = max_x * quotients[0]
                max_y = max_y * quotients[1]
                max_coords[i, j] = torch.tensor([max_x, max_y])

        return max_coords


    def MSE(self, tensor1, tensor2):
        squared_diff = (tensor1 - tensor2)**2
        mse = torch.mean(squared_diff)
        return mse

    def gaussian_2d(self, xy, mu_x, mu_y, sigma_x, sigma_y, rho):
        x, y = xy[0], xy[1]
        z = (x - mu_x) ** 2 / sigma_x ** 2 - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y) + (y - mu_y) ** 2 / sigma_y ** 2
        exponent = -1 / (2 * (1 - rho ** 2)) * z
        constant = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(np.maximum(1 - rho ** 2, 1e-10)))  
        # return (constant * np.exp(exponent, where=(exponent <= 0))).ravel()
        return (constant * np.exp(exponent)).ravel()



    def find_max_value_coordinates2(self, matrix):
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
        return (max_y, max_x)

    def ece(self, count, acc, conf, pred, label, bin_size, MSE, variance):
        # var = []
        # mse = []
        #creat mesh
        width, height = pred.shape[2:]
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                data = pred[i][j]
                pose = label[i][j]
                
                #intial guess
                mean = np.array(self.find_max_value_coordinates2(data))
                
                initial_guess = (mean[1], mean[0], 1,1,0)
    
                #fglattendata
                data_flatten = data.ravel()
             
                # cruve fit
                try:
                # curve fit
                    popt, pcov = opt.curve_fit(self.gaussian_2d, (X, Y), data_flatten, p0=initial_guess, maxfev=5000)
                except RuntimeError:
                    # Handle the error (e.g., print a message, log, etc.)
                    print(f"Curve fit failed for data at index ({i}, {j}). Skipping...")
                    continue

                #var
                # var.append(popt[2] + popt[3])
                variance.append(popt[2] + popt[3])

                # Calculate ECE bins
                confidence = self.gaussian_2d((popt[0], popt[1]), *popt)[0]
                bin_index = int(confidence / bin_size)

                conf[bin_index] += confidence

                mse_offset = mean_squared_error([pose[1], pose[0]], [popt[0], popt[1]])
                # mse.append(mse_offset)
                MSE.append(mse_offset)
                if mse_offset <= 25 :
                    acc[bin_index] += 1   

                count[bin_index] += 1
        # variance.append(np.mean(var))
        # MSE.append(np.mean(mse))
        
        
        return count, acc, conf, MSE, variance

    # def ece_update(self, pred, target, metric, count, bin_confidence, bin_accuracy, bin_count, bin_size):
    #     #flatten pred:
    #     flat_pred = pred.view(pred.shape[0] * pred.shape[1], pred.shape[2] * pred.shape[3])

    #     #max index
    #     max_index = torch.argmax(flat_pred, dim=1)
        
    #     #coordinates to indeices
    #     target = torch.round(target)
    #     target_shape = target.shape
    #     # print(target_shape)
    #     flat_target = target.view(target_shape[0] * target_shape[1], 2)
    #     target_idx = flat_target[:, 0] * pred.shape[2] + flat_target[:, 1]
    #     # check for outliers
    #     if torch.max(flat_target[:,0]) >= pred.shape[2] or torch.max(flat_target[:,1]) >= pred.shape[3]:
    #         return metric, count + 1, bin_confidence, bin_accuracy, bin_count

    #     #calculate acc and conf sum
    #     for i in range(target_idx.shape[0]):
    #         #check in which bin
    #         conf = flat_pred[i][int(max_index[i].item())]
    #         idx = int(conf / bin_size)
    #         # increase count
    #         bin_count[idx] += 1
    #         # Confidence sum
    #         bin_confidence[idx] += conf
    #         #accuracy sum
    #         if max_index[i] == target_idx[i]:
    #             bin_accuracy[idx] += 1

    #     #update ece
    #     metric.update(flat_pred, target_idx)
        
    #     return metric, count, bin_confidence, bin_accuracy, bin_count



    def ece2(self, pred, label, count, bin_size, outlier, c):
        width, height = pred.shape[2:]
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                data = pred[i][j].cpu().numpy()
                pose = label[i][j].cpu().numpy()
                # check for outlier
                if pose[0] >= height or pose[0] < 0 or pose[1] >= width or pose[1] < 0:
                    outlier += 1
                    continue

                #intial guess
                mean = np.array(self.find_max_value_coordinates2(data))
                initial_guess = (mean[1], mean[0], 1,1,0)
                #flattendata
                data_flatten = data.ravel()
             
                try:
                # curve fit
                    popt, pcov = opt.curve_fit(self.gaussian_2d, (X, Y), data_flatten, p0=initial_guess, maxfev=5000)
                except RuntimeError:
                    # Handle the error (e.g., print a message, log, etc.)
                    print(f"Curve fit failed for data at index ({i}, {j}). Skipping...")
                    continue
          
                cov = np.array([[popt[2]**2, popt[4] * popt[2] * popt[3]],
                                  [popt[4] * popt[2] * popt[3], popt[3]**2]])

                mean = [popt[1], popt[0]]
                confidence = st.multivariate_normal.cdf(pose, mean, cov)
   

                idx = int(confidence / bin_size)
                print(popt[:2], pose)
                print((popt[1] - pose[0]) + (popt[0] - pose[1]))

                count[idx] += 1
                c += 1
        
        return count, outlier, c


    # def ece2(self, pred, label, count, bin_size, outlier):
    #     width, height = pred.shape[2:]
    #     x = torch.linspace(0, width-1, width)
    #     y = torch.linspace(0, height-1, height)
    #     X, Y = torch.meshgrid(x, y)

    #     for i in range(pred.shape[0]):
    #         for j in range(pred.shape[1]):
    #             data = pred[i][j].cpu()
    #             pose = label[i][j].cpu()

    #             if pose[0] >= height or pose[0] < 0 or pose[1] >= width or pose[1] < 0:
    #                 outlier += 1
    #                 continue

    #             mean = torch.tensor(self.find_max_value_coordinates2(data), dtype=torch.float32, device='cuda')
    #             initial_guess = torch.tensor((mean[1], mean[0], 1, 1, 0), dtype=torch.float32, device='cuda')

    #             data_flatten = data.view(-1).float()

    #             try:
    #                 # Curve fit using torch
    #                 popt, pcov = opt.curve_fit(self.gaussian_2d, (X, Y), data_flatten, p0=initial_guess, maxfev=5000)
    #             except RuntimeError:
    #                 print(f"Curve fit failed for data at index ({i}, {j}). Skipping...")
    #                 continue

    #             cov = torch.tensor([[popt[2]**2, popt[4] * popt[2] * popt[3]],
    #                                 [popt[4] * popt[2] * popt[3], popt[3]**2]], dtype=torch.float32, device='cuda')

    #             confidence = st.multivariate_normal.cdf(pose, mean, cov)

    #             idx = int(confidence / bin_size)

    #             count[idx:] += 1

    #     return count, outlier


    def ece3(self, pred, label, count, outlier, c, quantiles):
        width, height = pred.shape[2:]

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                data = pred[i][j]
                pose = label[i][j]
                # check for outlier
                if pose[0] >= height or pose[0] < 0 or pose[1] >= width or pose[1] < 0:
                    outlier += 1
                    continue

                data = data.unsqueeze(dim=0)
                samples = sample_from_heatmap(data)
                v, quantiles = calibration(samples, pose, quantiles)
   

                count += v
                c += 1
        
        return count, outlier, c

    def cal_MSE(self, pred, label, label_scaled, MSE, c, device, width_q, height_q, likelihood, MSE_list, likelihood_list):
        width, height = pred.shape[2:]
        skip = False
        for i in range(pred.shape[0]):
            
            tmp_MSE = 0
            tmp_c = 0
            tmp_l = 0
            tmp_l_l = []
            tmp_mse_l = []
            skip = False
            for j in range(pred.shape[1]):
                
                data = pred[i][j]
                pose = label[i][j]
                pose_scaled = label_scaled[i][j]
                # check for outlier
                if pose[0] >= height or pose[0] < 0 or pose[1] >= width or pose[1] < 0:
                    skip = True
                    
                    break
                
                max_index = torch.argmax(data)
                max_index_2d = torch.tensor(divmod(max_index.item(), data.size(1)), dtype=torch.float32, device=device)

                max_index_2d[0] *= width_q
                max_index_2d[1] *= height_q
                
                d = torch.sum((max_index_2d - pose_scaled)**2)
                
                l = torch.log(data[int(pose[0])][int(pose[0])])
                tmp_l += l.item()
               
                tmp_l_l.append(l.item())
                tmp_mse_l.append(d.item())

                tmp_MSE += d
                tmp_c += 1
            
            if skip is False:
                MSE += tmp_MSE
                c += tmp_c
                likelihood += tmp_l
                MSE_list += tmp_mse_l
                likelihood_list += tmp_l_l
            
            
        return MSE, c, likelihood, MSE_list, likelihood_list


def sample_from_heatmap(heatmap):
    shape = heatmap.shape
    # Normalize the heatmap along the last two dimensions
    heatmap = heatmap / heatmap.sum(dim=(-1, -2), keepdim=True)

    # Create a categorical distribution
    dist = torch.distributions.Categorical(heatmap.view(heatmap.shape[-3], -1))

    # Sample from the distribution
    indices = dist.sample((100, )).repeat(10, 1)

    # Convert the indices to 2D coordinates
    coords = torch.stack((indices // shape[-1], indices % shape[-1]), dim=-1)

    # coords[..., [0, 1]] = coords[..., [1, 0]]

    return coords



def calibration(sample, gt_3D, quantiles):
    sample_mean = sample.median(0).values[None].float()
   
    errors = ((sample_mean - sample) ** 2).sum(-1).mean(-1) ** 0.5
    true_error = (
        (((sample_mean - gt_3D) ** 2).sum(-1).mean(-1) ** 0.5)
        .cpu()
        .numpy()
    )

    q_vals = np.quantile(errors.cpu().numpy(), quantiles, 0)  # .squeeze(1)
    v = (q_vals > true_error.squeeze()).astype(int)
 
    return v, quantiles



