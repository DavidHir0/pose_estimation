import numpy as np
import scipy.optimize as opt
import torch 
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from numpy import corrcoef



def gaussian_2d(xy, mu_x, mu_y, sigma_x, sigma_y, rho):
    x, y = xy[0], xy[1]
    z = (x - mu_x) ** 2 / sigma_x ** 2 - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y) + (y - mu_y) ** 2 / sigma_y ** 2
    exponent = -1 / (2 * (1 - rho ** 2)) * z
    constant = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(np.maximum(1 - rho ** 2, 1e-10)))  
    # return (constant * np.exp(exponent, where=(exponent <= 0))).ravel()
    return (constant * np.exp(exponent)).ravel()

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
    return (max_y, max_x)







safe = "cropped_9_10_epochs/2"

# load
best_pred = np.load(f"results/{safe}/best_prediction.npy")
best_poses = np.load(f"results/{safe}/best_pose.npy")
best_images = np.load(f"results/{safe}/best_images.npy")
preds = best_pred[0]
labels = best_pred[0]
mse_plot = []
var_plot = []
ece_values = []

# Calculate ECE
bin_size = 0.1  #  bin size for confidence intervals
num_bins = int(1 / bin_size)
bin_confidence = np.zeros(num_bins)
bin_accuracy = np.zeros(num_bins)
bin_count = np.zeros(num_bins)


for i in range(preds.shape[0]):
    sum = 0
    variance_sum = 0
    pred_points_x = []
    pred_points_y = []
    label_points_x = []
    label_points_y = []
    likelihood_pred = []
    likelihood_label = []
    image = best_images[i]
    variance = []
    mse_offset_sum = 0

   


    for j in range(preds.shape[1]):
        pred = preds[i][j]
        ground_truth = labels[i][j]
        pose = best_poses[i][j]
        #intial guess
        mean = np.array(find_max_value_coordinates(pred))
        initial_guess = (mean[1], mean[0], 1,1,0)

        #creat mesh
        width, height = 112, 112
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)


        #fglattendata
        data_flatten = pred.ravel()


        # cruve fit
        popt, pcov = opt.curve_fit(gaussian_2d, (X,Y), data_flatten, p0=initial_guess)

        #likelihood
        # likelihood = gaussian_2d((pose[1], pose[0]), *popt)
        likelihood = gaussian_2d((mean[1], mean[0]), *popt)
        likelihood_pred.append(gaussian_2d((mean[1], mean[0]), *popt))
        sum += likelihood
        variance.append(popt[2] + popt[3])
        variance_sum += popt[2] + popt[3]
        mse_offset = mean_squared_error([pose[0], pose[1]], [popt[1], popt[0]])
        mse_offset_sum += mse_offset
        mse_plot.append(mse_offset)
        var_plot.append(popt[2] + popt[3])
        


        # Calculate ECE bins
        confidence = gaussian_2d((popt[0], popt[1]), *popt)[0]
        bin_index = int(confidence / bin_size)
        bin_confidence[bin_index] += confidence
        if gaussian_2d((pose[1], pose[0]), *popt)[0] >= 0.01:
            bin_accuracy[bin_index] += 1   
        bin_count[bin_index] += 1
    

        # get points for plot
        pred_points_x.append(mean[0] * (image.shape[0] / width))
        pred_points_y.append(mean[1] * (image.shape[1] / height))
        label_points_x.append(pose[0] * (image.shape[0] / width))
        label_points_y.append(pose[1] * (image.shape[1] / height))

        
   


    mean_likelihood = sum / 20
    mean_variance = variance_sum / 20
    mean_mse_offset = mse_offset_sum / 20
    # mse_plot.append(mean_mse_offset)
    # var_plot.append(mean_variance)

    #plot
    normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
    normalized_image = normalized_image.astype(np.uint8)
    plt.imshow(normalized_image)
    plt.scatter(label_points_x, label_points_y,  color='red', label=f'preditction', marker='o')
    plt.scatter(pred_points_x, pred_points_y,  color='blue', label=f'ground_truth', marker='.')
    # for x, y, likelihood in zip(pred_points_x, pred_points_y, likelihood_pred):
    #     plt.text(x, y, f'{likelihood[0]:.3f}', color='black', fontsize=8, ha='left', va='bottom')
    for x, y, likelihood in zip(pred_points_x, pred_points_y, variance):
        plt.text(x, y, f'{likelihood:.3f}', color='black', fontsize=8, ha='left', va='bottom')
    # plt.title(f"Likelihood of Prediction means (mean likelihood = {mean_likelihood:.3f})")
    plt.title(f"Variance of Prediction means (mean Variance = {mean_variance:.3f})")
    plt.legend()
    os.makedirs(f"results/{safe}/likelihood", exist_ok=True)
    os.makedirs(f"results/{safe}/likelihood/best", exist_ok=True)
    plt.savefig(f"results/{safe}/likelihood/best/best_{i}.png", dpi=300)
    plt.close()

    # vcariance
    with open(f"results/{safe}/likelihood/best/variance_{i}.txt", 'w') as file:
        for g, v in enumerate(variance):
            file.write(f"{g}: {v}\n")

correlation = corrcoef(mse_plot, var_plot)[0, 1]
plt.scatter(mse_plot, var_plot, color='blue', label='MSE vs Variance', marker='.', alpha=0.7)
plt.ylabel('Variance')
plt.xlabel('Mean Squared Error')
plt.title(f"MSE vs Variance (best case), corr = {correlation:.3f}")
plt.legend()
plt.savefig(f"results/{safe}/likelihood/best/MSE_vs_Var.png")
plt.close()
print(np.sum(mse_plot))
bin_confidence /= (np.maximum(bin_count, 1e-10))
bin_accuracy /= (np.maximum(bin_count, 1e-10))

# clalculate ECE 
ece = np.abs(bin_accuracy - bin_confidence)
ece *= bin_count
ece /= np.sum(bin_count)
ece_values.append(np.sum(ece))
ece_mean = np.mean(ece_values)
print(f"Mean ECE: {ece_mean}")



print("worst")


# worts case
# load
best_pred = np.load(f"results/{safe}/worst_prediction.npy")
best_poses = np.load(f"results/{safe}/worst_pose.npy")
best_images = np.load(f"results/{safe}/worst_images.npy")
preds = best_pred[0]
labels = best_pred[0]
ece_values = []
mse_plot = []
var_plot = []

for i in range(preds.shape[0]):
    sum = 0
    variance_sum = 0
    pred_points_x = []
    pred_points_y = []
    label_points_x = []
    label_points_y = []
    likelihood_pred = []
    likelihood_label = []
    image = best_images[i]
    variance = []
    mse_offset_sum = 0

    # Calculate ECE
    bin_size = 0.1  #  bin size for confidence intervals
    num_bins = int(1 / bin_size)
    bin_confidence = np.zeros(num_bins)
    bin_accuracy = np.zeros(num_bins)
    bin_count = np.zeros(num_bins)


    for j in range(preds.shape[1]):
        pred = preds[i][j]
        ground_truth = labels[i][j]
        pose = best_poses[i][j]
        #intial guess
        mean = np.array(find_max_value_coordinates(pred))
        initial_guess = (mean[1], mean[0], 1,1,0)

        #creat mesh
        width, height = 112, 112
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)


        #fglattendata
        data_flatten = pred.ravel()


        # cruve fit
        popt, pcov = opt.curve_fit(gaussian_2d, (X,Y), data_flatten, p0=initial_guess)

        #likelihood
        likelihood = gaussian_2d((mean[1], mean[0]), *popt)
        likelihood_pred.append(gaussian_2d((mean[1], mean[0]), *popt))
        sum += likelihood
        variance.append(popt[2] + popt[3])
        variance_sum += popt[2] + popt[3]
        mse_offset = mean_squared_error([pose[0], pose[1]], [popt[1], popt[0]])
        mse_offset_sum += mse_offset
        mse_plot.append(mse_offset)
        var_plot.append(popt[2] + popt[3])

         # Calculate ECE bins
        confidence = gaussian_2d((pose[1], pose[0]), *popt)[0]
        bin_index = int(confidence / bin_size)
        bin_confidence[bin_index] += confidence
        if confidence >= 0.5:
            bin_accuracy[bin_index] += 1   
        bin_count[bin_index] += 1


        # get points for plot
        pred_points_x.append(mean[0] * (image.shape[0] / width))
        pred_points_y.append(mean[1] * (image.shape[1] / height))
        label_points_x.append(pose[0] * (image.shape[0] / width))
        label_points_y.append(pose[1] * (image.shape[1] / height))

    bin_confidence /= (np.maximum(bin_count, 1e-10))
    bin_accuracy /= (np.maximum(bin_count, 1e-10))

    # clalculate ECE 
    ece = np.abs(bin_accuracy - bin_confidence)
    ece *= bin_count
    ece /= np.sum(bin_count)
    ece_values.append(np.sum(ece))
    ece_mean = np.mean(ece_values)

    mean_likelihood = sum / 20
    mean_variance = variance_sum / 20
    mean_mse_offset = mse_offset_sum / 20
    # mse_plot.append(mean_mse_offset)
    # var_plot.append(mean_variance)

    #plot
    normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
    normalized_image = normalized_image.astype(np.uint8)
    plt.imshow(normalized_image)
    plt.scatter(label_points_x, label_points_y,  color='red', label=f'preditction', marker='o')
    plt.scatter(pred_points_x, pred_points_y,  color='blue', label=f'ground_truth', marker='.')
    # for x, y, likelihood in zip(pred_points_x, pred_points_y, likelihood_pred):
    #     plt.text(x, y, f'{likelihood[0]:.3f}', color='black', fontsize=8, ha='left', va='bottom')
    for x, y, likelihood in zip(pred_points_x, pred_points_y, variance):
        plt.text(x, y, f'{likelihood:.3f}', color='black', fontsize=8, ha='left', va='bottom')
    plt.legend()
    # plt.title(f"Likelihood of Prediction means (mean likelihood = {mean_likelihood[0]:.3f})")
    plt.title(f"Variance of Prediction means (mean Variance = {mean_variance:.3f})")
    os.makedirs(f"results/{safe}/likelihood", exist_ok=True)
    os.makedirs(f"results/{safe}/likelihood/worst", exist_ok=True)
    plt.savefig(f"results/{safe}/likelihood/worst/worst_{i}.png", dpi=300)
    plt.close()


    # vcariance
    with open(f"results/{safe}/likelihood/best/variance_{i}.txt", 'w') as file:
        for g, v in enumerate(variance):
            file.write(f"{g}: {v}\n")

correlation = corrcoef(mse_plot, var_plot)[0, 1]
plt.scatter(mse_plot, var_plot, color='red', label='MSE vs Variance', marker='.', alpha=0.7)
plt.ylabel('Variance')
plt.xlabel('Mean Squared Error')
plt.title(f"MSE vs Variance (worst case), corr = {correlation:.3f}")
plt.legend()
plt.savefig(f"results/{safe}/likelihood/worst/MSE_vs_Var.png")
plt.close()
print(np.sum(mse_plot))
print(f"Mean ECE: {ece_mean}")
