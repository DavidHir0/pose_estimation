import torch
import numpy as np
import matplotlib.pyplot as plt
# from pose_estimation.models import KpDetector
from pose_estimation.data_pp.loaders import static_loader
from pose_estimation.data_pp.utils import create_normalized_gaussian_maps, normalize_input
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import sys



# def sample_from_heatmap(heatmap):
#     # Normalize the heatmap along the last two dimensions
#     print(heatmap)
#     heatmap = heatmap / heatmap.sum(dim=(-1, -2), keepdim=True)

#     print(heatmap.sum(dim=(-1, -2)))

#     # Create a categorical distribution
#     dist = torch.distributions.Categorical(heatmap.view(heatmap.shape[-3], -1))

#     # Sample from the distribution
#     indices = dist.sample((100,)).repeat(10, 1)
#     print(indices)
#     print(indices.shape)

#     # Convert the indices to 2D coordinates
#     coords = torch.stack((indices // 64, indices % 64), dim=-1)

#     coords[..., [0, 1]] = coords[..., [1, 0]]
#     print(coords.shape)

#     return coords

# # Example usage:
# # Create a dummy heatmap (assuming size 64x64 for demonstration)
# dummy_heatmap = torch.abs(torch.randn(1, 1, 64, 64))

# # Call the function to get sampled coordinates
# sampled_coords = sample_from_heatmap(dummy_heatmap)

# # Print the sampled coordinates
# print("Sampled Coordinates:")
# # print(sampled_coords)




# # Example data for the sample and ground truth
# sample = np.random.normal(loc=5, scale=2, size=(100, 2))
# gt_3D = np.array([6, 7])  # Ground truth mean

# # Define quantiles from 0 to 1 in increments of 0.05
# quantiles = np.arange(0, 1.05, 0.05)

# # Calculate the mean of the samples along the first axis
# sample_mean = np.median(sample, axis=0)

# # Calculate errors: Euclidean distances between sample means and individual samples
# errors = np.sqrt(np.sum((sample_mean - sample) ** 2, axis=1))

# # Calculate true error: Euclidean distance between sample mean and ground truth
# true_error = np.sqrt(np.sum((sample_mean - gt_3D) ** 2))

# # Calculate quantiles of errors
# q_vals = np.quantile(errors, quantiles)

# # Create binary indicator based on whether the error is greater than true error
# v = (q_vals > true_error).astype(int)

# print("Sample Mean:", sample_mean)
# print("True Error:", true_error)
# print("Quantiles of Errors:", q_vals)
# print("Binary Indicator (1 if error > true_error, 0 otherwise):", v)
# print(quantiles)

# sys.exit()



# x_off = np.load(f"results/{load}/x_off.npy")
# y_off = np.load(f"results/{load}/y_off.npy")
# shape = x_off.shape[0] * x_off.shape[1] * x_off.shape[2]
# print(shape)
# print(np.sum(x_off)/shape)
# print(np.sum(y_off)/shape)
# sys.exit()

load = "cropped_6_wd/wd_0.0001/new_data_test/t=0.4/s3-d1"
safe = "cropped_6_wd/wd_0.0001/new_data_test/t=0.4/s3-d1"
lala = {0.4, 1.0}
for t in lala:
    # t  = 0.4
    # for t in np.arange(0.6, 1.6, 0.2):
    
    count = np.load(f"results/{load}/temp={t}_bin_count.npy")
    out = np.load(f"results/{load}/temp={t}_outlier_count.npy")
    MSE = np.load(f"results/{load}/temp={t}_MSE.npy")
    print(f"MSE, t={t}: {MSE**0.5}")
    likelihood = np.load(f"results/{load}/temp={t}_likelihood.npy")
    print(f"likelihood: {likelihood}")
    c = 2036760 - out
    
    c = np.load(f"results/{load}/temp={t}_data_point_count.npy")
    
    count2 = count.copy()
    count /= c
    ece = 0
    ece2 = 0


    bin_edges = np.arange(0, 1.005, 0.05)
    for i in range(count.shape[0]):
        
        tmp = count2[i] * (np.abs(count[i] - bin_edges[i]))
        tmp /= np.sum(count2)
        
        
        ece += tmp
        ece2 += np.abs(count[i] - bin_edges[i])
        
    print(ece2)
    print(ece)
  
    ece2 /= count.shape[0]

    
    # Plotting the histogram
    # plt.bar(bin_edges[:], count, width=0.05, color='blue', edgecolor='black', alpha=0.7)

    plt.plot(bin_edges, count, marker='o', label='real calibration', color='blue')
    plt.plot(bin_edges, bin_edges, marker='o', label='optimal calibration', color='red')
    # for i, s in enumerate(count):
    #     plt.scatter((i+1)*0.05, s, color='blue', marker='.')

    # plt.title(f"Temperature = {t:.1f}, ECE = {ece:.2f}, mean ECE = {ece/20:.2f}")
    plt.xlim(0,1)
    # plt.ylim(-0.01,1.01)
    la = np.arange(0, 1.1, 0.1)
    plt.title(f"Temperature: {t:.1f}, weighted-ECE: {ece:.8f}, ECE: {ece2:.8f}")
    plt.xticks(la, fontsize=8)
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f"results/{safe}/temp={round(t, 1)}_conf_acc.png")

    plt.close()


# # Initialize subplots
# fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30, 12), sharex=True, sharey=True)
# fig.suptitle('Accuracy and Confidence Scatter Plots for Different t values')

# # Flatten the 2D array of subplots
# axs = axs.flatten()

# # Iterate over different values of t
# for i, t in enumerate(np.arange(1, 2, 0.1)):
#     # Load data
#     outlier = np.load(f"results/{load}/outlier_count_{t}.npy")
#     ece = np.load(f"results/{load}/ECE_{t}.npy")
#     bins = np.load(f"results/{load}/bins_{t}.npy")
#     acc, conf, count = bins[0], bins[1], bins[2]

#     # calcullate ece
#     print(np.sum(count))
#     print(outlier)
#     acc /= count
#     conf /= count
#     acc = np.where(np.isnan(acc), 0, acc)
#     conf = np.where(np.isnan(conf), 0, conf)

#      #calculate ece
#     ece2 = np.abs(acc - conf)
#     count /= np.sum(count)
#     ece2 *= count
#     ece2 = np.sum(ece2)


#     # Plot the scatter plot for accuracy and confidence
#     axs[i].scatter(acc, conf, alpha=0.7)
#     axs[i].set_title(f't={t:.1f}, ece={ece:.3f}, my ece={ece2:.3f}')
#     axs[i].set_xlabel("Accuracy")
#     axs[i].set_ylabel("Confidence")
#     axs[i].set_xlim(0, 1)
#     axs[i].set_ylim(0, 1)


   





# # Save the figure
# plt.savefig(f"results/{safe}/acc_conf.png")
# plt.show()
