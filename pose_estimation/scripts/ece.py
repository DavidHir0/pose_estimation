import numpy as np
import scipy.optimize as opt
import torch 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import corrcoef

def gaussian_2d(xy, mu_x, mu_y, sigma_x, sigma_y, rho):
    x, y = xy[0], xy[1]
    z = (x - mu_x) ** 2 / sigma_x ** 2 - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y) + (y - mu_y) ** 2 / sigma_y ** 2
    exponent = -1 / (2 * (1 - rho ** 2)) * z
    constant = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(np.maximum(1 - rho ** 2, 1e-10)))  
    return (constant * np.exp(exponent)).ravel()

def find_max_value_coordinates(matrix):
    matrix = torch.tensor(matrix)
    max_value, max_index = torch.max(matrix.view(-1), dim=0)
    max_y = max_index // matrix.shape[0]
    max_x = max_index % matrix.shape[0]
    return (max_y, max_x)

# Load data
safe = "cropped_6_wd/wd_0.0001/platt"
ece = np.load(f"results/{safe}/ECE.npy")
MSE_list = np.load(f"results/{safe}/all_MSE.npy")
var_list = np.load(f"results/{safe}/all_var.npy")
print(ece)

# Filter out too big values in MSE_list
max_mse_threshold = 500  # Adjust the threshold as needed
filtered_MSE_list = MSE_list[MSE_list <= max_mse_threshold]
filtered_var_list = var_list[MSE_list <= max_mse_threshold]

# Identify outliers
outliers = MSE_list[MSE_list > max_mse_threshold]
print(len(MSE_list))
# Count outliers
num_outliers = len(outliers)

# Plot filtered MSE distribution with relative frequency
plt.hist(filtered_MSE_list, bins=50, color='blue', alpha=0.7, density=True, label='Filtered MSE')
plt.title('MSE Distribution (Relative Frequency)')
plt.xlabel('Mean Squared Error')
plt.ylabel('Relative Frequency')
plt.savefig(f"results/{safe}/MSE_distribution.png")
plt.ylim(0, 0.5)
plt.close()
# Plot outliers in a separate histogram
if num_outliers > 0:
    plt.hist(outliers, bins=20, color='red', alpha=0.7, density=True, label='Outliers')
    plt.legend()
    plt.ylim(0, 0.01)


plt.savefig(f"results/{safe}/MSE_distribution_outlier.png")

# Print the number of outliers
print(f"Number of outliers: {num_outliers}")

#MSE vs VAR

correlation_matrix = np.corrcoef(filtered_MSE_list, filtered_var_list)
correlation_coefficient = correlation_matrix[0, 1]




plt.figure()
plt.scatter(filtered_MSE_list, filtered_var_list, color='green', alpha=0.5, marker='.')
plt.title(f'Scatter Plot: MSE vs. Variance (correlation: {correlation_coefficient:.3f})')
plt.xlabel('Mean Squared Error')
plt.ylabel('Variance')
plt.savefig(f"results/{safe}/MSE_vs_Variance_Scatter.png")
plt.close()


