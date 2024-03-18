import numpy as np
import matplotlib.pyplot as plt
from pose_estimation.data_pp.utils import create_normalized_gaussian_maps
import torch
import os

safe = "cropped_6_wd/wd_0.0001/ece_test/with_map/re"
best_preds = np.load(f"results/{safe}/pred_pairs.npy")
index = 0

for k, best_pred in enumerate(best_preds):
    print(best_pred.shape)
    pred = best_pred[0]
    groundtruth = best_pred[1]


    sample_pred = pred[index]
    sample_target = groundtruth[index]

    fig, axes = plt.subplots(5, 8, figsize=(20, 15))

    title_colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'lime', 'pink',
                    'brown', 'gray', 'olive', 'teal', 'navy', 'indigo', 'chocolate', 'firebrick', 'darkgreen', 'sienna']
    for i in range(20):
        row = i // 4
        col = (i % 4) * 2

        axes[row, col].imshow(sample_pred[i], cmap='viridis')
        axes[row, col].set_title(f'Heatmap {i + 1} (Pred)', color=title_colors[i])
        axes[row, col].axis('off')

        axes[row, col + 1].imshow(sample_target[i], cmap='viridis')
        axes[row, col + 1].set_title(f'Heatmap {i + 1} (Target)', color=title_colors[i])
        axes[row, col + 1].axis('off')

    for i in range(20, 40):
        row = i // 8
        col = i % 8

        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/{safe}/{k}_heat_map.png")
    plt.close()