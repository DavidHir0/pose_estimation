import numpy as np
import matplotlib.pyplot as plt
from pose_estimation.data_pp.utils import create_normalized_gaussian_maps
import torch
import os


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

safe = "cropped_6_wd/wd_0.0001/ece_test/with_map"
tmp = "cropped_6_wd/wd_0.0001"
image_size = 448
size = 64
index = 14

for t in np.arange(1.0, 2.2, 0.1):
    # training_loss = np.load(f"results/{safe}/training_loss.npy")
    # training_loss = np.load(f"results/{tmp}/training_loss.npy")

    # test_loss = np.load(f"results/{safe}/evaluate_loss.npy")

    best_pred = np.load(f"results/{safe}/best_prediction_{t}.npy")

    images = np.load(f"results/{safe}/best_images_{t}.npy")

    worst_pred = np.load(f"results/{safe}/worst_prediction_{t}.npy")

    worst_images = np.load(f"results/{safe}/worst_images_{t}.npy")

    # MSE = np.load(f"results/{safe}/MSE.npy")

    # training_MSE = np.load(f"results/{safe}/training_MSE.npy")
    # validation_MSE = np.load(f"results/{safe}/validation_MSE.npy")
    # validation_loss = np.load(f"results/{safe}/validation_loss.npy")

    # training_MSE = np.load(f"results/{tmp}/training_MSE.npy")
    # validation_MSE = np.load(f"results/{tmp}/validation_MSE.npy")
    # validation_loss = np.load(f"results/{tmp}/validation_loss.npy")


    # plt.plot(range(training_MSE.shape[0]),training_MSE, label="Training Set")
    # plt.plot(range(validation_MSE.shape[0]),validation_MSE, label="Validation Set")
    # plt.xlabel('Epochs')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.savefig(f"results/{safe}/training_MSE.png", dpi=300)
    # plt.close()


    print(best_pred.shape)
    pred = best_pred[0]
    groundtruth = best_pred[1]

    # with open(f'results/{safe}/MSE.txt', 'w') as f:
    #     print(f'MSE Loss: {MSE:.4f}', file=f)

    # # print(training_loss)

    # with open(f'results/{safe}/test_loss.txt', 'w') as f:
    #     print(f'Test Loss: {test_loss:.4f}', file=f)
    # print(test_loss)
    # print(pred.shape)
    # print(images.shape)
    # print(groundtruth.shape)

    # plt.plot(range(training_loss.shape[0]),training_loss, label="Training Set")
    # plt.plot(range(validation_loss.shape[0]),validation_loss, label="Validation Set")
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig(f"results/{safe}/training_loss.png", dpi=300)
    # plt.close()
    # print(images.shape)
    upscale_height = images.shape[1] / pred.shape[2]
    upscale_width = images.shape[2]  / pred.shape[3]
    # upscale_height = image_size / size
    # upscale_width = image_size / size
    # print(upscale_height)
    # print(upscale_width)


    sample_pred = pred[index]
    sample_target = groundtruth[index]
    sample_image = images[index]
    pred_coords = []
    target_coords = []

    #poses = poses.long()

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
    plt.savefig(f"results/{safe}/{t}_best_heatmap_compare_all.png")
    plt.close()

    for index in range(pred.shape[0]):

        sample_pred = pred[index]
        sample_target = groundtruth[index]
        sample_image = images[index]
        pred_coords = []
        target_coords = []


        for i in range(sample_pred.shape[0]):
            pred_coords.append(find_max_value_coordinates(sample_pred[i]))
            target_coords.append(find_max_value_coordinates(sample_target[i]))

        print(pred_coords)
        print(target_coords)

        upscaled_points = [(x * upscale_width, y * upscale_height) for x, y in target_coords]
        upscaled_pred_points = [(x * upscale_height, y * upscale_width) for x, y in pred_coords]

        upscaled_x, upscaled_y = zip(*upscaled_points)
        upscaled_pred_x, upscaled_pred_y = zip(*upscaled_pred_points)
        print(upscaled_points)
        print()
        print(upscaled_pred_points)

        plt.scatter(upscaled_y, upscaled_x,  color='red', label=f'ground_truth', marker='o')
        plt.scatter(upscaled_pred_x, upscaled_pred_y,  color='blue', label=f'preditction', marker='.')
        image = sample_image
        normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
        normalized_image = normalized_image.astype(np.uint8)
        plt.imshow(normalized_image)
        plt.legend()
        # plt.savefig(f"results/{safe}/{t}_prediction_{index}.png")
        if os.path.exists(f"results/{safe}/{t}_prediction_{index}.png"):
            os.remove(f"results/{safe}/{t}_prediction_{index}.png")
            print(f"has been deleted.")
        else:
            print(f"does not exist.")

        plt.close()





    #worst
    pred = worst_pred[0]
    groundtruth = worst_pred[1]

    sample_pred = pred[index]
    sample_target = groundtruth[index]
    sample_image = worst_images[index]
    pred_coords = []
    target_coords = []

    #poses = poses.long()

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
    plt.savefig(f"results/{safe}/{t}_worst_heatmap_compare_all.png")
    plt.close()

    for index in range(pred.shape[0]):

        sample_pred = pred[index]
        sample_target = groundtruth[index]
        sample_image = worst_images[index]
        pred_coords = []
        target_coords = []


        for i in range(sample_pred.shape[0]):
            pred_coords.append(find_max_value_coordinates(sample_pred[i]))
            target_coords.append(find_max_value_coordinates(sample_target[i]))

        print(pred_coords)
        print(target_coords)

        upscaled_points = [(x * upscale_width, y * upscale_height) for x, y in target_coords]
        upscaled_pred_points = [(x * upscale_height, y * upscale_width) for x, y in pred_coords]

        upscaled_x, upscaled_y = zip(*upscaled_points)
        upscaled_pred_x, upscaled_pred_y = zip(*upscaled_pred_points)
        print(upscaled_points)
        print()
        print(upscaled_pred_points)

        plt.scatter(upscaled_y, upscaled_x,  color='red', label=f'ground_truth', marker='o')
        plt.scatter(upscaled_pred_x, upscaled_pred_y,  color='blue', label=f'preditction', marker='.')
        image = sample_image
        normalized_image = (image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))
        normalized_image = normalized_image.astype(np.uint8)
        plt.imshow(normalized_image)
        plt.legend()
        # plt.savefig(f"results/{safe}/{t}_worst_prediction_{index}.png")
        if os.path.exists(f"results/{safe}/{t}_worst_prediction_{index}.png"):
            os.remove(f"results/{safe}/{t}_worst_prediction_{index}.png")
            print(f"has been deleted.")
        else:
            print(f"does not exist.")
        plt.close()


