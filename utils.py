"""
General utility functions for the project
Author: Jeanne Malécot
"""
import glob
import os
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm
from scipy.io import loadmat

### Data loading functions ###

def load_data(path_data, path_labels):
    """
    Load the data and labels from the given paths
    """
    # Load the data
    data = []
    labels = []
    names = []
    for path_img in glob.glob(os.path.join(path_data + "/*.jpg")):
        names.append(os.path.basename(path_img[:-4]))
        data.append(path_img)
        path_label = os.path.join(path_labels, os.path.basename(path_img).replace(".jpg", ".jpg.mat"))
        labels.append(path_label)

    # Check if the data and labels have the same length
    if len(data) != len(labels):
        raise ValueError(
            "Number of data samples and labels do not match"
        )  

    labels = load_labels_mat(labels)

    dict_data = create_dict(data, labels, names)

    return dict_data

def create_dict(data, labels, names):
    """
    Create a dictionary with the data and labels suitable for the Monai Dataset class
    """
    data_dict = []
    for i in range(len(data)):
        data_dict.append({"image": data[i], "landmarks": torch.Tensor(labels[i]), "image_meta_dict":{"name": names[i]}, "landmarks_meta_dict":{}})
    return data_dict

def load_labels_mat(dir_list):
    """
    Load the labels from the .mat files
    """
    labels = []
    for label_dir in dir_list:
        labels.append((loadmat(label_dir)["p2"]).astype(np.int16))
        if (
            labels[-1].shape[0] != 68
        ):  # TODO: explain in report why, +eventually add logger
            # Number of landmarks can be != than 68
            orig_size = labels[-1].shape[0]
            labels[-1] = torch.cat([labels[-1], torch.zeros(68 - orig_size, 2)], dim=0)

    return labels


def load_config(config_dir):
    """
    Load the yaml configuration file
    """
    with open(config_dir, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def sort_by_date(directory):
    def get_creation_time(item):
        item_path = os.path.join(directory, item)
        return os.path.getctime(item_path)

    items = os.listdir(directory)
    sorted_items = sorted(items, key=get_creation_time, reverse=True)
    return sorted_items


def get_last_folder(path, model_name):
    """
    Return the last folder (by date) in the given path
    """
    list_dir = sort_by_date(path)
    for dir in list_dir:
        if model_name in dir and os.path.isdir(os.path.join(path, dir)):
            return dir
    return None


### Image processing functions ###

def save_heatmaps(images, save_dir, basename="image", cmap="jet"):
    """
    Save the images in the given directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_images = len(images)
    images_copy = images.clone() if torch.is_tensor(images) else np.copy(images)

    for i in range(n_images):
        if torch.is_tensor(images_copy[i]):
            image = images_copy[i].detach().cpu().numpy()
        else:
            image = images_copy[i]
            
        image = normalize_image(image)
        save_path = os.path.join(save_dir, f"{basename}_{i}.jpg")
        if cmap == "jet":
            # save the image with cmap jet
            cv2.imwrite(
                save_path, cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_JET)
            )
        elif cmap == "gray":
            # save the image with cmap gray
            cv2.imwrite(save_path, image)
        else:
            raise ValueError("The given cmap is not supported")

def save_dataset(dataset, save_dir, suffix, show_landmarks = True):
    """
    Save the dataset in the given directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, data in enumerate(tqdm(dataset, desc="Saving dataset")):
        image = data["image"].detach().cpu().numpy()

        #normalize between 0 and 1
        image = normalize_image(image)

        #create image in cv2 format
        if len(image.shape) == 2:  # Check if the image is single-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[0] == 1:  # Check if the image is single-channel
            image = cv2.cvtColor(image[0], cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if show_landmarks:
            landmarks = data["landmarks"].detach().cpu().numpy()
            c_radius = max(image.shape) // 500
            for landmark in landmarks:
                cv2.circle(
                    image, (int(landmark[0]), int(landmark[1])), c_radius, (0, 0, 255), c_radius*2
                )
            if "gt" in data:
                gtruths = data["gt"]
                for gtruth in gtruths:
                    cv2.circle(
                        image, (int(gtruth[0]), int(gtruth[1])), c_radius, (0, 255, 0), c_radius*2
                    )


        name = data["image_meta_dict"]["name"]
        if isinstance(name, list):
            name = name[-1]
        save_path = os.path.join(save_dir, f"{name}_{suffix}.jpg")
        cv2.imwrite(save_path, image)

def wandb_img(tensor, cmap="jet", caption="image"):
    """
    convert a tensor to a wandb image, to be saved with the desired colormap
    """
    image = tensor.detach().cpu().numpy()
    image = normalize_image(image)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap=cmap)
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close()

    image_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return wandb.Image(image)

def show_results(title, bin_heats, np_heats, gt_heatmap, ground_truths, image_plot, corners, sample_accuracy, heatmap_dir, top_line = None, global_accuracy = None):

    cmap = plt.get_cmap('gist_rainbow')  
    colors = [cmap(index / (17 - 1)) for index in range(17)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 10))
    fig.subplots_adjust(wspace=0.2)

    # First subplot: Original image with heatmap overlay
    alpha_mask = bin_heats * 0.3
    axes[0].imshow(image_plot[0], cmap='gray')
    axes[0].imshow(np_heats, cmap='jet', alpha=alpha_mask)
    axes[0].axis('off')
    axes[0].set_title(f"Input image with predictions")

    # Second subplot: Ground truth heatmap
    axes[1].imshow(gt_heatmap, cmap='jet')
    axes[1].axis('off')
    axes[1].set_title(f"Ground truth heatmap")

    # Third subplot: Heatmap only
    axes[2].imshow(np_heats, cmap='jet')
    axes[2].axis('off')
    axes[2].set_title(f"Predicted heatmap")

    # Fourth subplot: Binary heatmap with landmarks
    axes[3].imshow(bin_heats, cmap="gray")
    for idx, landmark in enumerate(ground_truths):
        
        axes[3].scatter(landmark[0], landmark[1], color=colors[idx%17], s=3)
        circle = plt.Circle((landmark[0], landmark[1]), 10, color=colors[idx%17], fill=False, linestyle='--')
        axes[3].add_patch(circle)

        if idx < len(corners):
            corner = corners[idx]
            if corner[0] != -1 and corner[1] != -1:
                axes[3].scatter(corner[0], corner[1], color=colors[idx%17], s=15, marker='x')
    
    if top_line is not None :
        for i in range(len(top_line)):
            # trace a horizontal dotted line
            axes[3].axhline(y=top_line[i], color='r', linestyle='--')

    axes[3].axis('off')
    axes[3].set_title(f"Binary heatmap with predicted and true landmarks")

    # Add accuracy text centered at the bottom
    fig.text(0.5, 0.02, f"Results for {title}", ha='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.05, f"Heatmap Accuracy: {sample_accuracy:.2f}%", ha='center', fontsize=10)
    if global_accuracy is not None:
        fig.text(0.5, 0.07, f"Total Accuracy: {global_accuracy:.2f}%", ha='center', fontsize=12)
    short_name = "_".join(title.lower().split(" "))
    plt.savefig(os.path.join(heatmap_dir, f"heatmap_{short_name}.png"))
    plt.close(fig)


def show_final(ground_truths, pred_landmarks, image_plot, heatmap_dir, accuracy, distance):

    cmap = plt.get_cmap('gist_rainbow')  
    colors = [cmap(index / (17 - 1)) for index in range(17)]

    fig, axes = plt.subplots(1, 2, figsize=(8, 10))
    plt.subplots_adjust(wspace=0.2)

    global_accuracy = accuracy['global_accuracy']
    lumbar_accuracy = accuracy['lumbar_accuracy']
    thoracic_accuracy = accuracy['thoracic_accuracy']

    global_distance = distance['global_distance']
    lumbar_distance = distance['lumbar_distance']
    thoracic_distance = distance['thoracic_distance']

    # First subplot: Original image with true landmarks
    axes[0].imshow(image_plot[0], cmap='gray')
    for i in range(17):
        for channel in range(4):
            axes[0].scatter(ground_truths[channel][i][0], ground_truths[channel][i][1], color=colors[i], s=10, marker='x')
    axes[0].axis('off')
    axes[0].set_title("Ground truth landmarks")

    
    # Second subplot: Original image with predicted landmarks
    axes[1].imshow(image_plot[0], cmap='gray')
    for i in range(17):
        for channel in range(4):
            if pred_landmarks[channel][i][0] != -1 and pred_landmarks[channel][i][1] != -1:
                axes[1].scatter(pred_landmarks[channel][i][0], pred_landmarks[channel][i][1], color=colors[i], s=10, marker='x')
    axes[1].axis('off')
    axes[1].set_title("Predicted landmarks")

    fig.text(0.5, 0.02, f"Final Results", ha='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.05, f"Average distance between pred and truth: {global_distance:.2f} (Lumbar vertebraes: {lumbar_distance:.2f}; Thoracic vertebraes: {thoracic_distance:.2f})", ha='center', fontsize=10)
    fig.text(0.5, 0.07, f"Accuracy: {global_accuracy:.2f}% (Lumbar vertebraes: {lumbar_accuracy:.2f}%; Thoracic vertebraes: {thoracic_accuracy:.2f}%)", ha='center', fontsize=10)
    plt.savefig(os.path.join(heatmap_dir, f"final_results.png"))
    plt.close(fig)

def normalize_image(image): 
    """
    map the values to [0, 255]
    """
    image = image - np.min(image)
    image = (image / np.max(image)) * 255
    return image


def has_n_landmarks(labels, num_landmarks):
    """
    Return the indices of the data and labels that have the expected number of landmarks
    """
    indices = []
    for i in range(len(labels)):
        if labels[i].shape[0] == num_landmarks:
            indices.append(i)
    return indices

def get_left_landmarks(landmarks):
    """
    Return the coordinates of the left landmarks (odd lines of the landmarks array)
    """
    return landmarks[::2]

def get_right_landmarks(landmarks):
    """
    Return the coordinates of the right landmarks (even lines of the landmarks array)
    """
    return landmarks[1::2]

def get_middle_line(left_ld, right_ld):
    """
    Return the middle line between the left and right landmarks
    """
    return (left_ld + right_ld) / 2
