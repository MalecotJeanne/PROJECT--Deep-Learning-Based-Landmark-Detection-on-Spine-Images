import os
import torch
from loguru import logger
import cv2
import numpy as np
from os import path

from monai.data import DataLoader
from tqdm import tqdm

from utils import save_heatmaps, save_dataset, show_results
from metrics import pick_accuracy
from models.utils import ld2hm
from transforms import testing_transforms
from postprocess import split_corners, order_corners, centeroid, find_center_coords, find_corner_coords

import matplotlib.pyplot as plt


def test_model(dataset, model, chkpt_dir, results_dir, config, device, log_path):
    """
    Test the model using a provided test dataset.
    Args:
        dataset (Dataset): The dataset to be used for testing.
        model (torch.nn.Module): The trained model to be evaluated.
        chkpt_dir (str): Directory to load the best model checkpoint.
        results_dir (str): Directory to save testing results.
        config (dict): Configuration dictionary containing testing parameters.
        device (torch.device): The device to run the testing on (CPU or GPU).
        log_path (str): Path to the log file for logging testing progress.
    """
    # logger.add(log_path, mode="a")

    # Load the model from the best checkpoint
    chkpt_path = os.path.join(chkpt_dir, "best_val_loss.pt")
    if not os.path.exists(chkpt_path):
        logger.error(f"Checkpoint {chkpt_path} not found.")
        return 

    logger.info(f"Loading model from checkpoint: {chkpt_path}")
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval() 

    accuracy_name = config["train"]["accuracy"]
    accuracy, supported_accuracy = pick_accuracy(accuracy_name)
    if not supported_accuracy:
        logger.warning(
            f"Accuracy metric {accuracy_name} is not supported. Using PCK instead."
        )

    # Create the data loader
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    test_accuracy = 0.0

    logger.info("Starting testing...")
    with open(log_path, "a") as log_file:
        log_file.write(
            f"====================\nStarting testing...\n====================\n"
        )
    output_dataset = []
    final_coords = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            inputs, landmarks = batch["image"], batch["landmarks"]
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            gt_heatmaps = ld2hm(landmarks, inputs.shape[-2:], 6, config['train']['gt_heatmap'], device = device).cpu().numpy()

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            image_name = batch["image_meta_dict"]["name"][-1]

            np_heats = outputs.cpu().numpy()
            np_heats = np.clip(np_heats, 0., 1.)[0]     
            np_heats = np.transpose(np_heats, (1, 2, 0))  

            center_bin_heat = centeroid(np_heats[:, :, 4])
            center_coords = find_center_coords(center_bin_heat, np_heats[:, :, 5])

            
            bin_heats = [centeroid(np_heats[:, :, c]) for c in range(4)]
            flages = ["tl", "tr", "br", "bl"]

            coord_list = [find_corner_coords(bin_heat, center_coords, flag) for bin_heat, flag in zip(bin_heats, flages)]

            os.makedirs(path.join(results_dir, "coords"), exist_ok=True)
            
            with open(path.join(results_dir, "coords", image_name + ".txt"), "w") as file:
                for i, coords in enumerate(coord_list):
                    file.write(str(i) + "\n")
                    for coord in coords:
                        file.write(str(coord[0]) + " " + str(coord[1]) + "\n")

            #prepare file to save a figure with both heatmaps (bin and not bin)
            image_plot = inputs[0].detach().cpu().numpy()
            ground_truths = landmarks.detach().cpu().numpy()
            heatmap_dir = path.join(results_dir, "testing_heatmaps", image_name)
            invalid_heatmap_dir = path.join(results_dir, "testing_heatmaps", "invalid", image_name)            

            #if the nb of landmarks is incorrect, save the heatmaps and pass to the next image
            if len(coord_list[0]) != 17 or len(coord_list[1]) != 17 or len(coord_list[2]) != 17 or len(coord_list[3]) != 17 :
                os.makedirs(invalid_heatmap_dir, exist_ok=True)    
                for i in range (4):
                    plt.figure(figsize=(18, 10))

                    plt.subplot(1, 4, 1)
                    plt.imshow(image_plot[0], cmap='gray')
                    plt.imshow(np_heats[:, :, i], cmap='jet', alpha=0.3)
                    plt.axis('off')

                    plt.subplot(1, 4, 2)
                    plt.subplots_adjust(wspace=0)
                    plt.imshow(gt_heatmaps[0, i, :, :], cmap='jet')
                    plt.axis('off')

                    plt.subplot(1, 4, 3)
                    plt.subplots_adjust(wspace=0)
                    plt.imshow(np_heats[:, :, i], cmap='jet')
                    plt.axis('off')

                    plt.subplot(1, 4, 4)
                    plt.imshow(bin_heats[i], cmap="gray")
                    #dot at the position of each landmarks
                    for idx, landmark in enumerate(ground_truths[0]):
                        if idx % 4 == i:
                            #ground truth
                            plt.scatter(landmark[0], landmark[1], color='g', s=4)
                            circle = plt.Circle((landmark[0], landmark[1]), 10, color='g', fill=False, linestyle='--')
                            plt.gca().add_patch(circle)
                    for corner in coord_list[i]:
                        #predicted
                        plt.scatter(corner[0], corner[1], color='r', s=10, marker='x')
                    
                    plt.axis("off")
                    plt.savefig(path.join(invalid_heatmap_dir,  f"heatmap_{i+1}.png"))
                    plt.close()

                continue
             
            os.makedirs(heatmap_dir, exist_ok=True)
            
            #compute accuracy for each image with valid number of points             
            corners = np.array(order_corners(coord_list))   
            final_coords.append(corners)

            pred_landmarks = np.expand_dims(corners, axis=0)
            pred_landmarks = torch.from_numpy(pred_landmarks).float().to(device)
            sample_accuracy = accuracy(pred_landmarks, landmarks)
            test_accuracy += sample_accuracy

            for i in range (4):
                show_results(i, bin_heats, np_heats, gt_heatmaps, ground_truths, image_plot, corners, sample_accuracy, heatmap_dir)

            # Save images with landmarks
            pred_landmarks = corners
            image = inputs[0]

            # reverse transformations   
            test_transfo = testing_transforms(config["transforms"])
            image_meta_dict, landmarks_meta_dict = (
                batch["image_meta_dict"],
                batch["landmarks_meta_dict"],
            )
            new_batch = {
                "image": image,
                "landmarks": pred_landmarks,
                "image_meta_dict": image_meta_dict,
                "landmarks_meta_dict": landmarks_meta_dict,
            }
            new_batch = test_transfo.inverse(new_batch)
            image = new_batch["image"]
            pred_landmarks = new_batch["landmarks"]

            output_dataset.append({"image": image[0], "landmarks": pred_landmarks, "image_meta_dict": image_meta_dict, "landmarks_meta_dict": landmarks_meta_dict})  
            

    save_dataset(output_dataset, os.path.join(results_dir, "testing_images"), "pred")

    n_valid_images = len(final_coords)
    logger.warning(f"Number of dropped predictions: {len(test_loader) - n_valid_images}/{len(test_loader)}")

    test_accuracy_valid = test_accuracy/ n_valid_images
    test_accuracy_all = test_accuracy / len(test_loader)

    logger.success(f"Testing complete: \n\tAccuracy: {test_accuracy_all:.2f}% \n\tAccuracy (valid images): {test_accuracy_valid:.2f}%")
    with open(log_path, "a") as log_file:
        log_file.write(
            f"Testing Accuracy: {test_accuracy_all:.2f}%\n" f"====================\n"
        )

    print(f"Testing complete: Accuracy: {test_accuracy_all:.2f}%\n")

