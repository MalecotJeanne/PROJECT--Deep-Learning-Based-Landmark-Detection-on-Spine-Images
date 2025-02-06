import os
import torch
from loguru import logger
import cv2
import numpy as np
from os import path

from monai.data import DataLoader
from tqdm import tqdm

from utils import save_heatmaps, save_dataset
from metrics import pick_accuracy
from models.utils import ld2hm, make_landmarks
from transforms import testing_transforms

def centeroid(heat, gaussian_thresh = 0.5):
    # Parse center point of connected components
    # Return [p][xy]
    ret, heat = cv2.threshold(heat, gaussian_thresh, 1., cv2.THRESH_BINARY)
    heat = np.array(heat * 255., np.uint8)
    save_dir = "temp"
    os.makedirs(save_dir, exist_ok=True)
    random_nb = np.random.randint(0, 1000)
    os.makedirs(path.join(save_dir, str(random_nb)), exist_ok=True)
    cv2.imwrite(path.join(save_dir, str(random_nb), "bin_heatmap.jpg"), heat)

    # num: point number + 1 background
    num, labels = cv2.connectedComponents(heat)
    coords = []
    for label in range(1, num):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coords.append([cX, cY])
    return coords


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
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            inputs, landmarks = batch["image"], batch["landmarks"]
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            outputs = model(inputs)

            # Calculate accuracy
            pred_landmarks = make_landmarks(outputs, device)
            test_accuracy += accuracy(pred_landmarks, landmarks)

            # Save heatmaps
            name = batch["image_meta_dict"]["name"][-1]
            save_heatmaps(
                outputs[-1],
                os.path.join(results_dir, f"testing_heatmaps/{name}/preds"),
                basename="ld",
            )

            # Save images with landmarks
            pred_landmarks = pred_landmarks[0].cpu().numpy()
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

    test_accuracy /= len(test_loader)
    logger.success(f"Testing complete: Accuracy: {test_accuracy:.2f}%")
    with open(log_path, "a") as log_file:
        log_file.write(
            f"Testing Accuracy: {test_accuracy:.2f}%\n" f"====================\n"
        )

    print(f"Testing complete: Accuracy: {test_accuracy:.2f}%\n")
