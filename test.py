import os
import torch
from loguru import logger

from monai.data import DataLoader
from monai.transforms import Compose, Invertd

from tqdm import tqdm
import wandb

from utils import save_heatmaps, normalize_image, get_last_folder
from losses import DistanceLoss, AdaptiveWingLoss, LandmarkAccuracy
from models.utils import hm2ld, make_same_type, make_landmarks
from transforms import testing_transforms


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
    logger.add(log_path, mode="a")

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

    # Create the data loader
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    test_accuracy = 0.0

    logger.info("Starting testing...")
    with open(log_path, "a") as log_file:
        log_file.write(
            f"====================\nStarting testing...\n====================\n"
        )

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            inputs, landmarks = batch["image"], batch["landmarks"]
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            outputs = model(inputs)

            # Calculate accuracy
            true_landmarks = landmarks.cpu().numpy()
            pred_landmarks = make_landmarks(outputs)
            test_accuracy += LandmarkAccuracy(10).evaluate(
                pred_landmarks, true_landmarks
            )

            # Save heatmaps
            save_heatmaps(
                outputs[-1],
                os.path.join(results_dir, "testing_heatmaps"),
                basename="ld",
            )

            # Save images with landmarks
            pred_landmarks = pred_landmarks[0]
            image = inputs[0]
            print(type(image), type(pred_landmarks))
            #reverse transformations
            test_transfo = testing_transforms(config["transforms"])
            transforms = Invertd(test_transfo)
            batch = {"image": image, "landmarks": pred_landmarks}
            batch = transforms(batch)
            image = batch["image"]
            pred_landmarks = batch["landmarks"]
            print(image.shape)
            print(pred_landmarks.shape)



    logger.success(
        f"Testing complete: Accuracy: {test_accuracy:.2f}%"
    )
    with open(log_path, "a") as log_file:
        log_file.write(
            f"Testing Accuracy: {test_accuracy:.2f}%\n"
            f"====================\n"
        )

    print(
        f"Testing complete: Accuracy: {test_accuracy:.2f}%\n"
    )
