import os
import torch
from loguru import logger

from monai.data import DataLoader

from tqdm import tqdm
import wandb

from utils import save_heatmaps, normalize_image, get_last_folder
from losses import DistanceLoss, AdaptiveWingLoss
from models.utils import hm2ld, calculate_accuracy, make_same_type


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
    checkpoint = torch.load(chkpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Init the wandb logger
    experiment_name = os.path.basename(log_path).split(".")[0]
    wandb.init(project="PRIM-project", config=config)
    wandb.run.name = "testing_" + experiment_name

    # Create the data loader
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    test_loss = 0.0
    test_accuracy = 0.0

    # Set the loss criterion
    criterion_name = config["train"]["criterion"]
    if criterion_name == "mse" or criterion_name == "MSE":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "distance" or criterion_name == "DistanceLoss":
        criterion = DistanceLoss()
    elif (
        criterion_name == "adaptive_wing"
        or criterion_name == "AdaptiveWing"
        or criterion_name == "AdapWingLoss"
    ):
        criterion = AdaptiveWingLoss()
    elif criterion_name == "l1" or criterion_name == "L1":
        criterion = torch.nn.L1Loss()
    elif (
        criterion_name == "cross_entropy_loss"
        or criterion_name == "CrossEntropyLoss"
        or criterion_name == "CELoss"
    ):
        criterion = torch.nn.CrossEntropyLoss()
    else:
        logger.error(f"Criterion {criterion_name} not supported")
        return

    logger.info("Starting testing...")
    with open(log_path, "a") as log_file:
        log_file.write(
            f"====================\nStarting testing...\n====================\n"
        )

    loss_method = config["train"]["loss_method"]

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            inputs, landmarks = batch["image"], batch["landmarks"]
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            outputs = model(inputs)
            outputs_ld = hm2ld(outputs, device)
            outputs_, landmarks_ = make_same_type(outputs, landmarks, loss_method,device)

            loss = criterion(outputs_, landmarks_)
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs_, landmarks_)

            # Save heatmaps
            save_heatmaps(
                outputs[-1],
                os.path.join(results_dir, "testing_heatmaps"),
                basename="ld",
            )
            for i in range(len(outputs[-1])):
                wandb.log(
                    {
                        "testing_heatmaps": [
                            wandb.Image(
                                normalize_image(outputs[-1][i].cpu().detach().numpy()),
                                caption=f"heatmap_{i}",
                            )
                        ]
                    }
                )

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    logger.success(
        f"Testing complete: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%"
    )
    with open(log_path, "a") as log_file:
        log_file.write(
            f"Testing Loss: {test_loss:.4f} | Testing Accuracy: {test_accuracy:.2f}%\n"
            f"====================\n"
        )

    print(
        f"Testing complete: Loss: {test_loss:.4f} | "
        f"Accuracy: {test_accuracy:.2f}%\n"
    )
