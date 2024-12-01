import os
import torch
import sys
import time
import wandb

from tqdm import tqdm
from loguru import logger
from monai.data import DataLoader

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from models.utils import make_same_type, make_landmarks
from utils import save_heatmaps, normalize_image, wandb_img
from losses import DistanceLoss, AdaptiveWingLoss, LandmarkAccuracy
from transforms import softmax2d

def train_model(dataset, model, chkpt_dir, results_dir, config, device, log_path):
    """
    TODO: Docstring
    """
    logger.add(log_path, mode="a")

    lr = config["train"]["learning_rate"]
    n_epochs = config["train"]["epochs"]

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

    # Set the optimizer
    optimizer_name = config["train"]["optimizer"]
    if optimizer_name == "adam" or optimizer_name == "Adam" or optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        logger.error(f"Optimizer {optimizer_name} not supported")
        return

    accuracy_name = config["train"]["accuracy"]
    logger.warning(f"Accuracy not implemented yet")

    # Init the wandb logger
    experiment_name = os.path.basename(log_path).split(".")[0]
    wandb.init(project="PRIM-project", config=config)
    wandb.run.name = experiment_name

    # Split the dataset into training and validation sets
    train_size = config["train"]["train_size"]
    trainset, valset = torch.utils.data.random_split(
        dataset, [train_size, 1 - train_size]
    )

    # Create the data loaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(valset, batch_size=config["batch_size"], shuffle=False)

    start_epoch = 0
    # Load the model from a checkpoint if provided
    if os.listdir(chkpt_dir):
        chkpt_dir = os.path.join(chkpt_dir, "last_epoch.pt")
        logger.info(f"Loading model from checkpoint: {chkpt_dir}")
        checkpoint = torch.load(chkpt_dir)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        logger.info("No checkpoint provided. Training from scratch.")

    model.to(device)

    wandb.watch(model)
    logger.info(f"To track model with wandb, go to: {wandb.run.get_url()}")

    # training
    print(f"\n====================\nStarting training...\n====================\n")
    # write in the log file
    with open(log_path, "a") as log_file:
        log_file.write(
            f"\n====================\nStarting training...\n====================\n\n"
        )

    loss_method = config["train"]["loss_method"]

    best_val_loss = float("inf")
    for epoch in range(start_epoch, n_epochs):

        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        epoch_time = time.time()
        for batch in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{n_epochs} (training)",
        ):
            inputs, landmarks = batch["image"], batch["landmarks"]
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)  # the outputs are heatmaps !  	
            print(f"sum: {torch.sum(outputs[0][0])}")
            outputs_, landmarks_ = make_same_type(outputs, landmarks, loss_method, device)

            loss = criterion(outputs_, landmarks_)
            loss.backward()
            optimizer.step()
         
            train_loss += loss.item()

            pred_landmarks = make_landmarks(outputs)
            train_accuracy += LandmarkAccuracy().evaluate(pred_landmarks, landmarks)

            # Empty the CUDA cache
            torch.cuda.empty_cache()

            # Delete unnecessary variables
            del inputs, landmarks, outputs_, landmarks_, pred_landmarks, loss
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        epoch_time = time.time() - epoch_time

        wandb.log({"train_loss": train_loss})
        wandb.log({"train_accuracy": train_accuracy})

        epoch_message = (
            f"Epoch [{epoch+1}/{n_epochs}] \n ----- \n"
            f"Training: \n"
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Time: {epoch_time:.2f}s \n"
        )
        print(
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )
        with open(log_path, "a") as log_file:
            log_file.write(epoch_message)

        # saving heatmaps
        save_heatmaps(
            outputs[-1],
            os.path.join(results_dir, f"training_heatmaps/epoch_{epoch+1}"),
            basename="ld",
        )
        # save normalized heatmaps in wandb
        for i in range(len(outputs[-1])):
            wandb.log({"training_heatmaps": wandb_img(outputs[-1][i], cmap="jet", caption=f"heatmap_{i}")})

        # validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        val_time = time.time()
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"(validation)",
            ):
                inputs, landmarks = batch["image"], batch["landmarks"]
                inputs, landmarks = inputs.to(device), landmarks.to(device)

                outputs = model(inputs)
                outputs = softmax2d(outputs)
                # outputs_ld = hm2ld(outputs, device)
                outputs_, landmarks_ = make_same_type(outputs, landmarks, loss_method, device)

                loss = criterion(outputs_, landmarks_)
                val_loss += loss.item()

                pred_landmarks = make_landmarks(outputs)
                val_accuracy += LandmarkAccuracy().evaluate(pred_landmarks, landmarks)

                # Empty the CUDA cache
                torch.cuda.empty_cache()

                # Delete unnecessary variables
                del inputs, landmarks, outputs_, landmarks_, pred_landmarks, loss
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_time = time.time() - val_time

        wandb.log({"val_loss": val_loss})
        wandb.log({"val_accuracy": val_accuracy})

        epoch_message = (
            f"Validation: \n"
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}% | "
            f"Time: {val_time:.2f}s \n"
            f"Total time epoch {epoch+1}: {epoch_time + val_time:.2f}s \n"
            f"====================\n"
        )
        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}% | "
            f"Time: {val_time:.2f}s \n\n"
            f"Total time epoch {epoch+1}: {epoch_time + val_time:.2f}s \n\n"
            f"====================\n"
        )
        with open(log_path, "a") as log_file:
            log_file.write(epoch_message)

        # saving heatmaps
        save_heatmaps(
            outputs[-1],
            os.path.join(results_dir, f"validation_heatmaps/epoch_{epoch+1}"),
            basename="ld",
        )
        # save heatmaps in wandb
        for i in range(len(outputs[-1])):
            wandb.log({"validation_heatmaps": wandb_img(outputs[-1][i], cmap="jet", caption=f"heatmap_{i}")})

        if val_loss < best_val_loss:
            # Save the model checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(chkpt_dir, f"best_val_loss.pt"))
            best_val_loss = val_loss

        # empty cuda cache
        torch.cuda.empty_cache()

    # Save the last model checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(chkpt_dir, f"last_epoch.pt"))
    logger.success("Finished Training: Youpi")
