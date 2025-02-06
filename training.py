"""
This script contains the training function for all the model, depending on the configuration file.
Author: Jeanne Malécot
"""
import os
import sys
import time
import re 

import torch
from loguru import logger
from monai.data import DataLoader
from tqdm import tqdm
import wandb

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_folder_path)

from metrics import pick_criterion, pick_accuracy
from models.utils import make_landmarks, make_same_type
from utils import save_heatmaps, wandb_img

def train_model(dataset, model, chkpt_dir, results_dir, config, device, log_path):
    """
    Train the model on the dataset, using the configuration provided.
    Args:
        dataset (Dataset): the dataset to train on
        model (torch.nn.Module): the model to train
        chkpt_dir (str): the directory to save the checkpoints, and eventually load the model from
        results_dir (str): the directory to save the results
        config (str): the configuration file (yaml)
        device (str): the device to use for training
        log_path (str): the path to the log file
    """
    # logger.add(log_path, mode="a")

    lr = config["train"]["learning_rate"]
    n_epochs = config["train"]["epochs"]

    # Set the loss criterion
    criterion_name = config["train"]["criterion"]
    criterion, supported_criterion = pick_criterion(criterion_name)
    if not supported_criterion:
        logger.error(f"Criterion {criterion_name} not supported. Using MSEloss instead.")
        
    # Set the accuracy metric
    accuracy_name = config["train"]["accuracy"]
    accuracy, supported_accuracy = pick_accuracy(accuracy_name)
    if not supported_accuracy:
        logger.error(f"Accuracy {accuracy_name} not supported. Using NMEaccuracy instead.")

    # Set the optimizer
    optimizer_name = config["train"]["optimizer"]
    if optimizer_name == "adam" or optimizer_name == "Adam" or optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        logger.error(f"Optimizer {optimizer_name} not supported. Using Adam instead.")  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set the scheduler
    scheduler = config["train"]["scheduler"]
    scheduler_type = scheduler["type"]
    step_size = scheduler["step_size"]
    gamma = scheduler["gamma"]
    if scheduler_type == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Init the wandb logger
    if os.listdir(chkpt_dir):
        print("resume wandb")
        url_pattern = r"https://wandb\.ai/\S+"
        with open(log_path, "r") as file:
            log_content = file.read()
        url = re.findall(url_pattern, log_content)[0]
        #use os to make url a path
        print(url)
        id_wandb = os.path.basename(url)
        print(id_wandb) 
        wandb.init(project="PRIM-project", id = id_wandb, resume = "allow" ,config=config)
        
    else:
        experiment_name = os.path.basename(log_path).split(".")[0]
        wandb.init(project="PRIM-project", config=config, resume = "allow")
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

        with open(os.path.join(chkpt_dir, "last_epoch.pt"), "rb") as f:
            checkpoint = torch.load(f)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        
        print(f"\n----- Resuming training from epoch {start_epoch+1} -----\n")
        with open(log_path, "a") as log_file:
            log_file.write(f"\n----- Resuming training from epoch {start_epoch+1} -----\n")


        del checkpoint
        torch.cuda.empty_cache()
        
    else: 
        print(f"\n====================\nStarting training...\n====================\n")
        with open(log_path, "a") as log_file:
            log_file.write("\n====================\nStarting training...\n====================\n\n")

    model.to(device)

    wandb.watch(model)
    logger.info(f"To track model with wandb, go to: {wandb.run.get_url()}")

    loss_method = config["train"]["loss_method"]

    best_val_loss = float("inf")
    best_val_accuracy = -float("inf")

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
            outputs_, gtruths_ = make_same_type(outputs, landmarks, loss_method, config['train']['gt_heatmap'], device)

            #save ground truth heatmaps at the first epoch
            if epoch == 0 and loss_method == "heatmap":
                for b in range(len(outputs)):
                    if b % 100 == 0:
                        name = batch["image_meta_dict"]["name"][b]
                        save_heatmaps(
                            gtruths_[b],
                            os.path.join(results_dir, f"gt_heatmaps/{name}"),
                            basename="ld",
                        )

            #save heatmaps every 10 epochs
            if epoch % 10 == 0:
                for b in range(len(outputs)):
                    if b % 100 == 0: 
                        name = batch["image_meta_dict"]["name"][b]
                        save_heatmaps(
                            outputs[b],
                            os.path.join(results_dir, f"training_heatmaps/epoch_{epoch+1}/{name}"),
                            basename="ld",
                        )

            loss = criterion(outputs_, gtruths_)
            loss.backward()
            optimizer.step()
         
            train_loss += loss.item()

            pred_landmarks = make_landmarks(outputs)
            true_landmarks = landmarks.cpu().detach().numpy()
            train_accuracy += accuracy(pred_landmarks, true_landmarks)

            # Empty the CUDA cache
            torch.cuda.empty_cache()

            # Delete unnecessary variables
            del inputs, landmarks, outputs_, gtruths_, pred_landmarks, true_landmarks, loss        
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        epoch_time = time.time() - epoch_time

        wandb.log({"epoch": epoch})
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

        # save normalized heatmaps in wandb
        for i in range(len(outputs[-1])):
            wandb.log({"training_heatmaps": wandb_img(outputs[-1][i], cmap="jet", caption=f"heatmap_{i}")})

        # Step the scheduler
        scheduler.step()

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
                outputs_, gtruths_ = make_same_type(outputs, landmarks, loss_method, config['train']['gt_heatmap'], device)

                #save heatmaps every 10 epochs
                if epoch % 10 == 0:
                    for b in range(len(outputs)):
                        if b % 100 == 0:
                            name = batch["image_meta_dict"]["name"][b]
                            save_heatmaps(
                                outputs[b],
                                os.path.join(results_dir, f"validation_heatmaps/epoch_{epoch+1}/{name}"),
                                basename="ld",
                            )

                loss = criterion(outputs_, gtruths_)
                val_loss += loss.item()

                pred_landmarks = make_landmarks(outputs)
                true_landmarks = landmarks.cpu().detach().numpy()
                val_accuracy += accuracy(pred_landmarks, true_landmarks)

                # Empty the CUDA cache
                torch.cuda.empty_cache()

                # Delete unnecessary variables
                del inputs, landmarks, outputs_, gtruths_, pred_landmarks, true_landmarks, loss
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

        if val_accuracy > best_val_accuracy:
            # Save the model checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(chkpt_dir, f"best_val_accuracy.pt"))
            best_val_accuracy = val_accuracy

         # Save the last model checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(chkpt_dir, f"last_epoch.pt"))

        # empty cuda cache
        torch.cuda.empty_cache()

    logger.success("Finished Training: Youpi!")
    logger.info(f"--- Best validation loss: {best_val_loss}")
    logger.info(f"--- Best validation accuracy: {best_val_accuracy}")

    wandb.finish()
