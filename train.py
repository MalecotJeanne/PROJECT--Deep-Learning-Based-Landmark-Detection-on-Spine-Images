"""
TODO: Docstring
"""

import os

import torch

from monai.data import DataLoader


def train_model(dataset, model, chkpt_dir, config, device, logger):
    """
    TODO: Docstring
    """
    lr = config["train"]["learning_rate"]
    n_epochs = config["train"]["epochs"]

    # Set the loss criterion
    criterion_name = config["train"]["criterion"]
    if criterion_name == "mse" or criterion_name == "MSE":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "cross_entropy" or criterion_name == "CrossEntropy":
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

    # Split the dataset into training and validation sets
    train_size = config["train"]["train_size"]
    trainset, valset = torch.utils.data.random_split(
        dataset, [train_size, 1 - train_size]
    )

    # Create the data loaders
    train_loader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(valset, batch_size=config["batch_size"], shuffle=False)

    start_epoch = 0
    # Load the model from a checkpoint if provided
    if chkpt_dir is not None:
        logger.info(f"Loading model from checkpoint: {chkpt_dir}")
        checkpoint = torch.load(chkpt_dir)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        logger.info("No checkpoint provided. Training from scratch.")

    model.to(device)

    # training

    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, landmarks = batch["image"], batch["landmarks"]
            inputs, landmarks = inputs.to(device), landmarks.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() 

        train_loss /= train_size
        logger.info(f"Epoch [{epoch+1}/{n_epochs}], Training Loss: {train_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, landmarks = batch["image"], batch["landmarks"]
                inputs, landmarks = inputs.to(device), landmarks.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, landmarks)
                val_loss += loss.item()

        val_loss /= (1 - train_size)
        logger.info(f"Epoch [{epoch+1}/{n_epochs}], Validation Loss: {val_loss:.4f}")

        # Save the model checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        # check if the directory exists
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        torch.save(checkpoint, os.path.join(chkpt_dir, f"model_{epoch}.pt"))

    logger.info("Finished Training")
