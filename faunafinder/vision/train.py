import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models
from models import transform_input

from data.data_loader import ImageDataset, split_dataset
from utils.logger import Logger

logger = Logger(f"logs/training_{datetime.datetime.now()}.log")


def compute_accuracy(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def finetune_cnn(num_classes, num_epochs):
    # Load the VGG-16 models
    model = models.vgg16(weights='VGG16_Weights.DEFAULT')

    # Replace the last fully-connected layer of the VGG-16 models with a new one
    # that has the same number of outputs as the number of classes in the dataset
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)

    # Set the models to run on the GPU, if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # define the dataset and data loader
    dataset = ImageDataset(root_dir='data', num_classes=num_classes, transform=transform_input)

    # split the dataset into train and validation
    train_dataset, val_dataset = split_dataset(dataset, split_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Train the models
    logger.log("Starting training...")
    for epoch in range(num_epochs):
        logger.log(f"Starting epoch {epoch + 1}/{num_epochs}")

        # enter the main training loop
        step = 0
        for inputs, labels in train_loader:
            # convert labels to tensor
            # labels = torch.tensor(labels)

            # Move the inputs and labels to the GPU, if available
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # log training loss
            step += 1
            if step % 5 == 0:
                logger.log(f"Loss at step {step}: {loss.item():.4f}")

        # Log validation accuracy
        val_acc = compute_accuracy(model, val_loader)
        logger.log(f"Validation accuracy at end of epoch {epoch + 1}: {val_acc:.4f}")

    # Save the fine-tuned models
    torch.save(model, f"trained_models/finetuned_vg_166_{datetime.datetime.now()}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_classes", type=int, help="number of classes for final models")
    parser.add_argument("-e", "--num_epochs", type=int, help="number of epochs for training")

    args = parser.parse_args()

    finetune_cnn(args.num_classes, args.num_epochs)
