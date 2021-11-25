"""
train.py
export BACKBONE=resnet3d
export LEARNING_RATE=0.0001
export TO_TRAIN=TRUE
export NUM_EPOCHS=50
export BATCH_SIZE=4 
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ConvClassifier
import numpy as numpy
from torchvision import transforms
import torch.optim as optim
from torchvision import transforms as T
from dataset import ToyDataset
from utils import calculate_accuracy, weighted_binary_cross_entropy
import numpy as np

transforms = T.Compose([T.Normalize((0.43216,0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
save_path = '/scratch/users/mpike27/CS230/models/'
weights_const = [0.05, 0.95]

def save_model(model, model_path, train_losses, val_losses, val_accs):
    print("Saving Model...")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(model_path + "train_loss.npy", "wb") as f:
        np.save(f, train_losses)
    with open(model_path + "val_loss.npy", "wb") as f:
        np.save(f, val_losses)
    with open(model_path + "val_acc.npy", "wb") as f:
        np.save(f, val_accs)
    torch.save(model.cpu().state_dict(), 
                model_path + "model_dict.pth")

def main():
    backbone = sys.argv[1]
    learning_rate = float(sys.argv[2])
    to_train = bool(sys.argv[3])
    num_epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    save_name = sys.argv[6]
    delta = int(sys.argv[7])
    train_path = sys.argv[8]
    val_path = sys.argv[9]
    test_path = sys.argv[10]

    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'

    # Initialize Datasets
    # trainDataset = SoccerDataset(data_path=train_path)
    # valDataset = SoccerDataset(data_path=val_path)
    # testDataset = SoccerDataset(data_path=test_path)
    trainDataset = ToyDataset(100, 0.05, transform=transforms)
    valDataset = ToyDataset(10, 0.05, transform=transforms)
    testDataset = ToyDataset(10, 0.05, transform=transforms)

    # Initialize Data Loaders
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize Model
    model = ConvClassifier(backbone)
    model = model.cuda()
    criterion = weighted_binary_cross_entropy
    weights = torch.Tensor(weights_const)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    val_accs = []
    best_acc = -1
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        for i, data in enumerate(tqdm(trainLoader)):
            inputs, labels = data
            optimizer.zero_grad()
            inputs = inputs.to(device)
            output = model(inputs).cpu()
            loss = criterion(output, labels, weights)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        train_running_loss /= i + 1
        print(f"The loss from Epoch {epoch}: {train_running_loss}")
        train_losses.append(train_running_loss)
        model.eval()
        val_running_loss = 0.0
        preds = []
        actual = []
        for i, data in enumerate(tqdm(valLoader)):
            inputs, labels = data
            inputs = inputs.to(device)
            output = model(inputs).cpu()
            preds.extend(output)
            actual.extend(labels)
            loss = criterion(output, labels, weights)
            val_running_loss += loss.item()
        val_running_loss /= i + 1
        val_losses.append(val_running_loss)
        val_acc = calculate_accuracy(torch.Tensor(preds), torch.Tensor(actual), delta)
        val_accs.append(val_acc)
        print(f"The val loss and delta accuracy from Epoch {epoch}: {val_running_loss}, {val_accs[-1]}")
        if epoch == 0 or val_acc > best_acc:
            model_path = f"{save_path}{save_name}/"
            save_model(model, model_path, train_losses, val_losses, val_accs)
            model = model.to(device)
            best_acc = val_acc
    print(f"BEST VALIDATION ACCURACY WAS: {best_acc}")
    print("THE END")


if __name__ == '__main__':
    main()