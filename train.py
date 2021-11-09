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

def main():
    backbone = sys.argv[1]
    learning_rate = float(sys.argv[2])
    to_train = bool(sys.argv[3])
    num_epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    momentum = float(sys.argv[6])
    save_name = sys.argv[7]
    delta = int(sys.argv[8])
    train_path = sys.argv[9]
    val_path = sys.argv[10]
    test_path = sys.argv[11]

    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'

    # Initialize Datasets
    trainDataset = SoccerDataset(data_path=train_path)
    valDataset = SoccerDataset(data_path=val_path)
    testDataset = SoccerDataset(data_path=test_path)

    # Initialize Data Loaders
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, num_workers)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers)

    # Initialize Model
    model = ConvClassifier(backbone)
    model = model.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        for i, data in enumerate(tqdm(trainLoader)):
            inputs, labels = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"The loss from Epoch {epoch}: {train_running_loss}")
        train_losses.append(train_running_loss)
        model.eval()
        val_running_loss = []
        preds = []
        actual = []
        for i, data in enumerate(tqdm(valLoader)):
            inputs, labels = data
            output = model(inputs)
            preds.extend(output)
            actual.extend(labels)
            loss = criterion(output, labels)
            val_running_loss += loss.item()
        val_losses.append(val_running_loss)
        val_accs.append(calculate_accuracy(preds, actual, delta))
        print(f"The val loss and delta accuracy from Epoch {epoch}: {val_running_loss}, {val_accs[-1]}")
    print("Saving Model...")
    model_path = f"./models/{save_name}/"
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

if __name__ == '__main__':
    main()