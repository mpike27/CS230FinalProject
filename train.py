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
from model import ConvClassifier, ConvLSTMClassifier
import numpy as numpy
from torchvision import transforms
import torch.optim as optim
from torchvision import transforms as T
from dataset import ToyDataset, SoccerDataset
from utils import *
import numpy as np

transforms = T.Compose([T.Normalize((0.43216,0.394666, 0.37645), (0.22803, 0.22145, 0.216989))])
save_path = '/scratch/users/mpike27/CS230/models/'
binary_weights_const = [0.05, 0.95]
clip_size_to_weights = {
    5: [2.2175e+02, 1.9626e+01, 3.2439e+01, 1.5316e+02, 1.7077e+02, 2.9714e+02, \
        7.7785e+01, 3.3766e+02, 5.4025e+01, 7.5416e+01, 3.3016e+02, 1.8571e+03, \
        4.1269e+02, 4.1269e+02, 7.4285e+03, 7.4285e+03, 1.4857e+04, 1.0000e+00],
    20: [2.9966e+01, 3.1127e+00, 4.8306e+00, 2.7625e+01, 2.5257e+01, 3.7617e+01, \
        1.1787e+01, 4.0182e+01, 7.6537e+00, 1.0780e+01, 5.8933e+01, 2.5257e+02, \
        5.0514e+01, 6.8000e+01, 1.7680e+03, 8.8400e+02, 1.7680e+03, 1.0000e+00],
    60: [  6.6286,   1.0357,   1.0000,   5.6585,   5.3953,   7.0303,   2.4681, \
          7.0303,   1.8125,   2.2308,  15.4667,  77.3333,  10.0870,  25.7778, \
        232.0000, 116.0000, 232.0000,   1.5890]
}

def save_model(model, model_path, train_losses, val_losses, val_f1s):
    print("Saving Model...")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(model_path + "train_loss.npy", "wb") as f:
        np.save(f, train_losses)
    with open(model_path + "val_loss.npy", "wb") as f:
        np.save(f, val_losses)
    with open(model_path + "val_f1.npy", "wb") as f:
        np.save(f, val_f1s)
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
    clip_size = int(sys.argv[11])
    hidden_size = int(sys.argv[12])
    num_classes = int(sys.argv[13])
    model_type = sys.argv[14]
    num_layers = int(sys.argv[15])

    print('backbone: ', backbone)
    print('learning_rate: ', learning_rate)
    print('to_train: ', to_train)
    print('num_epochs: ', num_epochs)
    print('batch_size: ', batch_size)
    print('save_name: ', save_name)
    print('delta: ', delta)
    print('train_path: ', train_path)
    print('val_path: ', val_path)
    print('test_path: ', test_path)
    print('clip_size: ', clip_size)
    print('hidden_size: ', hidden_size)
    print('num_classes: ', num_classes)
    print('model_type: ', model_type)
    print('num_layers: ', num_layers)

    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'

    # Initialize Datasets
    trainDataset = SoccerDataset(data_path=train_path, clip_size=clip_size, transform=transforms)
    valDataset = SoccerDataset(data_path=val_path, clip_size=clip_size, transform=transforms)
    testDataset = SoccerDataset(data_path=test_path, clip_size=clip_size, transform=transforms)
    # trainDataset = ToyDataset(100, num_classes, transform=transforms)
    # valDataset = ToyDataset(10, num_classes, transform=transforms)
    # testDataset = ToyDataset(10, num_classes, transform=transforms)

    # Initialize Data Loaders
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=False, num_workers=2)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=False, num_workers=2)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize Model
    if model_type == 'LSTM':
        model = ConvLSTMClassifier(backbone, hidden_size, num_layers, num_classes)
    elif model_type == 'CONV':
        model = ConvClassifier(backbone, num_classes)
    else:
        raise Exception("Invalid Model Type")
    model = model.cuda()
    weights = torch.Tensor(clip_size_to_weights[clip_size])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    val_f1s = []
    best_f1 = -1
    model_path = f"{save_path}{save_name}/"
    prev_logits = [torch.zeros(18) for i in range(4)]
    for epoch in range(num_epochs):
        with torch.set_grad_enabled(True):
            model.train()
            train_running_loss = 0.0
            print('Training')
            for i, data in enumerate(trainLoader):
                inputs, labels = data
                # print(f"One train class is {labels}")
                optimizer.zero_grad()
                inputs = inputs.to(device)
                
                output = model(inputs).cpu()
                loss = criterion(output, labels.long())
                loss.backward()
                optimizer.step()
                inputs = inputs.to(0)
                train_running_loss += loss.item()
            train_running_loss /= i + 1
            print(f"The loss from Epoch {epoch}: {train_running_loss}")
            train_losses.append(train_running_loss)
        
        with torch.set_grad_enabled(False):
            model.eval()
            val_running_loss = 0.0
            preds = []
            actual = []
            print("Evaluating")
            for i, data in enumerate(valLoader):
                inputs, labels = data
                inputs = inputs.to(device)
                output = model(inputs).cpu()
                preds.extend(output)
                actual.extend(labels)
                loss = criterion(output, labels.long())
                inputs = inputs.to(0)
                val_running_loss += loss.item()
            val_running_loss /= i + 1
            val_losses.append(val_running_loss)
            # breakpoint()
            print(preds[0])
            print(prev_logits[0])
            prev_logits = preds
            preds = torch.Tensor([logit.argmax() for logit in preds])
            # val_f1 = calculate_accuracy(preds, torch.Tensor(actual))
            val_f1 = calculate_f1(preds, torch.Tensor(actual), num_classes)
            val_f1s.append(val_f1)
            print(f"Actual: {torch.Tensor(actual)}")
            print(f"Preds: {preds}")
            print(f"The val loss and delta f1 from Epoch {epoch}: {val_running_loss}, {val_f1s[-1]}")
            if epoch == 0 or val_f1 > best_f1:
                save_model(model, model_path, train_losses, val_losses, val_f1s)
                model = model.to(device)
                best_f1 = val_f1
        plot_metrics(model_path, train_losses, val_losses, val_f1s)
    print(f"BEST VALIDATION F1 WAS: {best_f1}")
    print("THE END")


if __name__ == '__main__':
    main()