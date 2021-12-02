import torch
from sklearn import metrics
import matplotlib.pyplot as plt

def weighted_binary_cross_entropy(output, target, weights=None):
    """
    cc: https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114
    """
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))



def calculate_accuracy_binary(preds, actual, delta):
    zeros = torch.zeros(delta)
    outer_range = torch.logical_or(torch.cat((actual[delta:], zeros), 0), torch.cat((zeros, actual[:-delta]), 0)).long()
    total_range = torch.logical_or(outer_range, actual).long()
    preds = (preds > 0.5).float()
    positive_accuracy = float(torch.logical_and(total_range, preds).long().sum()) / len(total_range)
    negative_accuracy = float(torch.logical_and(torch.logical_not(actual), torch.logical_not(preds)).long().sum()) / len(total_range)
    return (positive_accuracy + negative_accuracy) / 2.

def calculate_accuracy(preds, actual):
    return float(preds[preds == actual].shape[0]) / preds.shape[0]

def create_one_hot(size, idx):
    if (type(idx) == 'int' and idx >= size) or torch.any(idx > size[-1]): \
        raise Exception("Don't do that dummy")
    if type(idx) == 'int':
        idx = [idx]
    tensor = torch.zeros(size)
    for i, elem in enumerate(idx):
        tensor[i,elem.long()] = 1
    return tensor

def calculate_f1(preds, actual, num_classes):
    # breakpoint()
    conf = metrics.confusion_matrix(actual, preds)
    report = metrics.classification_report(actual, preds, digits=3, output_dict=True, zero_division=1)
    f1 = 0.0
    for i in range(num_classes):
        key = str(float(i))
        if key in report:
            f1 += report[key]['f1-score']
    return f1 / num_classes

def plot_metrics(directory, train_losses, val_losses, val_f1s):
    plt.clf()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(train_losses, label="Training Loss")
    axs[0, 0].set_title('Training Loss')
    axs[0, 1].plot(val_losses, label="Validation Loss")
    axs[0, 1].set_title('Validation Loss')
    axs[1, 0].plot(val_f1s, label="Validation F1s")
    axs[1, 0].set_title('Validation F1s')
    plt.tight_layout()
    plt.savefig(directory+'AllPlots.png')
    plt.close()