import torch

def calculate_accuracy(preds, actual, delta):
    zeros = torch.zeros(delta)
    outer_range = torch.logical_or(torch.cat((actual[delta:], zeros), 0), torch.cat((zeros, actual[:delta])), 0).long()
    total_range = torch.logical_or(outer_range, actual).long()
    return float(torch.logical_and(total_range, preds).long().sum()) / len(total_range)