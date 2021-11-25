import torch

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

def calculate_accuracy(preds, actual, delta):
    zeros = torch.zeros(delta)
    outer_range = torch.logical_or(torch.cat((actual[delta:], zeros), 0), torch.cat((zeros, actual[:-delta]), 0)).long()
    total_range = torch.logical_or(outer_range, actual).long()
    preds = (preds > 0.5).float()
    positive_accuracy = float(torch.logical_and(total_range, preds).long().sum()) / len(total_range)
    negative_accuracy = float(torch.logical_and(torch.logical_not(actual), torch.logical_not(preds)).long().sum()) / len(total_range)
    return (positive_accuracy + negative_accuracy) / 2.