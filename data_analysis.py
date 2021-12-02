import torch
import os
import matplotlib.pyplot as plt
from preprocessdata import ACTIONS

root_path = '/scratch/users/mpike27/CS230/data/'

def getLabels(split, clip_size):
    save_path = root_path + split + "/SoccerNet/Tensors/block_num_to_label_" + str(clip_size) + '.pt'
    return torch.load(save_path)

def calculateDistribution(labels):
    num_occurences = torch.bincount(labels)
    dist = torch.bincount(labels) / torch.sum(num_occurences)
    weights = torch.max(num_occurences + 1) / (num_occurences + 1)
    return dist, weights

def plot(dist, clip_size, split):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.locator_params(axis="x", nbins=18)
    plt.title(f'Distribution for Clip Size = {clip_size}')
    plt.xlabel('Class')
    plt.xticks(rotation = 90)
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.ylabel('Distribution')
    ax.bar(ACTIONS, dist)
    plt.savefig(f'Clip_size_{clip_size}_split_{split}.png')

def main():
    with open('data_analysis.txt', 'w') as txt:
        for clip_size in [5, 20, 60]:
            txt.write(f'\nClip_size = {clip_size}\n')
            for split in ['train', 'val', 'test']:
                labels = getLabels(split, clip_size)
                dist, weights = calculateDistribution(torch.Tensor(labels).int())
                txt.write(f"Distribution for {split} set:\n")
                if split == 'train':
                    txt.write(f"Appropriate weighting: {weights}\n")
                txt.write(f"{dist}\n")
                plot(dist, clip_size, split)

if __name__ == '__main__':
    main()