import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


##### Mean and Std check #####
# ds = datasets.CIFAR10(root="./data", train=True, download=True,
#                       transform=transforms.ToTensor())

# x = torch.stack([img for img, _ in ds])
# print(x.mean(dim=(0, 2, 3)))
# print(x.std(dim=(0, 2, 3)))

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), #shift
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def stratified_split_idx(labels, test_size, seed):
    labels = np.asarray(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    all_idx = np.arange(len(labels))
    train_idx, test_idx = next(splitter.split(all_idx, labels))

    return train_idx, test_idx