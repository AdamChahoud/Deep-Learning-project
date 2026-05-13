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

def stratified_split_indices(labels, test_size, seed):
    labels = np.asarray(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    all_idx = np.arange(len(labels))
    train_idx, test_idx = next(splitter.split(all_idx, labels))

    return train_idx, test_idx

#### Gives balanced amongs classes random num_labeled indices ####
def make_labeled_indices(labels, candidate_indices, num_labeled, seed, num_classes):
    labels = np.asarray(labels)
    candidate_indices = np.asarray(candidate_indices)
    if num_labeled % num_classes != 0:
        raise ValueError("num_labeled must be divisible by num_classes")
    per_class = num_labeled // num_classes
    rng = np.random.default_rng(seed)
    chosen = []

    for class_id in range(num_classes):
        class_idx = candidate_indices[labels[candidate_indices] == class_id]

        if len(class_idx) < per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_idx)} samples, "
                f"while {per_class} samples are required"
            )
        chosen_class_idx = rng.choice(class_idx, per_class, replace=False)
        chosen.extend(chosen_class_idx.tolist())

    chosen = np.array(chosen)
    rng.shuffle(chosen)

    return chosen