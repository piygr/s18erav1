from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T

import torch

dataset_mean = None
dataset_std = None

def get_dataset_mean_variance(dataset):
    imgs = [item[0] for item in dataset]
    imgs = torch.stack(imgs, dim=0)

    mean = []
    std = []
    for i in range(imgs.shape[1]):
        mean.append(imgs[:, i, :, :].mean().item())
        std.append(imgs[:, i, :, :].std().item())

    return tuple(mean), tuple(std)


def get_dataloader(**kwargs):

    dataset_mean = (0.1307,)
    dataset_std = (0.3081,)

    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=dataset_mean, std=dataset_std)
        ]
    )


    train_data = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=image_transform)
    test_data = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=image_transform)

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)