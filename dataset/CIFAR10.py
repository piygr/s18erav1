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


def get_dataset_labels():
    return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def get_data_label_name(idx):
    if idx < 0:
        return ''

    return get_dataset_labels()[idx]


def get_data_idx_from_name(name):
    if not name:
        return -1

    return get_dataset_labels.index(name.lower()) if name.lower() in get_dataset_labels() else -1



def get_dataloader(**kwargs):

    from config import vae_config as cfg

    dataset_mean, dataset_std = (0.4914, 0.4822, 0.4465), \
                                (0.2470, 0.2435, 0.2616)

    image_transform = T.Compose(
        [
            T.Resize((cfg['image_size'], cfg['image_size'])),
            T.ToTensor(),
            T.Normalize(mean=dataset_mean, std=dataset_std)
        ]
    )


    train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=image_transform)
    test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=image_transform)

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)