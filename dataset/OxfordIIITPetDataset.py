from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T
import numpy as np
from PIL import Image

from config import get_config
cfg = get_config(ds='unet')

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


class SegmentOxfordIIITPetDataset(Dataset):
    def __init__(self, root='../data', download=True, train=True, target_transform=None, mask_transform=None):
        if train:
            split = 'trainval'
        else:
            split = 'test'

        self.ds = torchvision.datasets.OxfordIIITPet(root=root,
                                                     target_types='segmentation',
                                                     download=download,
                                                     split=split)

        self.target_transform = target_transform
        self.mask_transform = mask_transform


    def __getitem__(self, idx):
        data, seg = self.ds[idx]


        if self.target_transform:
            data = self.target_transform(data)

        if self.mask_transform:
            seg = self.mask_transform(seg)

        return data, seg


    def __len__(self):
        return len(self.ds)


def get_dataloader(**kwargs):

    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    image_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=dataset_mean, std=dataset_std)
        ]
    )

    mask_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor()
        ]
    )


    train_data = SegmentOxfordIIITPetDataset(train=True, download=True, target_transform=image_transform, mask_transform=mask_transform)
    test_data = SegmentOxfordIIITPetDataset(train=False, download=True, target_transform=image_transform, mask_transform=mask_transform)

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)