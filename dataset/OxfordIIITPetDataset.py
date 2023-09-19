from torch.utils.data import Dataset
import torchvision
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np

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
    def __init__(self, root='../data', download=True, train=True, transform=None):
        if train:
            split = 'trainval'
        else:
            split = 'test'

        self.ds = torchvision.datasets.OxfordIIITPet(root=root,
                                                     target_types='segmentation',
                                                     download=download,
                                                     split=split)

        self.transform = transform



    def __getitem__(self, idx):
        data, seg = self.ds[idx]

        data = np.array(data, np.int16)
        seg = np.array(seg, np.int16)

        if self.transform:
            data_aug = self.transform(image=data, )
            data = data_aug['image']

            seg_aug = self.transform(image=seg, )
            seg = seg_aug['image']

        return data, seg


    def __len__(self):
        return len(self.ds)


def get_dataloader(**kwargs):
    transofrm = A.Compose(
        [
            A.LongestMaxSize(max_size=cfg['image_size']),
            A.PadIfNeeded(
                min_height=cfg['image_size'], min_width=cfg['image_size'], border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2()
        ]
    )

    '''if not dataset_mean or not dataset_std:
        get_dataset_mean_variance(SegmentOxfordIIITPetDataset(train=True, download=True, transform=transofrm))'''

    train_data = SegmentOxfordIIITPetDataset(train=True, download=True, transform=transofrm)
    test_data = SegmentOxfordIIITPetDataset(train=False, download=True, transform=transofrm)

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)