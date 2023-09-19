from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T

import torch

def get_dataset_mean_variance(dataset):

    global dataset_mean
    global dataset_std

    if dataset_mean and dataset_std:
        return dataset_std, dataset_std

    else:
        imgs = [item[0] for item in dataset]
        imgs = torch.stack(imgs, dim=0)

        mean = []
        std = []
        for i in range(imgs.shape[1]):
            mean.append(imgs[:, i, :, :].mean().item())
            std.append(imgs[:, i, :, :].std().item())

        dataset_mean = tuple(mean)
        dataset_std = tuple(std)

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
                                                     split=split,
                                                     transform=transform)



    def __getitem__(self, idx):
        data, seg = self.ds[idx]

        return data, seg

    def __len__(self):
        return len(self.ds)


def get_dataloader(**kwargs):
    transofrm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize()
        ]
    )

    if not dataset_mean or dataset_std:
        get_dataset_mean_variance(SegmentOxfordIIITPetDataset(train=True, download=True, transform=transofrm))

    train_data = SegmentOxfordIIITPetDataset(train=True, download=True, transform=transofrm)
    test_data = SegmentOxfordIIITPetDataset(train=False, download=True, transform=transofrm)

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)