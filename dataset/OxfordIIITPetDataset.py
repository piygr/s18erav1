from torch.utils.data import Dataset
import torchvision
import torch

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
    train_data = SegmentOxfordIIITPetDataset(train=True, download=True)
    test_data = SegmentOxfordIIITPetDataset(train=False, download=True)

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)