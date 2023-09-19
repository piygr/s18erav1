import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize(img, mean, std):
    MEAN = torch.tensor(mean)
    STD = torch.tensor(std)

    img = img * STD[:, None, None] + MEAN[:, None, None]
    i_min = img.min().item()
    i_max = img.max().item()

    img_bar = (img - i_min)/(i_max - i_min)

    return img_bar

def plot_image_seg(img, seg, mean, std):
    plt.subplot(1, 2, 1)
    img = np.array(img, np.int16)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    seg = np.array(seg, np.int16)
    plt.imshow(seg)