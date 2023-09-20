import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def plot_prediction_sample(input, target, pred):
    plt.subplot(1, 3, 1)
    plt.imshow(input.cpu().permute(1, 2, 0))

    plt.subplot(1, 3, 2)
    plt.imshow(target.cpu().permute(1, 2, 0))

    plt.subplot(1, 3, 3)
    plt.imshow(pred.cpu().permute(1, 2, 0))
