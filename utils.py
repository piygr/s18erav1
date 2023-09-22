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
    i = 0
    if input:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(input.cpu().permute(1, 2, 0))


    if target:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(target.cpu().permute(1, 2, 0))

    if pred:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(pred.cpu().permute(1, 2, 0))


def plot_vae_images(input_imgs, input_labels, pred_imgs):
    image_count = len(input_imgs)
    cols = 4
    rows = 10
    c = 1
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if j == 1:
                plt.subplot(rows, cols, c)
                plt.imshow(input_imgs[i - 1].cpu().permute(1, 2, 0))
                plt.title('-Input image-', fontsize=10)

            else:
                plt.subplot(rows, cols, c)
                plt.imshow(pred_imgs[i - 1][j - 2].detach().cpu().permute(1, 2, 0))
                plt.title('Input label: ' + str(input_labels[i - 1][j - 2]), fontsize=10)

            c += 1