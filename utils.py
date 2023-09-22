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
    if input is not None:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(input.cpu().permute(1, 2, 0))


    if target is not None:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(target.cpu().permute(1, 2, 0))

    if pred is not None:
        i += 1
        plt.subplot(1, 3, i)
        plt.imshow(pred.cpu().permute(1, 2, 0))


def plot_vae_images(input_imgs, input_labels, pred_imgs, cols=5, rows=10, label_fn=None):

    plt.figure(figsize=(6, 12))
    c = 1
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if j == 1:
                plt.subplot(rows, cols, c)
                plt.tight_layout()
                plt.imshow(input_imgs[i - 1].cpu().permute(1, 2, 0), aspect='auto')
                plt.title('-Input image-', fontsize=8)
                plt.xticks([])
                plt.yticks([])

            else:
                lbl = input_labels[i - 1][j - 2]
                if label_fn:
                    lbl = label_fn(lbl)
                plt.subplot(rows, cols, c)
                plt.tight_layout()
                plt.imshow(pred_imgs[i - 1][j - 2].detach().cpu().permute(1, 2, 0), aspect='auto')
                plt.title('Input label: ' + str(lbl), fontsize=8)
                # plt.title(str(input_labels[i - 1][j - 2]), fontsize=8)
                plt.xticks([])
                plt.yticks([])

            c += 1

    plt.show()