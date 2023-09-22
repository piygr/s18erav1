import torch.nn as nn

from dataset.MNIST import MultiChannelMNIST
from models.UNet import UNet
#from models.VAE import VAE
from torchsummary import summary
import pytorch_lightning as pl

from torchvision import transforms as T

from loss import bce_loss, dice_loss
from utils import device, plot_vae_images
import random
import torch

def init(
        train_dataloader,
        val_dataloader,
        model,
        cfg=None,
        in_channels=3,
        out_channels=1,
        show_summary=False,
        max_lr=None,
        loss_fn=bce_loss,
        upsample='transpose_conv',
        downsample='maxpool',
        accelerator=None
):

    '''if net == 'UNet':


    else:
        cfg = vae_config

        #enc_out_dim=512, latent_dim=256, input_height=28, num_embed=10
        model = VAE(
            enc_out_dim=cfg['enc_out_dim'],
            latent_dim=cfg['latent_dim'],
            num_embed=cfg['num_classes'],
            input_height=cfg['image_size']
        )'''

    if show_summary:
        summary(model.to(device), input_size=(in_channels, cfg['image_size'], cfg['image_size']))

    trainer_args = dict(
        precision='16',
        max_epochs=cfg['num_epochs']
    )

    if accelerator:
        #trainer_args['precision'] = '16-mixed'
        trainer_args['accelerator'] = accelerator

    trainer = pl.Trainer(
        **trainer_args
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return model

def validate_vae(net, cfg, val_data, count=40, label_fn=None):
    image_transform = T.Compose(
        [
            T.Resize((cfg['image_size'], cfg['image_size'])),
            T.ToTensor()
        ]
    )

    test_data = val_data #MultiChannelMNIST(root='../data', train=False, download=True, transform=image_transform)
    input_images = []
    input_labels = []
    for i in range(10):
        i_label = []
        for data, label in test_data:
            if label == i:
                input_images.append(data)
                for k in range(count // 10):
                    temp = random.randint(0, 9)
                    if temp == i:
                        temp += random.choice([-1, 1])

                    i_label.append(temp)
                input_labels.append(i_label)

                break


    net.eval()
    pred_images = []
    for i in range(10):
        p_imgs = []
        for j in range( len(input_labels[i]) ):
            x = input_images[i].unsqueeze(0), torch.tensor(input_labels[i][j]).unsqueeze(0)
            #print(input_labels[i])
            x_hat = net(x)
            p_imgs.append(x_hat.squeeze(0))

        pred_images.append( p_imgs )

    plot_vae_images(input_images, input_labels, pred_images, cols=count // 10, rows = 10, label_fn=label_fn)

