import torch.nn as nn
from models.UNet import UNet
from models.VAE import VAE
from torchsummary import summary
import pytorch_lightning as pl
from config import unet_config, vae_config

from loss import bce_loss, dice_loss
from utils import device

def init(
        train_dataloader,
        val_dataloader,
        net='UNet',
        in_channels=3,
        out_channels=1,
        show_summary=False,
        max_lr=None,
        loss_fn=bce_loss,
        upsample='transpose_conv',
        downsample='maxpool',
        accelerator=None
):

    if net == 'UNet':
        cfg = unet_config

        model = UNet(in_channels=in_channels,
                     out_channels=out_channels,
                     max_lr=max_lr,
                     loss_fn=loss_fn,
                     upsample=upsample,
                     downsample=downsample)

    else:
        cfg = vae_config
        
        model = VAE(
            enc_out_dim=cfg['enc_out_dim'],
            latent_dim=cfg['latent_dim'],
            num_embed=cfg['num_classes']
        )

    if show_summary:
        summary(model.to(device), input_size=(in_channels, cfg['image_size'], cfg['image_size']))

    trainer_args = dict(
        precision='16-mixed',
        max_epochs=cfg['num_epochs']
    )

    if accelerator:
        trainer_args['accelerator'] = accelerator

    trainer = pl.Trainer(
        **trainer_args
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return model