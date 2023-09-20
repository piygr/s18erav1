import torch.nn as nn
from models.UNet import UNet
from torchsummary import summary
import pytorch_lightning as pl
from config import unet_config

from loss import bce_loss, dice_loss

def init(
        train_dataloader,
        val_dataloader,
        in_channels=3,
        out_channels=1,
        show_summary=False,
        max_lr=None,
        loss_fn=bce_loss,
        upsample='transpose_conv',
        downsample='maxpool',
        accelerator=None
):

    model = UNet(in_channels=in_channels,
                 out_channels=out_channels,
                 max_lr=max_lr,
                 loss_fn=loss_fn,
                 upsample=upsample,
                 downsample=downsample)


    if show_summary:
        summary(model, input_size=(in_channels, unet_config['image_size'], unet_config['image_size']))


    trainer_args = dict(
        precision='16-mixed',
        max_epochs=unet_config['num_epochs']
    )

    if accelerator:
        trainer_args['accelerator'] = accelerator


    trainer = pl.Trainer(
        **trainer_args
    )

    trainer.fit(model, train_dataloader, val_dataloader)