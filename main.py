import torch.nn as nn
from models.UNet import UNet
from torchsummary import summary
import pytorch_lightning as pl

def init(
        train_dataloader,
        val_dataloader,
        in_channels=3,
        out_channels=1,
        show_summary=False,
        max_lr=None,
        loss_fn=nn.BCELoss,
        upsample='transpose_conv',
        downsample='strided_conv',
        num_epochs=1
):

    model = UNet(in_channels=in_channels,
                 out_channels=out_channels,
                 max_lr=max_lr,
                 loss_fn=loss_fn,
                 upsample=upsample,
                 downsample=downsample)


    if show_summary:
        summary(model, input_size=(3, 224, 224))


    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=num_epochs
    )

    trainer.fit(model, train_dataloader, val_dataloader)