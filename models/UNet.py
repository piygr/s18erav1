import torch.nn as nn
import pytorch_lightning as pl
import torch

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample='strided_conv'):
        super(ContractingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if downsample == 'strided_conv':
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
        else:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        skip = x  # store the output for the skip connection
        x = self.downsample(x)

        return x, skip


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample='transpose_conv'):
        super(ExpandingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if upsample == 'transpose_conv':
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
            self.upsample2 = None
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.upsample2 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1)

    def forward(self, x, skip):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.upsample(x)
        if self.upsample2:
            x = self.upsample2(x)

        # concatenate the skip connection
        x = torch.cat((x, skip), dim=1)

        return x


class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, loss_fn=nn.BCELoss):
        super(UNet, self).__init__()

        self.contract1 = ContractingBlock(in_channels, 64)
        self.contract2 = ContractingBlock(64, 128)
        self.contract3 = ContractingBlock(128, 256)
        self.contract4 = ContractingBlock(256, 512)

        self.expand1 = ExpandingBlock(512, 256)
        self.expand2 = ExpandingBlock(256, 128)
        self.expand3 = ExpandingBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        _, x = self.contract4(x)

        # Expanding path
        x = self.expand1(x, skip3)
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)

        logits = self.final_conv(x)

        return logits


    def training_step(self, train_batch, batch_idx):
        data, target = train_batch
        pred = self.forward(data)


    def validation_step(self, val_batch, batch_idx):
        data, target = val_batch
        pred = self.forward(data)


    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg['lr'])
        '''scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=cfg.LEARNING_RATE,
                                                        epochs=self.trainer.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()),
                                                        pct_start=8 / self.trainer.max_epochs,
                                                        div_factor=100,
                                                        final_div_factor=100,
                                                        three_phase=False,
                                                        verbose=False
                                                        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            },
        }'''

        return {
            "optimizer": optimizer
        }