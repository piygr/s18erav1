import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch_lr_finder import LRFinder
import torchvision.transforms.functional as TF


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ContractingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if not downsample:
            self.downsample = None
        elif downsample == 'strided_conv':
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
        if not self.downsample:
            return x, skip

        x = self.downsample(x)

        print('x.shape: ', x.shape, ' skip.shape: ', skip.shape)
        return x, skip


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=None):
        super(ExpandingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if not upsample:
            self.upsample = None
            self.upsample2 = None
        elif upsample == 'transpose_conv':
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
            self.upsample2 = None
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.upsample2 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, padding=1)

    def forward(self, x, skip):

        if skip:
            # concatenate the skip connection
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            print('pre x.shape: ', x.shape, ' skip.shape: ', skip.shape)
            x = torch.cat((skip, x), dim=1)

            print('x.shape: ', x.shape)

            print('skip.shape: ', skip.shape, ' x.shape: ', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if self.upsample:
            x = self.upsample(x)

        if self.upsample2:
            x = self.upsample2(x)

        return x


class UNet(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 max_lr=None,
                 loss_fn=nn.BCELoss,
                 upsample='transpose_conv',
                 downsample='strided_conv'):
        super(UNet, self).__init__()

        self.contract1 = ContractingBlock(in_channels, 64, downsample=downsample)
        self.contract2 = ContractingBlock(64, 128, downsample=downsample)
        self.contract3 = ContractingBlock(128, 256, downsample=downsample)

        self.bottleneck = ContractingBlock(256, 512)
        self.bottleneck2 = ExpandingBlock(512, 256)

        self.expand1 = ExpandingBlock(512, 256, upsample=upsample)
        self.expand2 = ExpandingBlock(256, 128, upsample=upsample)
        self.expand3 = ExpandingBlock(128, 64, upsample=upsample)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.loss_fn = loss_fn

        self.max_lr = max_lr

        self.metric = dict(
            train_steps=0,
            step_train_loss=[],
            epoch_train_loss=[],
            val_steps=0,
            step_val_loss=[],
            epoch_val_loss=[]
        )


    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)    #skip1 : 64
        x, skip2 = self.contract2(x)    #skip2 : 128
        x, skip3 = self.contract3(x)    #skip3 : 256

        _, x = self.bottleneck(x)       #x :    256
        x = self.bottleneck2(x, None)

        # Expanding path
        #x = self.expand0(x, skip4)
        x = self.expand1(x, skip3)      #x: 512,
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)

        logits = self.final_conv(x)

        return logits


    def training_step(self, train_batch, batch_idx):
        data, target = train_batch
        pred = self.forward(data)

        loss = self.loss_fn(pred, target)
        self.metric['step_train_loss'].append(loss)
        self.metric['train_steps'] += 1

        self.log_dict(dict(train_loss=loss))
        return loss


    def validation_step(self, val_batch, batch_idx):
        data, target = val_batch
        pred = self.forward(data)
        loss = self.loss_fn(pred, target)
        self.metric['step_val_loss'].append(loss)
        self.metric['val_steps'] += 1

        self.log_dict(dict(val_loss=loss))


    def on_validation_epoch_end(self):
        if self.metric['train_steps'] > 0:
            print('Epoch ', self.current_epoch+1)

            epoch_loss = sum(self.metric['step_train_loss']) / len(self.metric['step_train_loss'])
            self.metric['epoch_train_loss'].append(epoch_loss)
            self.metric['step_train_loss'] = []

            print('Train Loss: ', epoch_loss)

            epoch_loss = sum(self.metric['step_val_loss']) / len(self.metric['step_val_loss'])
            self.metric['epoch_val_loss'].append(epoch_loss)
            self.metric['step_val_loss'] = []
            print('Val Loss: ', epoch_loss)



    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader


    def find_lr(self, optimizer):

        lr_finder = LRFinder(self, optimizer, self.loss_fn)
        lr_finder.range_test(self.train_dataloader(), end_lr=100, num_iter=100)
        _, best_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()
        self.max_lr = best_lr


    def configure_optimizers(self):
        if not self.max_lr:
            optimizer = torch.optim.Adam(self.parameters(), lr=10e-6, weight_decay=10e-2)
            self.find_lr(optimizer)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr, weight_decay=10e-2)
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