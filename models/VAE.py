import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=28, num_embed=10):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        #label embedding
        self.label_embed  = nn.Embedding(num_embed, embedding_dim=enc_out_dim)

        self.metric = dict(
            train_steps=0,
            step_train_loss=[],
            epoch_train_loss=[],
            val_steps=0,
            step_val_loss=[],
            epoch_val_loss=[],
            sample=[]
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        x, label = x
        x_encoded = self.encoder(x)
        x_encoded_embedded = x_encoded + self.label_embed(label)
        mu, log_var = self.fc_mu(x_encoded_embedded), self.fc_var(x_encoded_embedded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, label = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        x_encoded_embedded = x_encoded + self.label_embed(label)

        mu, log_var = self.fc_mu(x_encoded_embedded), self.fc_var(x_encoded_embedded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        error_dict = {
            'elbo': elbo,
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        }

        self.log_dict(error_dict)

        self.metric['step_train_loss'].append(elbo)
        self.metric['train_steps'] += 1

        return elbo


    def validation_step(self, val_batch, batch_idx):
        x, label = val_batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        x_encoded_embedded = x_encoded + self.label_embed(label)

        mu, log_var = self.fc_mu(x_encoded_embedded), self.fc_var(x_encoded_embedded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        error_dict = {
            'elbo': elbo,
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        }
        self.log_dict(error_dict)

        self.metric['step_val_loss'].append(elbo)
        self.metric['val_steps'] += 1

    def on_validation_epoch_end(self):
        if self.metric['train_steps'] > 0:
            print('Epoch ', self.current_epoch)

            epoch_loss = sum(self.metric['step_train_loss']) / len(self.metric['step_train_loss'])
            self.metric['epoch_train_loss'].append(epoch_loss.item())
            self.metric['step_train_loss'] = []

            print('Train Loss: ', epoch_loss.item())

            epoch_loss = sum(self.metric['step_val_loss']) / len(self.metric['step_val_loss'])
            self.metric['epoch_val_loss'].append(epoch_loss.item())
            self.metric['step_val_loss'] = []
            print('Val Loss: ', epoch_loss.item())