# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolutional import ResDecoder, ResEncoder

__all__ = [
    'VAE',
]


class VAE(nn.Module):
    ''' beta-Variational AutoEncoder

    Implementation of the ideas presented here:

    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        - https://arxiv.org/abs/1312.6114

    β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR, 2017
        - https://openreview.net/forum?id=Sy2fzU9gl

    Parameters:
    -----------
    latent_dim : int, dimension of the hidden representation vector
    beta : int, KLD weighting coefficent (see β-VAE)
    
    '''

    def __init__(self, latent_dim=64, beta=5,
                    in_channels=1, num_hiddens=128, num_res_hiddens=32, num_res_layers=2, out_channels=1):

        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = ResEncoder(in_channels, num_hiddens, num_res_hiddens, num_res_layers)

        # Pre-Variational Convolution
        self.pre_variational_conv = nn.Conv2d(num_hiddens, latent_dim*4, kernel_size=1, stride=1)

        # Compute Mean and LogVar
        self.fc1 = nn.Linear(latent_dim*4, latent_dim*2)
        self.fc21 = nn.Linear(latent_dim*2, latent_dim)
        self.fc22 = nn.Linear(latent_dim*2, latent_dim)

        # Setup for Decoder
        self.fc3 = nn.Linear(latent_dim, latent_dim*2)
        self.fc4 = nn.Linear(latent_dim*2, latent_dim*4)

        # Decoder
        self.decoder = ResDecoder(latent_dim*4, num_hiddens, num_res_hiddens, num_res_layers, out_channels)

    def encode(self, x):
        conv_out = self.pre_variational_conv( self.encoder(x) )
        h1 = self.fc1(conv_out.view(-1, self.latent_dim*4))

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)
        deconv_input = h4.view(-1, self.latent_dim*4, 1, 1)

        return self.decoder(deconv_input)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return mu, logvar, z, reconstruction

    def loss_function(self, x, recon_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        return BCE + self.beta * KLD
