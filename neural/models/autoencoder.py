import torch.nn as nn
import torch.nn.functional as F

from .convolutional import ResDecoder, ResEncoder

__all__ = [
    'AutoEncoder',
]


class AutoEncoder(nn.Module):
    '''Convolutional AutoEncoder
    '''

    def __init__(self, latent_dim=512,
            in_channels=1, num_hiddens=256, num_res_hiddens=64,num_res_layers=4, out_channels=1):

        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = ResEncoder(in_channels, num_hiddens, num_res_hiddens, num_res_layers)

        # Post-Encoder Convolution
        self.post_encoder = nn.Conv2d(num_hiddens, latent_dim, kernel_size=1, stride=1)

        # Decoder
        self.decoder = ResDecoder(latent_dim, num_hiddens, num_res_hiddens, num_res_layers, out_channels)

    def forward(self, x):
        latent_vector = self.post_encoder( self.encoder(x) )
        reconstruction = self.decoder(latent_vector)

        return latent_vector, reconstruction

    def loss_function(self, x, recon_x):
        return F.binary_cross_entropy(recon_x, x)
