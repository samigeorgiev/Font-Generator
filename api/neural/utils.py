# pylint: disable=no-member

import torch

from models import AutoEncoder

PATH_TO_EMBEDDER = 'neural/checkpoints/ae-512-224x224-loss-0.024.pth'


def init_model(path_to_checkpoint=PATH_TO_EMBEDDER):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = AutoEncoder(
        latent_dim=512,
        in_channels=1,
        num_hiddens=256,
        num_res_hiddens=64,
        num_res_layers=4,
        out_channels=1,
    ).to(device)

    net.load_state_dict(
        torch.load(open(path_to_checkpoint, 'rb'), map_location=device)
    )
    
    print()
    print('='*30, end='\n\n')
    print(net.eval())
    print(end='\n\n')
    print('='*30, end='\n\n')

    return device, net
