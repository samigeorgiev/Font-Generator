# pylint: disable=no-member
# pylint: disable=not-callable

import torch
from sklearn.neighbors import NearestNeighbors

from utils import init_model


def contrastive_similarity(a, b):
    prod = a * b
    
    N = torch.sum(torch.clamp_max(prod, 0))
    P = torch.sum(torch.relu(prod))

    return - N*P

def pair_font(base, contrast, n_neighbors=5,
                device=None, net=None, embeddings=None, names=None):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree',
        metric=contrastive_similarity, n_jobs=-1)

    base_idx = (base == names).nonzero()
    base_embedding = embeddings[base_idx]


if __name__ == '__main__':
    device, net = init_model()
    
