# pylint: disable=no-member
# pylint: disable=not-callable

import torch
from sklearn.neighbors import NearestNeighbors

from utils import init_model
from dataset import FontsLoader


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
    loaded_set = FontsLoader.get_set_loader()

    print()
    for i, batch in enumerate(loaded_set, 1):
        embeddings = net.post_encoder( net.encoder(batch['image']) )

        torch.save(embeddings, f'./font_embeddings/font-embeddings-batch-{i}.pt')
        torch.save(batch['name'], f'./font_embeddings/font-names-batch-{i}.pt')

        print(f'Saved font-embeddings-batch-{i}')
    print()
