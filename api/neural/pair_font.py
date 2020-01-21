# pylint: disable=no-member
# pylint: disable=not-callable

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from neural.dataset import FontsLoader
from neural.utils import init_model


def contrastive_similarity(a, b):
    prod = a * b
    
    N = np.sum(prod * (prod < 0))
    P = np.sum(prod * (prod > 0))

    return - N*P

def pair_font(base_embedding, embeddings, contrast=0, n_neighbors=5):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree',
        metric=contrastive_similarity, n_jobs=-1)

    # Build the Ball Tree
    nbrs.fit(embeddings)

    # KNN Search
    indices = nbrs.kneighbors([base_embedding], n_neighbors=10, return_distance=False)[0]

    return indices[(contrast*10)//1]


if __name__ == '__main__':
    device, net = init_model()
    dataset = FontsLoader.get_set()

    base_embedding = net.post_encoder(
        net.encoder(
            dataset.get_image_by_name('001842-font-773-regular-Fruktur.png')['image'].unsqueeze(0)
        )
    )[0]

    base_embedding = torch.flatten(base_embedding).detach().cpu().numpy()
    embeddings = torch.load('./font_embeddings/font-embeddings-batch-1.pt').view(32, -1).detach().cpu().numpy()
    names = torch.load('./font_embeddings/font-names-batch-1.pt')

    idx = pair_font(base_embedding, embeddings)
    print(names[idx])
