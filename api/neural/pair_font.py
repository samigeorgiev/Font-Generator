# pylint: disable=no-member
# pylint: disable=not-callable

import pickle

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

knn_search_ball_tree = pickle.load(open('./neural/checkpoints/knn_search_ball_tree.b', 'rb'))
# knn_search_ball_tree.metric = contrastive_similarity

def pair_font(base_embedding, contrast=0):
    # KNN Search
    indices = knn_search_ball_tree.kneighbors([base_embedding], n_neighbors=201, return_distance=False)[0]

    similarity = (contrast * -1 + 1) / 2
    return indices[int(similarity*200)]


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
