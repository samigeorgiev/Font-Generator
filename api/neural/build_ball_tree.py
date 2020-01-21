# pylint: disable=no-member
# pylint: disable=not-callable

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle

from neural.dataset.CONFIG import BATCH_SIZE


def contrastive_similarity(a, b):
    prod = a * b
    
    N = np.sum(prod * (prod < 0))
    P = np.sum(prod * (prod > 0))

    return - N*P

if __name__ == '__main__':
    NUM_BATCHES = 5

    embeddings = torch.load('./font_embeddings/font-embeddings-batch-1.pt')
    for i in tqdm(range(2, NUM_BATCHES+1), desc='Loading embeddings...'):
        embeddings = torch.cat((
            embeddings,
            torch.load(f'./font_embeddings/font-embeddings-batch-{i}.pt')
        ), dim=0)

    embeddings = embeddings.view(BATCH_SIZE * NUM_BATCHES, -1).detach().cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree',
        metric=contrastive_similarity, n_jobs=-1)

    # Build the Ball Tree
    nbrs.fit(embeddings)

    pickle.dump(nbrs, open('knn_search.b', 'wb'))
