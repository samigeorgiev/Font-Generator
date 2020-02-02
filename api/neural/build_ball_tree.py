# pylint: disable=no-member
# pylint: disable=not-callable

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle

from dataset.CONFIG import BATCH_SIZE


def contrastive_similarity(a, b):
    prod = a * b
    
    N = np.sum(prod * (prod < 0))
    P = np.sum(prod * (prod > 0))

    return - N*P

if __name__ == '__main__':
    NUM_BATCHES = 20

    embeddings = torch.load('neural/font_embeddings/font-embeddings-batch-1.pt')
    for i in tqdm(range(2, NUM_BATCHES+1), desc='Loading embeddings...'):
        embeddings = torch.cat((
            embeddings,
            torch.load(f'neural/font_embeddings/font-embeddings-batch-{i}.pt')
        ), dim=0)

    embeddings = embeddings.view(BATCH_SIZE * NUM_BATCHES, -1).detach().cpu().numpy()

    knn_search_ball_tree = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', n_jobs=-1)

    # Build the Ball Tree
    knn_search_ball_tree.fit(embeddings)

    pickle.dump(knn_search_ball_tree, open('./neural/checkpoints/knn_search_ball_tree.b', 'wb'), protocol=4)
