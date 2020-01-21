# pylint: disable=no-member

import csv
import random

import torch
from tqdm import tqdm

from neural.dataset.CONFIG import BATCH_SIZE
from neural.pair_font import pair_font
from neural.utils import init_model


NUM_BATCHES = 10
device, net = init_model()

embeddings = torch.load('./neural/font_embeddings/font-embeddings-batch-1.pt')
for i in tqdm(range(2, NUM_BATCHES+1), desc='Loading embeddings...'):
    embeddings = torch.cat((
        embeddings,
        torch.load(f'./neural/font_embeddings/font-embeddings-batch-{i}.pt')
    ), dim=0)

with open('./neural/data/metadata.tsv') as fd:
    metadata = list(csv.reader(fd, delimiter="\t", quotechar='"'))
embeddings = embeddings.view(BATCH_SIZE * NUM_BATCHES, -1).detach().cpu().numpy()


def get_pair_by_contrast(name, contrast):
    # print('Pairing...')
    # base_embedding = random.choice(embeddings)
    # idx = pair_font(base_embedding, embeddings)

    # print(metadata[idx][0])

    print(random.choice(metadata)[0].split(' ')[0])
    return random.choice(metadata)[0].split(' ')[0]
