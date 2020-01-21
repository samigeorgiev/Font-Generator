# pylint: disable=no-member

import csv
import random

import torch

from neural.dataset import FontsLoader
from neural.pair_font import pair_font
from neural.utils import init_model

dataset = FontsLoader.get_set()
device, net = init_model()

embeddings = torch.load('./neural/font_embeddings/font-embeddings-batch-10.pt').view(32, -1).detach().cpu().numpy()
names = torch.load('./neural/font_embeddings/font-names-batch-10.pt')

with open('./neural/data/metadata.tsv') as fd:
    metadata = list(csv.reader(fd, delimiter="\t", quotechar='"'))

def get_pair_by_contrast(name, contrast):
    # base_embedding = net.post_encoder(
    #     net.encoder(
    #         dataset.get_image_by_name('001842-font-773-regular-Fruktur.png')['image'].unsqueeze(0)
    #     )
    # )[0]

    # base_embedding = torch.flatten(base_embedding).detach().cpu().numpy()
    # idx = pair_font(base_embedding, embeddings)
    # print(names[idx])
    # new_font = names[idx].split('.')[0].split('-')[-1]
    # print(new_font)

    print(random.choice(metadata)[0].split(' ')[0])

    return random.choice(metadata)[0].split(' ')[0]
