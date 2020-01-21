import torch

from dataset import FontsLoader
from utils import init_model


if __name__ == '__main__':
    _, net = init_model()
    loaded_set = FontsLoader.get_set_loader()

    print()
    for i, batch in enumerate(loaded_set, 1):
        embeddings = net.post_encoder( net.encoder(batch['image']) )

        torch.save(embeddings, f'./font_embeddings/font-embeddings-batch-{i}.pt')
        torch.save(batch['name'], f'./font_embeddings/font-names-batch-{i}.pt')

        print(f'Saved font-embeddings-batch-{i}')
    print()
