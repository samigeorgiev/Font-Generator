import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'NeuralCF',
]


class NeuralCF(nn.Module):
    ''' Neural Collaborative Filtering Recommender System
    '''

    def __init__(self, num_users=100, user_embedding_dim=256, item_embedding_dim=512, num_cf_layers=4):
        super(NeuralCF, self).__init__()

        ### Hyperparameters ###

        self.num_users = num_users
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.num_cf_layers = num_cf_layers

        ### NN Parameters ###

        self.user_embedding_layer = nn.Embedding(num_users, user_embedding_dim)
        self.item_dense_layer = nn.Sequential(
            nn.Linear(item_embedding_dim, user_embedding_dim),
            nn.Tanh(),
        )

        self.cf_layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(user_embedding_dim / (2**i), user_embedding_dim / (2**(i+1))),
                nn.ReLU(True)
            ) for i in range(num_cf_layers)
        )

        self.predictor = nn.Sequential(
            nn.Linear(user_embedding_dim / (2**num_cf_layers), 1),
            nn.Sigmoid(),
        )

    def forward(self, user, item):
        latent_sum = self.user_embedding_layer(user) \
                    + self.item_dense_layer(item)

        features = self.cf_layers(latent_sum)
        return self.predictor(features)

    def loss_function(self, input, target):
        return F.binary_cross_entropy(input, target)
