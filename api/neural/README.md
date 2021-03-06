# Neural Font Embedder and Recommender


## Neural Font Embedder

A Convolutional [ResNet](https://arxiv.org/abs/1512.03385) AutoEncoder is used to obtain Font Embeddings. <br>
The ResNet architecture is as follows:

![resnet-block](./assets/resnet-block.png)

### Contrastive Similarity and [Ball Tree](https://en.wikipedia.org/wiki/Ball_tree) is used to traverse the vector space:

![contrastive-similarity](./assets/cs-metric.png)

## Neural Collaborative Filtering Recommender System

Architecture:

![neural-cf](./assets/neural-cf.png)

Reference: [Recommender Systems using Deep Learning in PyTorch from scratch](https://towardsdatascience.com/recommender-systems-using-deep-learning-in-pytorch-from-scratch-f661b8f391d7)
