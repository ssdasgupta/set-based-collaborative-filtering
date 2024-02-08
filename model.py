import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MatrixFactorization(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 embedding_dim=20):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embeddings = nn.Embedding(n_users, embedding_dim, sparse=True)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim, sparse=True)

    def forward(self, user, item):
        user_item_interaction = (self.user_embeddings(user) * self.item_embeddings(item)).sum(1)
        return user_item_interaction

    def predict_item(self, user):
        return self.forward(user, torch.arange)


class MatrixFactorizationWithBias(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 embedding_dim=20):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)

    def forward(self, user, item):
        if len(user.shape) > len(item.shape):
            item = item.reshape(-1, 1)
        elif len(user.shape) < len(item.shape):
            user = user.reshape(-1, 1)
        user_item_interaction = (self.user_embeddings(user) * self.item_embeddings(item)).sum(-1)
        user_bias = self.user_biases(user)
        item_bias = self.item_biases(item)
        return user_item_interaction + user_bias.squeeze(-1) + item_bias.squeeze(-1)

    def predict_item(self, user):
        return self.forward(user, torch.arange(self.n_items).to(device))