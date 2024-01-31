import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MatrixFactorization(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 n_users_attr,
                 n_items_attr,
                 embedding_dim=20):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dim, sparse=True)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim, sparse=True)
        self.user_attribute_embeddings = nn.Embedding(n_users_attr, embedding_dim, sparse=True)
        self.item_attribute_embeddings = nn.Embedding(n_items_attr, embedding_dim, sparse=True)

    def forward(self, user, item, user_attributes, item_attributes):
        user_item_interaction = (self.user_embeddings(user) * self.item_embeddings(item)).sum(1)
        user_attribute_user_interaction = (self.user_attribute_embeddings(user_attributes) * self.user_embeddings(user)).sum(1)
        item_attribute_item_interaction = (self.item_attribute_embeddings(item_attributes) * self.item_embeddings(item)).sum(1)
        return user_item_interaction + user_attribute_user_interaction + item_attribute_item_interaction

    def predict(self, user, item, user_attributes, item_attributes):
        return self.forward(user, item, user_attributes, item_attributes)


class MatrixFactorizationWithBias(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 n_users_attr,
                 n_items_attr,
                 embedding_dim=20):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dim, sparse=True)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim, sparse=True)
        self.user_attribute_embeddings = nn.Embedding(n_users_attr, embedding_dim, sparse=True)
        self.item_attribute_embeddings = nn.Embedding(n_items_attr, embedding_dim, sparse=True)
        self.user_biases = nn.Embedding(n_users, 1, sparse=True)
        self.item_biases = nn.Embedding(n_items, 1, sparse=True)
        self.user_attribute_biases = nn.Embedding(n_users_attr, 1, sparse=True)
        self.item_attribute_biases = nn.Embedding(n_items_attr, 1, sparse=True)

    def forward(self, user, item, user_attributes, item_attributes):
        user_item_interaction = (self.user_embeddings(user) * self.item_embeddings(item)).sum(1)
        user_attribute_user_interaction = (self.user_attribute_embeddings(user_attributes) * self.user_embeddings(user)).sum(1)
        item_attribute_item_interaction = (self.item_attribute_embeddings(item_attributes) * self.item_embeddings(item)).sum(1)
        user_bias = self.user_biases(user).squeeze()
        item_bias = self.item_biases(item).squeeze()
        user_attribute_bias = self.user_attribute_biases(user_attributes).squeeze()
        item_attribute_bias = self.item_attribute_biases(item_attributes).squeeze()
        return user_item_interaction + user_attribute_user_interaction + item_attribute_item_interaction + user_bias + item_bias + user_attribute_bias + item_attribute_bias

    def predict(self, user, item, user_attributes, item_attributes):
        return self.forward(user, item, user_attributes, item_attributes)