import torch

from model.box.box_wrapper import BoxTensor
from model.box.modules import BoxEmbedding
from model.vector_model import MatrixFactorization

global use_cuda
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else -1


class BoxRec(MatrixFactorization):
    def __init__(self,
                n_users,
                n_items,
                embedding_dim=20,
                box_type="BoxTensor",
                volume_temp=0.1,
                intersection_temp=0.1,
    ):
        super().__init__(n_users, n_items, embedding_dim)

        # Create embeddings
        self.user_embeddings = BoxEmbedding(
            self.n_users, self.embedding_dim, box_type=box_type
        )
        self.item_embeddings = BoxEmbedding(
            self.n_items, self.embedding_dim, box_type=box_type
        )
        self.box_type = box_type
        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp

    def forward(self, user, item):
        if len(user.shape) > len(item.shape):
            item = item.reshape(-1, 1)
        elif len(user.shape) < len(item.shape):
            user = user.reshape(-1, 1)
        user_boxes = self.user_embeddings(user)  # Batch_size * 2 * dim
        item_boxes = self.item_embeddings(item)  # Batch_size * ns+1 * 2 * dim

        if self.intersection_temp == 0.0:
            score = user_boxes.intersection_log_soft_volume(
                item_boxes, 
                volume_temp=self.volume_temp,
                bayesian=False
            )
        else:
            score = user_boxes.gumbel_intersection_log_volume(
                item_boxes,
                volume_temp=self.volume_temp,
                intersection_temp=self.intersection_temp,
            )

        return score

    # def word_similarity(self, w1, w2):
    #     with torch.no_grad():
    #         word1 = self.embeddings_word(w1)
    #         word2 = self.embeddings_word(w2)
    #         if self.intersection_temp == 0.0:
    #             score = word1.intersection_log_soft_volume(word2, temp=self.volume_temp)
    #         else:
    #             score = word1.gumbel_intersection_log_volume(
    #                 word2,
    #                 volume_temp=self.volume_temp,
    #                 intersection_temp=self.intersection_temp,
    #             )
    #         return score

    # def conditional_similarity(self, w1, w2):
    #     with torch.no_grad():
    #         word1 = self.embeddings_word(w1)
    #         word2 = self.embeddings_word(w2)
    #         if self.intersection_temp == 0.0:
    #             score = word1.intersection_log_soft_volume(word2, temp=self.volume_temp)
    #         else:
    #             score = word1.gumbel_intersection_log_volume(
    #                 word2,
    #                 volume_temp=self.volume_temp,
    #                 intersection_temp=self.intersection_temp,
    #             )
    #         #  Word1 Word2  queen   royalty 5.93
    #         # Word2 is more geenral P(royalty | queen) = 1
    #         # Thus we need p(w2 | w1)
    #         score -= word1._log_soft_volume_adjusted(
    #             word1.z,
    #             word1.Z,
    #             temp=self.volume_temp,
    #             gumbel_beta=self.intersection_temp,
    #         )
    #         return score
