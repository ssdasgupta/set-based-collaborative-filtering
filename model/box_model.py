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

    def box_predict(self, combination_box):
        all_items = torch.arange(self.n_items).to(device)
        combination_box = combination_box.to(device)
        item_boxes = self.item_embeddings(all_items)
        if self.intersection_temp == 0.0:
            scores = combination_box.intersection_log_soft_volume(
                item_boxes, volume_temp=self.volume_temp
            )
        else:
            scores = combination_box.gumbel_intersection_log_volume(
                item_boxes,
                volume_temp=self.volume_temp,
                intersection_temp=self.intersection_temp,
            )
        return scores

    def similarity_score(self, user_boxes, item_boxes):
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



    def forward(self, user, item):
        if len(user.shape) > len(item.shape):
            item = item.reshape(-1, 1)
        elif len(user.shape) < len(item.shape):
            user = user.reshape(-1, 1)
        user_boxes = self.user_embeddings(user)  # Batch_size * 2 * dim
        item_boxes = self.item_embeddings(item)  # Batch_size * ns+1 * 2 * dim

        return self.similarity_score(user_boxes, item_boxes)

class BoxRecConditional(BoxRec):
    def __init__(self, n_users, n_items, embedding_dim=20, box_type="BoxTensor", volume_temp=0.1, intersection_temp=0.1):
        super().__init__(n_users, n_items, embedding_dim, box_type, volume_temp, intersection_temp)

    def similarity_score(self, user_boxes, item_boxes):
        intersection_score = super().similarity_score(user_boxes, item_boxes)
        log_volume_items = item_boxes.log_soft_volume_adjusted(volume_temp=self.volume_temp,
                                                      intersection_temp=self.intersection_temp)
        conditional_prob = intersection_score - log_volume_items
        assert (conditional_prob <= 0).all(), "Log probability can not be positive"
        return conditional_prob
