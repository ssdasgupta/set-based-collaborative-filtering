import torch
from torch import nn
import torch.optim as optim
import os
from tqdm import tqdm
import pprint
import wandb

from eval_metrics import run_eval
from loss import nll, nce, max_margin

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(
            self,
            model,
            n_users,
            n_items,
            n_user_attrs,
            n_item_attrs,
            train_loader,
            val_loader,
            eval_type = 'fixed_neg_eval',
            dataset = None,
            loss_type = 'bce',
            optimizer_type = 'adam',
            n_train_negs = 1,
            n_test_negs = 100,
            attribute_loss_const = 1.0,
            device = 'cpu',
            model_name = 'mf_bias',
            use_wandb = False,
            model_dir = 'models',
            save_model = False,
    ):
        self.model = model
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_attrs = n_user_attrs
        self.n_item_attrs = n_item_attrs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_type = eval_type
        self.device = device
        self.model_name = model_name
        self.n_train_negs = n_train_negs
        self.n_test_negs = n_test_negs
        self.attribute_loss_const = attribute_loss_const
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.use_wandb = use_wandb
        self.model_dir = model_dir
        self.save_model = save_model

        self.dataset = dataset
        if self.eval_type == 'full_eval':
            dataset.get_gt_dict()

        self.criterion = {
            'user_item': self.get_criteria(self.loss_type),
            'user_attr': self.get_criteria(self.loss_type),
            'item_attr': self.get_criteria(self.loss_type),
        }
        self.eval_metrices = None
        self.best_hr = 0.0
        self.best_ndcg = 0.0
        self.best_hr_attr = 0.0
        self.best_ndcg_attr = 0.0

    def gt_df_to_matrix(self, gt_df):
        columns = gt_df.columns
        n_rows = len(gt_df[columns[0]].unique())
        n_cols = self.n_items
        gt_matrix = torch.zeros(n_rows, n_cols, device=self.device)
        gt_matrix[gt_df[columns[0]], gt_df[columns[1]]] = 1
        return gt_matrix
    

    def get_criteria(self, loss_type):
        if loss_type == 'mse':
            return nn.MSELoss(reduction='sum')
        elif loss_type == 'bce':
            return nll
        elif loss_type == 'bce_logits':
            return nce
        elif loss_type == 'max-margin':
            return max_margin
        else:
            raise NotImplementedError

    def get_optimizer(self, optimizer_type, model, lr, wd):
        if optimizer_type == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == 'sparseadam':
            return optim.SparseAdam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError

    def random_negative_sample(self, ids, n, n_negs=1):
        # user_ids: (batch_size, )
        # item_ids: (batch_size, )
        batch_size = ids.size(0)
        neg_ids = torch.randint(low=0, high=n, size=(batch_size, n_negs), device=self.device)
        return neg_ids
    
    def run_single_batch(self, batch, n_negs=1):
        user_stream, item_stream, tuple_type = batch[0][:,0], batch[0][:,1], batch[0][:,2]
        user_stream = user_stream.to(self.device)
        item_stream = item_stream.to(self.device)
        tuple_type = tuple_type.to(self.device)
        # user-item training
        user = user_stream[tuple_type == 0]
        item = item_stream[tuple_type == 0]
        user_neg = self.random_negative_sample(user, self.n_users, n_negs)
        item_neg = self.random_negative_sample(item, self.n_items, n_negs)
        pos_outputs = self.model(user, item)
        neg_outputs_item = self.model(user, item_neg)
        neg_outputs_user = self.model(user_neg, item)
        neg_outputs = torch.cat([neg_outputs_item, neg_outputs_user], dim=-1)

        user_item_loss = self.criterion['user_item'](pos=pos_outputs,
                                                neg=neg_outputs)

        if self.n_user_attrs > 0 and 1 in tuple_type:
            # user-attr training
            user_ua = user_stream[tuple_type == 1]
            attr_ua = item_stream[tuple_type == 1]
            user_attr_neg = self.random_negative_sample(attr_ua, self.n_user_attrs, n_negs)
            pos_outputs_ua = self.model(user_ua, attr_ua + self.n_items)
            neg_outputs_ua = self.model(user_ua, user_attr_neg + self.n_items)

            user_attr_loss = self.criterion['user_attr'](pos=pos_outputs_ua,
                                                    neg=neg_outputs_ua)
        else:
            user_attr_loss = torch.tensor([0.0], device=self.device)

        if self.n_item_attrs > 0 and 2 in tuple_type:
            # item-attr training
            attr = user_stream[tuple_type == 2]
            item = item_stream[tuple_type == 2]
            item_neg = self.random_negative_sample(item, self.n_items)
            pos_outputs_ia = self.model(attr + self.n_users, item)
            neg_outputs_ia = self.model(attr + self.n_users, item_neg)
            # pred = torch.cat([pos_outputs, neg_outputs], dim=-1)
            # label = torch.cat([torch.ones_like(pos_outputs), torch.zeros_like(neg_outputs)], dim=-1)
            item_attr_loss = self.criterion['item_attr'](pos=pos_outputs_ia,
                                                    neg=neg_outputs_ia)
        else:
            item_attr_loss = torch.tensor([0.0], device=self.device)

        return user_item_loss, user_attr_loss, item_attr_loss

    def evaluate(self):
        eval_metrices = self.evaluate_loss()
        if self.eval_type == 'fixed_neg_eval':
            eval_metrices.update(self.evaluate_with_fixed_negatives())
        elif self.eval_type == 'full_eval':
            eval_metrices.update(self.evaluate_with_ranking())
        else:
            eval_metrices.update(self.evaluate_rank_with_true_negatives())

        
        return eval_metrices

    def handle_nan_inf(self, loss):
        if torch.isnan(loss).any():
            if self.use_wandb:
                wandb.log({'nan_loss': loss.item()})
            raise ValueError('nan loss')
        if torch.isinf(loss).any():
            print('inf loss')
            if self.use_wandb:
                wandb.log({'inf_loss': loss.item()})
            raise ValueError('inf loss')
        
    def save_model_state(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name + '_' + name + '.pth'))
    
    def get_best_score(self, eval_metrices):
        if eval_metrices['hr'] > self.best_hr:
            self.best_hr = eval_metrices['hr']
            if self.save_model:
                self.save_model_state('best_hr')
        if eval_metrices['ndcg'] > self.best_ndcg:
            self.best_ndcg = eval_metrices['ndcg']
            if self.save_model:
                self.save_model_state('best_ndcg')
        if eval_metrices['hr_attr'] > self.best_hr_attr:
            self.best_hr_attr = eval_metrices['hr_attr']
            if self.save_model:
                self.save_model_state('best_hr_attr')
        if eval_metrices['ndcg_attr'] > self.best_ndcg_attr:
            self.best_ndcg_attr = eval_metrices['ndcg_attr']
            if self.save_model:
                self.save_model_state('best_ndcg_attr')
        
        eval_metrices.update({
            'best_hr': self.best_hr,
            'best_ndcg': self.best_ndcg,
            'best_hr_attr': self.best_hr_attr,
            'best_ndcg_attr': self.best_ndcg_attr
        })
        return eval_metrices      

    def train(self, epochs, lr, wd):
        self.model.to(self.device)
        optimizer = self.get_optimizer(self.optimizer_type, self.model, lr, wd)
        train_losses = []
        train_user_item_losses = []
        train_item_attr_losses = []
        train_user_attr_losses = []
        self.eval_metrices = self.evaluate()
        pprint.pprint(self.eval_metrices)
        self.eval_metrices.update({'epoch': 0})
        if self.use_wandb:
            wandb.log(self.eval_metrices)
        for epoch in range(epochs):
            train_loss = 0.0
            train_user_item_loss = 0.0
            train_item_attr_loss = 0.0
            train_user_attr_loss = 0.0
            self.model.train()
            for batch in tqdm(self.train_loader):
                user_item_loss, user_attr_loss, item_attr_loss = self.run_single_batch(batch, 
                                                                                   n_negs=self.n_train_negs)
                optimizer.zero_grad()
                loss =  (1 - self.attribute_loss_const) * user_item_loss + self.attribute_loss_const * (user_attr_loss + item_attr_loss)
                self.handle_nan_inf(loss)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_user_item_loss += user_item_loss.item()
                train_item_attr_loss += item_attr_loss.item()
                train_user_attr_loss += user_attr_loss.item()

            train_loss = train_loss / len(self.train_loader)
            user_item_loss = train_user_item_loss / len(self.train_loader)
            item_attr_loss = train_item_attr_loss / len(self.train_loader)
            user_attr_loss = train_user_attr_loss / len(self.train_loader)
            train_losses.append(train_loss)
            train_user_item_losses.append(user_item_loss)
            train_item_attr_losses.append(item_attr_loss)
            train_user_attr_losses.append(user_attr_loss)

            if self.use_wandb:
                wandb.log({'loss': loss.item(),
                            'user_item_loss': user_item_loss,
                            'item_attr_loss': item_attr_loss,
                            'user_attr_loss': user_attr_loss},
                            commit=False)

            self.eval_metrices = self.evaluate()
            self.eval_metrices.update({'epoch': epoch + 1})
            self.eval_metrices = self.get_best_score(self.eval_metrices)
            pprint.pprint(self.eval_metrices)
            if self.use_wandb:
                wandb.log(self.eval_metrices)
            print('epoch: {}, train loss: {}, user_item_loss: {}, item_attr_loss: {}'.format(epoch + 1, train_loss, user_item_loss, item_attr_loss))

        return train_losses, train_user_attr_losses, train_item_attr_losses, train_user_item_losses

    def evaluate_loss(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        with torch.no_grad():
            test_user_item_loss, test_user_attr_loss, test_item_attr_loss = 0.0, 0.0, 0.0
            for batch in tqdm(self.val_loader):
                user_item_loss, user_attr_loss, item_attr_loss = self.run_single_batch(batch,
                                                                            n_negs=self.n_test_negs)
                loss = (1 - self.attribute_loss_const) * user_item_loss + self.attribute_loss_const * (user_attr_loss + item_attr_loss)
                test_user_item_loss += user_item_loss.item()
                test_user_attr_loss += user_attr_loss.item()
                test_item_attr_loss += item_attr_loss.item()
            test_user_item_loss = test_user_item_loss / len(self.val_loader)
            test_user_attr_loss = test_user_attr_loss / len(self.val_loader)
            test_item_attr_loss = test_item_attr_loss / len(self.val_loader)
    
            eval_metrices = {
                'loss': loss.item(),
                'valid_user_item_loss': test_user_item_loss,
                'valid_item_attr_loss': test_item_attr_loss
            }
        return eval_metrices

    def get_mask(self, user_list, item_list, gt_dict=None):
        """Get the mask for the user-item pairs in the batch."""
        mask = torch.ones((user_list.size(0), self.n_items), dtype=torch.bool, device=self.device)
        for i, user in enumerate(user_list):
            idx_mask = torch.tensor(gt_dict[user.item()], device=self.device)
            mask[i, idx_mask] = False
            mask[i, item_list[i]] = True
        return mask
    
    def get_true_negatives(self, bin_matrix, n_negatives=100):
        shuffled = torch.randperm(bin_matrix.shape[1], device=self.device)
        bin_matrix_shuffled = bin_matrix[:, shuffled]
        indx = bin_matrix_shuffled.sort(dim=1).indices[:,:n_negatives]
        return shuffled[indx]
    
    def get_hr_ndcg_101(self, scores):
        target_idx = torch.tensor(scores.shape[1] - 1)
        pred_order = torch.argsort(scores, dim=-1, descending=True)
        rank = torch.where(pred_order == target_idx)[1] + 1
        hr_101 = sum(rank <= 10) / len(rank)
        ndcg_101 = sum(1.0 / torch.log2(rank + 1)) / len(rank)
        return hr_101.item(), ndcg_101.item()
    
    def evaluate_rank_with_true_negatives(self):
        self.model.eval()
        with torch.no_grad():
            val_user_movie = self.dataset.val_user_movie
            user = val_user_movie['user_id'].values
            movie = val_user_movie['movie_id'].values
            user = torch.tensor(user, device=self.device)
            movie = torch.tensor(movie, device=self.device)

            all_negative_movies = self.get_true_negatives(self.gt_user_movie_matrix, n_negatives=100)
            assert self.gt_user_movie_matrix.gather(1, all_negative_movies).sum() == 0
            negative_movies = all_negative_movies[user]

            all_movies = torch.cat([negative_movies, movie.reshape(-1,1)], dim=-1)
            scores = self.model(user, all_movies)
            hr_101, ndcg_101 = self.get_hr_ndcg_101(scores)

            val_attribute_movie = self.dataset.val_attribute_movie
            attribute = val_attribute_movie['attribute_id'].values
            movie = val_attribute_movie['movie_id'].values
            attribute = torch.tensor(attribute, device=self.device)
            movie = torch.tensor(movie, device=self.device)

            all_negative_movies = self.get_true_negatives(self.gt_attribute_movie_matrix, n_negatives=100)
            assert self.gt_attribute_movie_matrix.gather(1, all_negative_movies).sum() == 0
            negative_movies = all_negative_movies[attribute]

            all_movies = torch.cat([negative_movies, movie.reshape(-1,1)], dim=-1)
            assert self.gt_attribute_movie_matrix[attribute, movie].all()

            scores = self.model(attribute + self.n_users, all_movies)
            hr_101_attr, ndcg_101_attr = self.get_hr_ndcg_101(scores)

            return {'hr': hr_101,
                    'ndcg': ndcg_101,
                    'hr_attr': hr_101_attr,
                    'ndcg_attr': ndcg_101_attr}

    def evaluate_with_fixed_negatives(self):
        self.model.eval()
        with torch.no_grad():
            val_user_movie_dict = self.dataset.val_neg_user_movie.to_dict('list')
            users = [int(x) for x in val_user_movie_dict.keys()]
            items = [x for x in val_user_movie_dict.values()]
            users = torch.tensor(users, device=self.device)
            items = torch.tensor(items, device=self.device)
            total_valid  = torch.arange(len(users))
            chunks = torch.chunk(total_valid, 10)
            scores = torch.empty((0, items.shape[-1]), device=self.device)
            for chunk in chunks:
                score_chunk = self.model(users[chunk], items[chunk])
                scores = torch.cat((scores, score_chunk), dim=0)
            hr_101, ndcg_101 = self.get_hr_ndcg_101(scores)

            val_attribute_movie_dict = self.dataset.val_neg_attribute_movie.to_dict('list')
            attributes = [int(x) for x in val_attribute_movie_dict.keys()]
            items = [x for x in val_attribute_movie_dict.values()]
            attributes = torch.tensor(attributes, device=self.device)
            items = torch.tensor(items, device=self.device)
            scores = self.model(attributes + self.n_users, items)
            hr_101_attr, ndcg_101_attr = self.get_hr_ndcg_101(scores)

            return {
                        'hr': hr_101,
                        'ndcg': ndcg_101,
                        'hr_attr': hr_101_attr,
                        'ndcg_attr': ndcg_101_attr
                    }

    def evaluate_with_ranking(self):
        self.model.eval()
        with torch.no_grad():
            user_movie_rank_list = []
            attribute_movie_rank_list = []
            for batch in tqdm(self.val_loader):
                user_stream, item_stream, tuple_type = batch[0][:,0], batch[0][:,1], batch[0][:,2]
                user_stream = user_stream.to(self.device)
                item_stream = item_stream.to(self.device)
                tuple_type = tuple_type.to(self.device)

                user = user_stream[tuple_type == 0]
                item = item_stream[tuple_type == 0]
                mask = self.get_mask(user, item, self.dataset.gt_user_movie_dict)
                all_item_score = self.model.predict_item(user.reshape(1, -1)).T
                all_item_score[~mask] = - torch.inf
                pred_order = torch.argsort(all_item_score, dim=-1, descending=True)
                rank = torch.where(pred_order == item.reshape(-1, 1))[1] + 1
                user_movie_rank_list.append(rank)

                if 2 in tuple_type:
                    attr = user_stream[tuple_type == 2]
                    item = item_stream[tuple_type == 2]
                    mask = self.get_mask(attr, item, self.dataset.gt_attribute_movie_dict)
                    all_item_score = self.model.predict_item((attr + self.n_users).reshape(1, -1)).T
                    all_item_score[~mask] = - torch.inf
                    pred_order = torch.argsort(all_item_score, dim=-1, descending=True)
                    rank = torch.where(pred_order == item.reshape(-1, 1))[1] + 1
                    attribute_movie_rank_list.append(rank)

            user_movie_rank_list = torch.cat(user_movie_rank_list)
            attribute_movie_rank_list = torch.cat(attribute_movie_rank_list)
            hr = sum(user_movie_rank_list <= 10) / len(user_movie_rank_list)
            ndcg = sum(1.0 / torch.log2(user_movie_rank_list + 1)) / len(user_movie_rank_list)
            hr_attr = sum(attribute_movie_rank_list <= 10) / len(attribute_movie_rank_list)
            ndcg_attr = sum(1.0 / torch.log2(attribute_movie_rank_list + 1)) / len(attribute_movie_rank_list)

            return {
                        'hr': hr.item(),
                        'ndcg': ndcg.item(),
                        'hr_attr': hr_attr.item(),
                        'ndcg_attr': ndcg_attr.item()
                    }
