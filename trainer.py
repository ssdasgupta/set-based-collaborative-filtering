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
            val_neg_df,
            gt_dict,
            loss_type,
            optimizer_type,
            n_negs,
            device,
            model_dir,
            model_name,
            wandb
    ):
        self.model = model
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_attrs = n_user_attrs
        self.n_item_attrs = n_item_attrs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_neg_df = val_neg_df
        self.gt_dict = gt_dict
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.n_negs = n_negs
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.wandb = wandb
    

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


    def train(self, epochs, lr, wd):
        self.model.to(self.device)
        optimizer = self.get_optimizer(self.optimizer_type, self.model, lr, wd)  # do we need to use sparseadam? or adam is fine?
        criterion = {
            'user_item': self.get_criteria(self.loss_type),
            'user_attr': self.get_criteria(self.loss_type),
            'item_attr': self.get_criteria(self.loss_type),
        }
        train_losses = []
        test_losses = []
        self.eval_metrices = self.evaluate()
        pprint.pprint(self.eval_metrices)
        for epoch in range(epochs):
            train_loss = 0.0
            self.model.train()
            for batch in tqdm(self.train_loader):
                user_stream, item_stream, tuple_type = batch[0][:,0], batch[0][:,1], batch[0][:,2]
                user_stream = user_stream.to(self.device)
                item_stream = item_stream.to(self.device)
                tuple_type = tuple_type.to(self.device)
                # user-item training
                user = user_stream[tuple_type == 0]
                item = item_stream[tuple_type == 0]
                user_neg = self.random_negative_sample(user, self.n_users, self.n_negs)
                item_neg = self.random_negative_sample(item, self.n_items, self.n_negs)
                pos_outputs = self.model(user, item)
                neg_outputs_item = self.model(user, item_neg)
                neg_outputs_user = self.model(user_neg, item)
                neg_outputs = torch.cat([neg_outputs_item, neg_outputs_user], dim=-1)

                # pred = torch.cat([pos_outputs, neg_outputs], dim=-1)
                # label = torch.cat([torch.ones_like(pos_outputs), torch.zeros_like(neg_outputs)], dim=-1)
                user_item_loss = criterion['user_item'](pos=pos_outputs,
                                                        neg=neg_outputs)

                if self.n_user_attrs > 0 and 1 in tuple_type:
                    # user-attr training
                    user_ua = user_stream[tuple_type == 1]
                    attr_ua = item_stream[tuple_type == 1]
                    user_attr_neg = self.random_negative_sample(attr_ua, self.n_user_attrs)
                    pos_outputs_ua = self.model(user_ua, attr_ua + self.n_items)
                    neg_outputs_ua = self.model(user_ua, user_attr_neg + self.n_items)
                    # pred_ua = torch.cat([pos_outputs_ua, neg_outputs_ua], dim=-1)
                    # label_ua = torch.cat([torch.ones_like(pos_outputs_ua), torch.zeros_like(neg_outputs_ua)], dim=-1)
                    user_attr_loss = criterion['user_attr'](pos=pos_outputs_ua,
                                                            neg=neg_outputs_ua)
                else:
                    user_attr_loss = 0.0

                if self.n_item_attrs > 0 and 2 in tuple_type:
                    # item-attr training
                    attr = user_stream[tuple_type == 2]
                    item = item_stream[tuple_type == 2]
                    item_neg = self.random_negative_sample(item, self.n_items)
                    pos_outputs_ia = self.model(attr + self.n_users, item)
                    neg_outputs_ia = self.model(attr + self.n_users, item_neg)
                    # pred = torch.cat([pos_outputs, neg_outputs], dim=-1)
                    # label = torch.cat([torch.ones_like(pos_outputs), torch.zeros_like(neg_outputs)], dim=-1)
                    item_attr_loss = criterion['item_attr'](pos=pos_outputs_ia,
                                                            neg=neg_outputs_ia)
                else:
                    item_attr_loss = 0.0

                optimizer.zero_grad()
                loss = user_item_loss + 0.5 * user_attr_loss + 0.5 * item_attr_loss
                if torch.isnan(loss).any():
                    print('nan loss')
                    if self.wandb:
                        wandb.log({'nan_loss': loss.item()})
                    break
                if torch.isinf(loss).any():
                    print('inf loss')
                    if self.wandb:
                        wandb.log({'inf_loss': loss.item()})
                    break
                if self.wandb:
                    wandb.log({'loss': loss.item()})
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(self.train_loader)
            train_losses.append(train_loss)

            self.eval_metrices = self.evaluate()
            if self.val_neg_df is not None:
                mr_101, mrr_101, hr_101, ndcg_101 = self.evaluate_with_random_negatives()
                self.eval_metrices['mr_101'] = mr_101
                self.eval_metrices['mrr_101'] = mrr_101
                self.eval_metrices['hr_101'] = hr_101
                self.eval_metrices['ndcg_101'] = ndcg_101
            pprint.pprint(self.eval_metrices)
            print('epoch: {}, train loss: {}, test loss: {}'.format(epoch, train_loss, train_loss))
            if self.wandb:
                wandb.log(self.eval_metrices)

        return train_losses, test_losses
    
    def get_mask(self, user_list, item_list):
        mask = torch.ones((user_list.size(0), self.n_items), dtype=torch.bool, device=self.device)
        for i, user in enumerate(user_list):
            idx_mask = torch.tensor(self.gt_dict[user.item()], device=self.device)
            mask[i, idx_mask] = False
            mask[i, item_list[i]] = True
        return mask
    
    def evaluate_with_random_negatives(self):
        self.model.eval()
        with torch.no_grad():
            val_dict = self.val_neg_df.to_dict('list')
            users = [int(x) for x in val_dict.keys()]
            items = [x for x in val_dict.values()]
            users = torch.tensor(users, device=self.device)
            items = torch.tensor(items, device=self.device)
            scores = self.model(users, items)
            if self.wandb:
                wandb.log({'scores': scores})
            target_idx = torch.tensor(scores.shape[1] - 1)
            pred_order = torch.argsort(scores, dim=-1, descending=True)
            rank = torch.where(pred_order == target_idx)[1] + 1
            mr_101 = sum(rank) / len(rank)
            mrr_101 = sum(1.0 / rank) / len(rank)
            hr_101 = sum(rank <= 10) / len(rank)
            ndcg_101 = sum(1.0 / torch.log2(rank + 1)) / len(rank)
        return mr_101.item(), mrr_101.item(), hr_101.item(), ndcg_101.item()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            mrr, mr = [], []
            for batch in tqdm(self.val_loader):
                user_stream, item_stream = batch[0][:,0], batch[0][:,1]
                user_stream = user_stream.to(self.device)
                item_stream = item_stream.to(self.device)
                mask = self.get_mask(user_stream, item_stream)
                all_item_score = self.model.predict_item(user_stream.reshape(1, -1)).T
                all_item_score[~mask] = - torch.inf
                pred_order = torch.argsort(all_item_score, dim=-1, descending=True)
                rank = torch.where(pred_order == item_stream.reshape(-1, 1))[1] + 1
                mr.extend(rank.tolist())
                mrr.extend((1.0 / (rank)).tolist())
            eval_metrices = {
                'mrr': sum(mrr) / len(mrr), 
                'mr': sum(mr) / len(mr)
            }
        return eval_metrices