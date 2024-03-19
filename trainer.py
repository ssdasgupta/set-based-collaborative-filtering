import torch
from torch import nn
import torch.optim as optim
import os
from tqdm import tqdm
import pprint
import wandb

from eval_metrics import recall_At_k, precision_At_k, AP, NDCG, MRR, MAP, MR

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
            test_loader,
            train_gt_dict,
            test_gt_dict,
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
        self.test_loader = test_loader
        self.train_gt_dict = train_gt_dict
        self.test_gt_dict = test_gt_dict
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.n_negs = n_negs
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.wandb = wandb
    

    def get_criteria(self, loss_type):
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'max-margin':
            return nn.MarginRankingLoss()
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
        neg_ids = torch.randint(low=0, high=n, size=(batch_size, n_negs)).to(self.device)
        return neg_ids


    def train(self, epochs, lr, wd):
        self.model.train()
        self.model.to(self.device)
        optimizer = self.get_optimizer(self.optimizer_type, self.model, lr, wd)  # do we need to use sparseadam? or adam is fine?
        criterion = {
            'user_item': self.get_criteria(self.loss_type),
            'user_attr': self.get_criteria(self.loss_type),
            'item_attr': self.get_criteria(self.loss_type),
        }
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            train_loss = 0.0
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
                pos_outputs = self.model(user, item).unsqueeze_(-1)
                neg_outputs_item = self.model(user, item_neg)
                neg_outputs_user = self.model(user_neg, item)
                neg_outputs = torch.cat([neg_outputs_item, neg_outputs_user], dim=-1)

                pred = torch.cat([pos_outputs, neg_outputs], dim=-1)
                label = torch.cat([torch.ones_like(pos_outputs), torch.zeros_like(neg_outputs)], dim=-1)
                user_item_loss = criterion['user_item'](pred, label)

                if self.n_user_attrs > 0 and 1 in tuple_type:
                    # user-attr training
                    user_ua = user_stream[tuple_type == 1]
                    attr_ua = item_stream[tuple_type == 1]
                    user_attr_neg = self.random_negative_sample(attr_ua, self.n_user_attrs)
                    pos_outputs_ua = self.model(user_ua, attr_ua + self.n_items).unsqueeze_(-1)
                    neg_outputs_ua = self.model(user_ua, user_attr_neg + self.n_items)
                    pred_ua = torch.cat([pos_outputs_ua, neg_outputs_ua], dim=-1)
                    label_ua = torch.cat([torch.ones_like(pos_outputs_ua), torch.zeros_like(neg_outputs_ua)], dim=-1)
                    user_attr_loss = criterion['user_attr'](pred_ua, label_ua)
                else:
                    user_attr_loss = 0.0

                if self.n_item_attrs > 0 and 2 in tuple_type:
                    # item-attr training
                    attr = user_stream[tuple_type == 2]
                    item = item_stream[tuple_type == 2]
                    item_neg = self.random_negative_sample(item, self.n_items)
                    pos_outputs = self.model(attr + self.n_users, item).unsqueeze_(-1)
                    neg_outputs = self.model(attr + self.n_users, item_neg)
                    pred = torch.cat([pos_outputs, neg_outputs], dim=-1)
                    label = torch.cat([torch.ones_like(pos_outputs), torch.zeros_like(neg_outputs)], dim=-1)
                    item_attr_loss = criterion['item_attr'](pred, label)
                else:
                    item_attr_loss = 0.0

                optimizer.zero_grad()
                loss = 0.6 * user_item_loss + 0.2 * user_attr_loss + 0.2 * item_attr_loss
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
            pprint.pprint(self.eval_metrices)
            print('epoch: {}, train loss: {}, test loss: {}'.format(epoch, train_loss, train_loss))
            if self.wandb:
                wandb.log(self.eval_metrices)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name))
        return train_losses, test_losses

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            recall_1, recall_5, recall_50, precision_1, precision_5, precision_10, ap, ndcg_1, ndcg_5, ndcg_10, mrr, mr = [], [], [], [], [], [], [], [], [], [], [], []
            for user, items in tqdm(self.test_gt_dict.items()):
                gt = self.test_gt_dict[user]
                mask = torch.LongTensor(self.train_gt_dict[user]).to(self.device)

                user = torch.LongTensor([user]).to(self.device)
                items = torch.LongTensor(items).to(self.device)

                all_item_score = self.model.predict_item(user)
                # remove the items that are already in the training set
                all_item_score[mask] = -1e9

                pred_order = torch.argsort(all_item_score, descending=True).tolist()

                recall_1.append(recall_At_k(pred_order, gt, k=1))
                recall_5.append(recall_At_k(pred_order, gt, k=5))
                recall_50.append(recall_At_k(pred_order, gt, k=10))
                precision_1.append(precision_At_k(pred_order, gt, k=1))
                precision_5.append(precision_At_k(pred_order, gt, k=5))
                precision_10.append(precision_At_k(pred_order, gt, k=10))
                ap.append(AP(pred_order, gt, k=10))
                ndcg_1.append(NDCG(pred_order, gt, k=1))
                ndcg_5.append(NDCG(pred_order, gt, k=5))
                ndcg_10.append(NDCG(pred_order, gt, k=10))
                mrr.append(MRR(pred_order, gt, k=10))
                mr.append(MR(pred_order, gt))
            eval_metrices = {
                'recall_1': sum(recall_1) / len(recall_1),
                'recall_5': sum(recall_5) / len(recall_5),
                'recall_50': sum(recall_50) / len(recall_50),
                'precision_1': sum(precision_1) / len(precision_1),
                'precision_5': sum(precision_5) / len(precision_5),
                'precision_10': sum(precision_10) / len(precision_10),
                'ap': sum(ap) / len(ap),
                'ndcg_1': sum(ndcg_1) / len(ndcg_1),
                'ndcg_5': sum(ndcg_5) / len(ndcg_5),
                'ndcg_10': sum(ndcg_10) / len(ndcg_10),
                'mrr': sum(mrr) / len(mrr), 
                'mr': sum(mr) / len(mr)
            }
        return eval_metrices