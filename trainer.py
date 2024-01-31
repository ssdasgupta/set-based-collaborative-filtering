import torch
from torch import nn
import torch.optim as optim
import os



class Trainer:
    def __init__(
            self,
            n_user,
            n_item,
            n_user_attrs,
            n_item_attrs,
            train_loader,
            device,
            model_dir,
            model_name,
            verbose
    ):
        self.n_user = n_user
        self.n_item = n_item
        self.n_user_attrs = n_user_attrs
        self.n_item_attrs = n_item_attrs
        self.train_loader = train_loader
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.verbose = verbose
        


    def get_criteria(self, loss_type):
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'bpr':
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
            return optim.SparseAdam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise NotImplementedError

    def random_negative_sample(self, ids, n, n_negs=1):
        # user_ids: (batch_size, )
        # item_ids: (batch_size, )
        batch_size = ids.size(0)
        neg_ids = torch.randint(low=0, high=n, size=(batch_size, n_negs))
        return neg_ids

    def get_attributes(self, id, type='user'):
        # sample from a list
        pass


    def train(self, epochs, lr):
        self.model.train()
        self.model.to(self.device)
        optimizer = optim.SparseAdam(self.model.parameters(), lr=lr) # do we need to use sparseadam? or adam is fine?
        criterion = {
            'user_item': nn.BCEWithLogitsLoss(),
            'user_attr': nn.BCEWithLogitsLoss(),
            'item_attr': nn.BCEWithLogitsLoss(),
        }
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            train_loss = 0.0
            for i, (user, item, rating) in enumerate(self.train_loader):
                user_attributes = get_attributes(user)
                item_attributes = get_attributes(item)

                user = user.to(self.device)
                item = item.to(self.device)
                user_attributes = user_attributes.to(self.device)
                item_attributes = item_attributes.to(self.device)

                rating = rating.to(self.device)
                user_neg = self.random_negative_sample(user, self.n_user)
                item_neg = self.random_negative_sample(item, self.n_item)
                pos_outputs = self.model(user, item)
                neg_outputs_item = self.model(user, item_neg)
                neg_outputs_user = self.model(user_neg, item)
                neg_outputs = torch.cat([neg_outputs_item, neg_outputs_user], dim=0)
                pred = torch.cat([pos_outputs, neg_outputs], dim=0)
                label = torch.cat([torch.ones(pos_outputs.size(0)), torch.zeros(neg_outputs.size(0))], dim=0)
                user_item_loss = criterion['user_item'](pred, label)

                user_attr_neg = self.random_negative_sample(user_attributes, self.n_user_attrs)
                pos_outputs = self.model(user, user_attributes)
                neg_outputs = self.model(user, item_attributes)
                pred = torch.cat([pos_outputs, neg_outputs], dim=0)
                label = torch.cat([torch.ones(pos_outputs.size(0)), torch.zeros(neg_outputs.size(0))], dim=0)
                user_attr_loss = criterion['user_attr'](pred, label)

                item_attr_neg = self.random_negative_sample(item_attributes, self.n_item_attrs)
                pos_outputs = self.model(item_attributes, user_attributes)
                
                
                rating = rating.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(user, item, user_attributes, item_attributes)
                loss = criterion(outputs, rating)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(self.train_loader)
            train_losses.append(train_loss)

            # test_loss = 0.0
            # for i, (user, item, user_attributes, item_attributes, rating) in enumerate(test_loader):
            #     user = user.to(device)
            #     item = item.to(device)
            #     user_attributes = user_attributes.to(device)
            #     item_attributes = item_attributes.to(device)
            #     rating = rating.to(device)
            #     outputs = self.model(user, item, user_attributes, item_attributes)
            #     loss = criterion(outputs, rating)
            #     test_loss += loss.item()
            # test_loss = test_loss / len(test_loader)
            # test_losses.append(test_loss)
            if self.verbose:
                print('epoch: {}, train loss: {}, test loss: {}'.format(epoch, train_loss))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name))
        return train_losses, test_losses

    def evaluate(self, model, test_loader, device):
        model.eval()
        criterion = nn.MSELoss()
        test_loss = 0.0
        with torch.no_grad():
            for i, (user, item, user_attributes, item_attributes, rating) in enumerate(test_loader):
                user = user.to(device)
                item = item.to(device)
                user_attributes = user_attributes.to(device)
                item_attributes = item_attributes.to(device)
                rating = rating.to(device)
                outputs = model(user, item, user_attributes, item_attributes)
                loss = criterion(outputs, rating)
                test_loss += loss.item()
        test_loss = test_loss / len(test_loader)
        return test_loss