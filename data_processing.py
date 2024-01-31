import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

class DataProcessing:
    def __init__(self, model_dir, batch_size=128):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.train_df = pd.read_csv(os.path.join(self.model_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.model_dir, 'test.csv'))
        self.user_attributes_df = pd.read_csv(os.path.join(self.model_dir, 'user_attr.csv'))
        self.item_attributes_df = pd.read_csv(os.path.join(self.model_dir, 'item_attr.csv'))
        self.full_data = pd.read_csv(os.path.join(self.model_dir, 'user_item.csv'))
    
    def get_data(self):
        return self.train_df, self.test_df, self.user_attributes_df, self.item_attributes_df
    
    def get_id_dict(self, df, field='id'):
        ids = sorted(list(set(df[field].unique().tolist())))
        id2id = {id: i for i, id in enumerate(ids)}
        return id2id

    def get_loader(self):
        print('Preprocessing data...')
        self.user2id = self.get_id_dict(self.full_data, field='user_id')
        self.item2id = self.get_id_dict(self.full_data, field='item_id')
        self.user_attribute2id = self.get_id_dict(self.user_attributes_df, field='attr_id')
        self.item_attribute2id = self.get_id_dict(self.item_attributes_df, field='attr_id')

        self.train_df['user_id'] = self.train_df['user_id'].map(self.user2id)
        self.train_df['item_id'] = self.train_df['item_id'].map(self.item2id)
    
        self.user_attributes_df['user_id'] = self.user_attributes_df['user_id'].map(self.user2id)
        self.item_attributes_df['item_id'] = self.item_attributes_df['item_id'].map(self.item2id)
        self.user_attributes_df['attr_id'] = self.user_attributes_df['attr_id'].map(self.user_attribute2id)
        self.item_attributes_df['attr_id'] = self.item_attributes_df['attr_id'].map(self.item_attribute2id)


        train_user_ids = self.train_df['user_id'].values
        train_item_ids = self.train_df['item_id'].values
        # train_user_attributes = self.user_attributes_df['attr_id'].values
        # train_item_attributes = self.item_attributes_df['attr_id'].values
        train_ratings = torch.ones(len(train_user_ids))
        train_dataset = TensorDataset(torch.LongTensor(train_user_ids),
                                    torch.LongTensor(train_item_ids),
                                    torch.FloatTensor(train_ratings))
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        return self.train_loader
    
    def get_test_gt_dict(self):
        test_gt_dict = {}
        for user_id, item_id in zip(self.test_df['user_id'].values, self.test_df['item_id'].values):
            if user_id not in test_gt_dict:
                test_gt_dict[user_id] = []
            test_gt_dict[user_id].append(item_id)
        return test_gt_dict
    
    def get_test_loader(self):
        print('Preprocessing test data...')
        self.test_df['user_id'] = self.test_df['user_id'].map(self.user2id)
        self.test_df['item_id'] = self.test_df['item_id'].map(self.item2id)
        test_user_ids = self.test_df['user_id'].values
        test_item_ids = self.test_df['item_id'].values
        test_ratings = torch.ones(len(test_user_ids))
        test_dataset = TensorDataset(torch.LongTensor(test_user_ids),
                                    torch.LongTensor(test_item_ids),
                                    torch.FloatTensor(test_ratings))
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self.test_loader