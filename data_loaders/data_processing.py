import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_loaders.tensordataloader import TensorDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matrix_table = {
    'user-item': 0,
    'user-attr': 1,
    'attr-item': 2,
}

class DataProcessing:
    def __init__(self, model_dir, batch_size=128):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.train_df = pd.read_csv(os.path.join(self.model_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.model_dir, 'test.csv'))
        self.user_attributes_df = pd.read_csv(os.path.join(self.model_dir, 'user_attr.csv'))
        self.item_attributes_df = pd.read_csv(os.path.join(self.model_dir, 'item_attr.csv'))
        self.full_data = pd.read_csv(os.path.join(self.model_dir, 'user_item.csv'))
        self.user2id = self.get_id_dict(self.full_data, field='user_id')
        self.item2id = self.get_id_dict(self.full_data, field='item_id')
        self.user_attribute2id = self.get_id_dict(self.user_attributes_df, field='attr_id')
        self.item_attribute2id = self.get_id_dict(self.item_attributes_df, field='attr_id')
    
    def get_data(self):
        return self.train_df, self.test_df, self.user_attributes_df, self.item_attributes_df
    
    def get_id_dict(self, df, field='id'):
        ids = sorted(list(set(df[field].unique().tolist())))
        id2id = {id: i for i, id in enumerate(ids)}
        return id2id

    def get_loader(self):
        print('Preprocessing data...')

        self.train_df['user_id'] = self.train_df['user_id'].map(self.user2id)
        self.train_df['item_id'] = self.train_df['item_id'].map(self.item2id)
        user_ids = self.train_df['user_id'].values
        item_ids = self.train_df['item_id'].values
        self.train_gt_dict = self.get_train_gt_dict()
        user_item_tuples = list(zip(user_ids, item_ids, [matrix_table['user-item']] * len(item_ids)))


        self.user_attributes_df['user_id'] = self.user_attributes_df['user_id'].map(self.user2id)
        self.user_attributes_df['attr_id'] = self.user_attributes_df['attr_id'].map(self.user_attribute2id)
        user_ids = self.user_attributes_df['user_id'].values
        attr_ids = self.user_attributes_df['attr_id'].values
        user_attr_tuples = list(zip(user_ids, attr_ids, [matrix_table['user-attr']] * len(attr_ids)))


        self.item_attributes_df['item_id'] = self.item_attributes_df['item_id'].map(self.item2id)
        self.item_attributes_df['attr_id'] = self.item_attributes_df['attr_id'].map(self.item_attribute2id)
        item_ids = self.item_attributes_df['item_id'].values
        attr_ids = self.item_attributes_df['attr_id'].values
        attr_item_tuples = list(zip(attr_ids, item_ids, [matrix_table['attr-item']] * len(attr_ids)))

        all_tuples = user_item_tuples + user_attr_tuples + attr_item_tuples
        all_tuples = [list(t) for t in all_tuples]
        train_dataset = TensorDataset(torch.LongTensor(all_tuples))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        return self.train_loader
    
    def get_train_gt_dict(self):
        train_gt_dict = {}
        for user_id, item_id in zip(self.train_df['user_id'].values, self.train_df['item_id'].values):
            if user_id not in train_gt_dict:
                train_gt_dict[user_id] = []
            train_gt_dict[user_id].append(item_id)
        return train_gt_dict
    
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
        self.test_gt_dict = self.get_test_gt_dict()
        test_dataset = TensorDataset(torch.LongTensor(test_user_ids),
                                    torch.LongTensor(test_item_ids))
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self.test_loader


class MovieLensDataProcessing:
    def __init__(self, data_dir, dataset_type='movielens', batch_size=128):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.read_data_files()
        self.read_id_files()

    def read_data_files(self):
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.val_df = pd.read_csv(os.path.join(self.data_dir, 'val.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        if self.dataset_type == 'movielens-genre':
            self.item_attributes_df = pd.read_csv(os.path.join(self.data_dir, 'tag2movie.csv'))
    
    def read_id_files(self):
        self.user2id_df = pd.read_csv(os.path.join(self.data_dir, 'user2id.csv'))
        self.item2id_df = pd.read_csv(os.path.join(self.data_dir, 'movie2id.csv'))
        if self.dataset_type == 'movielens-genre':
            self.item_attribute2id_df = pd.read_csv(os.path.join(self.data_dir, 'tag2id.csv'))
        
        self.user2id = {i: i for i in self.user2id_df['userId'].values}
        self.item2id = {i: i for i in self.item2id_df['movieId'].values}
        if self.dataset_type == 'movielens-genre':
            self.item_attribute2id = {i: i for i in self.item_attribute2id_df['tagId'].values}

    def get_id_dict(self, df, field='id'):
        ids = sorted(list(set(df[field].unique().tolist())))
        id2id = {id: i for i, id in enumerate(ids)}
        return id2id

    def load_data(self, data, tag_data=None):
        user = data['userId'].values
        item = data['movieId'].values
        data_tuples = list(zip(user, item, [matrix_table['user-item']] * len(item)))
        if self.dataset_type == 'movielens-genre' and tag_data is not None:
            tag = tag_data['tagId'].values
            item = tag_data['movieId'].values
            tag_tuples = list(zip(tag, item, [matrix_table['attr-item']] * len(item)))
            data_tuples += tag_tuples
        dataset = TensorDataset(torch.LongTensor(data_tuples).to(device))
        return TensorDataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

    def get_loader(self):
        if self.dataset_type == 'movielens-genre':
            return self.load_data(self.train_df, self.item_attributes_df)
        return self.load_data(self.train_df)

    def get_val_loader(self):
        return self.load_data(self.val_df)

    def get_test_loader(self):
        return self.load_data(self.test_df)
    
    def get_tag_loader(self):
        return self.load_data(self.item_attributes_df)

    def get_gt_dict(self, df):
        gt_dict = {}
        for userId, itemId in zip(df['userId'].values, df['movieId'].values):
            if userId not in gt_dict:
                gt_dict[userId] = []
            gt_dict[userId].append(itemId)
        return gt_dict
