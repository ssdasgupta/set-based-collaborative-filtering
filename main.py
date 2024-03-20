import random
import numpy as np
import torch


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from model import MatrixFactorization, MatrixFactorizationWithBias
from model.box_model import BoxRec
from trainer import Trainer
from data_loaders.data_processing import DataProcessing, MovieLensDataProcessing


import argparse
import datetime
import wandb
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print('Parsing arguments...')
    parser = argparse.ArgumentParser()

    ## model related input parameters
    parser.add_argument('--model_dir',
                        type=str,
                        default='model',
                        help='model directory path')
    parser.add_argument('--model_name',
                        type=str,
                        default='model.pth',
                        help='model name')
    parser.add_argument('--model',
                        type=str, default='mf',
                        choices=['mf', 'mf_bias', 'box'],
                        help='model name')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=20,
                        help='embedding dimension. For boxes use half the dimension size')
    parser.add_argument('--box_type',
                        type=str,
                        default='BoxTensor',
                        help='box type')
    parser.add_argument('--volume_temp',
                        type=float,
                        default=1.0,
                        help='volume temperature')
    parser.add_argument('--intersection_temp',
                        type=float,
                        default=0.1,
                        help='intersection temperature')

    ## data related input parameters
    parser.add_argument('--dataset',
                        type=str,
                        default='synthetic',
                        help='model type')
    parser.add_argument('--data_dir',
                        type=str,
                        default='data/',
                        help='model type')
    parser.add_argument('--user_attributes_data',
                        type=str, default=None,
                        help='user attributes data path')
    parser.add_argument('--item_attributes_data',
                        type=str,
                        default=None,
                        help='item attributes data path')

    ## training related input parameters
    parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--loss_type',
                        type=str,
                        default='bce',
                        choices=['bce', 'max-margin', 'mse'],
                        help='loss type')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='optimizer type')
    parser.add_argument('--n_negs', type=int, default=5, help='the number of negative samples')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

    ## Logging and reproducibility
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--wandb', action='store_true', help='wandb')
    parser.add_argument('--save_model', action='store_true', help='save model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()

    args.model_dir = os.path.join(args.model_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.wandb:
        run = wandb.init(project="box-rec-sys", reinit=True)
        wandb.config.update(args)
    
    if args.dataset == 'synthetic':
        dataset = DataProcessing(args.data_dir, args.batch_size)
    elif args.dataset == 'movielens' or args.dataset == 'movielens-genre':
        dataset = MovieLensDataProcessing(data_dir=args.data_dir,
                                          dataset_type=args.dataset,
                                          batch_size=args.batch_size)
        gt_dict = dataset.get_gt_dict(dataset.gt_df)
    else:
        raise NotImplementedError

    n_users = len(dataset.user2id)
    n_items = len(dataset.item2id)
    if args.dataset == 'synthetic':
        n_user_attrs = len(dataset.user_attribute2id)
        n_item_attrs = len(dataset.item_attribute2id)
    elif args.dataset == 'movielens-genre':
        n_user_attrs = 0
        n_item_attrs = len(dataset.item_attribute2id)
    elif args.dataset == 'movielens':
        n_user_attrs = 0
        n_item_attrs = 0
    else:
        raise NotImplementedError

    train_loader = dataset.get_loader()
    val_loader = dataset.get_val_loader()


    print('Building model... ')
    if args.model == 'mf':
        model = MatrixFactorization(n_users= n_users + n_item_attrs,
                                    n_items=n_items + n_user_attrs,
                                    embedding_dim=args.embedding_dim)
    elif args.model == 'mf_bias':
        model = MatrixFactorizationWithBias(n_users= n_users + n_item_attrs,
                                            n_items=n_items + n_user_attrs,
                                            embedding_dim=args.embedding_dim)
    elif args.model == 'box':
        model = BoxRec(n_users=n_users + n_item_attrs,
                        n_items=n_items + n_user_attrs,
                        embedding_dim=args.embedding_dim,
                        box_type=args.box_type,
                        volume_temp=args.volume_temp,
                        intersection_temp=args.intersection_temp)

    model.to(device)

    print('Training model...')
    trainer = Trainer(
        model=model,
        n_users=n_users,
        n_items=n_items,
        n_user_attrs=n_user_attrs,
        n_item_attrs=n_item_attrs,
        train_loader=train_loader,
        val_loader=val_loader,
        gt_dict = gt_dict,
        loss_type=args.loss_type,
        optimizer_type=args.optimizer_type,
        n_negs=args.n_negs,
        device=device,
        model_dir=args.model_dir,
        model_name=args.model_name,
        wandb=args.wandb
    )
    train_losses, test_losses = trainer.train(
        epochs=args.n_epochs,
        lr=args.lr,
        wd=args.wd,
    )

    if args.save_model:
        if args.verbose:
            print('Saving model...')
        torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name))


if __name__ == '__main__':
    main()
