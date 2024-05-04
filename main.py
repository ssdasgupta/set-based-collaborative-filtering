"""Main script to run the training process."""
import random
import json
import argparse
import datetime
import os
import shutil
import numpy as np
import torch
import wandb
from model import MatrixFactorization, MatrixFactorizationWithBias
from model.box_model import BoxRec, BoxRecConditional
from trainer import Trainer
from data_loaders.data_processing import (
    DataProcessing,
    MovieLensDataProcessing, JointDataProcessing
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """Main function to parse arguments and run the training process."""
    print('Parsing arguments...')
    parser = argparse.ArgumentParser()

    ## model related input parameters
    parser.add_argument('--model_dir',
                        type=str,
                        default='checkpoints',
                        help='model directory path')
    parser.add_argument('--model_name',
                        type=str,
                        default='model',
                        help='model name')
    parser.add_argument('--model',
                        type=str, default='mf',
                        choices=['mf', 'mf_bias', 'box', 'box_conditional'],
                        help='model name')
    parser.add_argument('--fixed_neg_eval',
                        action='store_true',
                        help='fixed sampled negative evaluation')
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
    parser.add_argument('--dataset_type',
                        type=str,
                        default='joint',
                        choices=['synthetic', 'movielens', 'user-item', 'attribute-item', 'joint'],
                        help='model type')
    parser.add_argument('--data_dir',
                        type=str,
                        default='data/',
                        help='model type')
    parser.add_argument('--dataset',
                        type=str,
                        default='user_genre_movie',
                        choices= ['user_genre_movie', 'user_attribute_movie'],
                        help='dataset')
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
                        choices=['bce', 'max-margin', 'mse', 'bce_logits'],
                        help='loss type')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='optimizer type')
    parser.add_argument('--n_train_negs', type=int, default=5, help='the number of negative samples for training')
    parser.add_argument('--n_test_negs', type=int, default=100, help='the number of negative samples for validation')
    parser.add_argument('--attribute_loss_const', type=float, default=0.1, help='attribute loss constant')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

    ## Logging and reproducibility
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--wandb', action='store_true', help='wandb')
    parser.add_argument('--save_model', action='store_true', help='save model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()

    args.model_dir = os.path.join(args.model_dir,
                                  args.model,
                                  args.dataset,
                                  'dim_' + str(args.embedding_dim) + '-' + 'negs_' + str(args.n_train_negs)
                )
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.wandb:
        run = wandb.init(project="box-rec-sys", reinit=True)
        wandb.config.update(args)

    if args.dataset_type == 'synthetic':
        dataset = DataProcessing(args.data_dir, args.batch_size)
        n_users = len(dataset.user2id)
        n_items = len(dataset.item2id)
        n_user_attrs = len(dataset.user_attribute2id)
        n_item_attrs = len(dataset.item_attribute2id)
    elif args.dataset_type == 'movielens':
        dataset = MovieLensDataProcessing(data_dir=args.data_dir,
                                          dataset_type=args.dataset_type,
                                          batch_size=args.batch_size)
        print('Loading ground truth...')
        print('Ground truth loaded')
        n_users = len(dataset.user2id)
        n_items = len(dataset.item2id)
        n_user_attrs = 0
        n_item_attrs = 0

    elif args.dataset_type == 'joint' or args.dataset_type == 'attribute-item' or args.dataset_type == 'joint':
        dataset = JointDataProcessing(data_dir=args.data_dir,
                                        dataset_type=args.dataset_type,
                                        batch_size=args.batch_size)
        n_users = dataset.n_users
        n_items = dataset.n_movies
        n_user_attrs = 0
        n_item_attrs = dataset.n_attributes

    else:
        raise NotImplementedError

    print('Building data loaders...')
    train_loader = dataset.get_loader()
    val_loader = dataset.get_val_loader()
    if args.fixed_neg_eval:
        dataset.read_neg_data_files()
    print('Data loaders built')


    print('Building model... ')
    if args.model == 'mf':
        model = MatrixFactorization(n_users=n_users + n_item_attrs,
                                    n_items=n_items + n_user_attrs,
                                    embedding_dim=args.embedding_dim)
    elif args.model == 'mf_bias':
        model = MatrixFactorizationWithBias(n_users=n_users + n_item_attrs,
                                            n_items=n_items + n_user_attrs,
                                            embedding_dim=args.embedding_dim)
    elif args.model == 'box':
        model = BoxRec(n_users=n_users + n_item_attrs,
                        n_items=n_items + n_user_attrs,
                        embedding_dim=args.embedding_dim,
                        box_type=args.box_type,
                        volume_temp=args.volume_temp,
                        intersection_temp=args.intersection_temp)
    elif args.model == 'box_conditional':
        model = BoxRecConditional(n_users=n_users + n_item_attrs,
                        n_items=n_items + n_user_attrs,
                        embedding_dim=args.embedding_dim,
                        box_type=args.box_type,
                        volume_temp=args.volume_temp,
                        intersection_temp=args.intersection_temp)
    else:
        raise NotImplementedError

    model.to(device)
    
    if args.save_model:
        if args.verbose:
            print('Saving model...')
        if os.path.exists(args.model_dir):
            shutil.rmtree(args.model_dir) # [[[Be very very careful in using this, always think twice before changing anything in model saving mechanism.]]]
        os.makedirs(args.model_dir)
        # save args as json
        args.num_users = n_users + n_item_attrs
        args.num_items = n_items + n_user_attrs
        with open(os.path.join(args.model_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f)

    print('Training model...')
    trainer = Trainer(
        model=model,
        n_users=n_users,
        n_items=n_items,
        n_user_attrs=n_user_attrs,
        n_item_attrs=n_item_attrs,
        train_loader=train_loader,
        val_loader=val_loader,
        fixed_neg_eval=args.fixed_neg_eval,
        dataset=dataset,
        loss_type=args.loss_type,
        optimizer_type=args.optimizer_type,
        n_train_negs=args.n_train_negs,
        n_test_negs=args.n_test_negs,
        attribute_loss_const=args.attribute_loss_const,
        device=device,
        model_name=args.model_name,
        use_wandb=args.wandb,
        model_dir=args.model_dir,
        save_model=args.save_model,
    )
    trainer.train(
        epochs=args.n_epochs,
        lr=args.lr,
        wd=args.wd,
    )


if __name__ == '__main__':
    main()
