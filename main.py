import torch


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from model import MatrixFactorization
from model import MatrixFactorizationWithBias
from trainer import Trainer
from data_processing import DataProcessing


import argparse
import pickle
import datetime
import wandb
import os

#seed for reproducibility
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print('Parsing arguments...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    parser.add_argument('--predict', action='store_true', help='predict the model')
    parser.add_argument('--model_dir', type=str, default='model', help='model directory path')
    parser.add_argument('--model_name', type=str, default='model.pth', help='model name')
    parser.add_argument('--model', type=str, default='mf', help='model name')
    parser.add_argument('--data_dir', type=str, default='data/', help='model type')
    parser.add_argument('--user_attributes_data', type=str, default='data/user_attributes.csv', help='user attributes data path')
    parser.add_argument('--item_attributes_data', type=str, default='data/item_attributes.csv', help='item attributes data path')
    parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--loss_type', type=str, default='mse', help='loss type')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='optimizer type')
    parser.add_argument('--n_negs', type=int, default=5, help='the number of negative samples')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--embedding_dim', type=int, default=20, help='embedding dimension')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--wandb', action='store_true', help='wandb')
    args = parser.parse_args()

    args.model_dir = os.path.join(args.model_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if args.wandb:
        run = wandb.init(project="box-rec-sys", reinit=True)
   
    dataset = DataProcessing(args.data_dir, args.batch_size)
    n_users = len(dataset.user2id) 
    n_user_attrs = len(dataset.user_attribute2id)
    n_items = len(dataset.item2id)
    n_item_attrs = len(dataset.item_attribute2id)
    train_loader = dataset.get_loader()
    test_loader = dataset.get_test_loader()


    print('Building model...')
    model = MatrixFactorizationWithBias(n_users= n_users + n_item_attrs,
                                        n_items=n_items + n_user_attrs,
                                        embedding_dim=args.embedding_dim)

    if args.train:
        print('Training model...')
        trainer = Trainer(
            model=model,
            n_users=n_users,
            n_items=n_items,
            n_user_attrs=n_user_attrs,
            n_item_attrs=n_item_attrs,
            user_attributes_df=dataset.user_attributes_df,
            item_attributes_df=dataset.item_attributes_df,
            train_loader=train_loader,
            test_loader=test_loader,
            train_gt_dict=dataset.train_gt_dict,
            test_gt_dict=dataset.test_gt_dict,
            loss_type=args.loss_type,
            optimizer_type=args.optimizer_type,
            n_negs=args.n_negs,
            device=device,
            model_dir=args.model_dir, 
            model_name=args.model_name,
            verbose=args.verbose,
            wandb=args.wandb
        )
        train_losses, test_losses = trainer.train(
            epochs=args.n_epochs, 
            lr=args.lr,
            wd=args.wd,
        )

        if args.verbose:
            print('Saving losses...')
        with open(os.path.join(args.model_dir, 'train_losses.pkl'), 'wb') as f:
            pickle.dump(train_losses, f)
        with open(os.path.join(args.model_dir, 'test_losses.pkl'), 'wb') as f:
            pickle.dump(test_losses, f)

    
    # <<<<<<<<<TODO>>>>>>>>>> #
    ## 1. Evaluate the model. Implement rank based metrics.
            ### 1.1. Implement MRR, NDCG, Recall, Precision, MAP.
            ### 1.2. Test Dataloader.
            #### 1.2.1. Test is on the rules
            #### 1.2.2. Test is on the heldout item list.

            ### 1.3 Implement test pipeline.
    ## 2. Implement negative sampling.

    # if args.evaluate:
    #     print('Evaluating model...')
    #     model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    #     test_loss = evaluate(model, test_loader, device)
    #     print('test loss: {}'.format(test_loss))
    # if args.predict:
    #     print('Predicting model...')
    #     model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    #     model.eval()
    #     with torch.no_grad():
    #         for i, (user, item, user_attributes, item_attributes, rating) in enumerate(test_loader):
    #             user = user.to(device)
    #             item = item.to(device)
    #             user_attributes = user_attributes.to(device)
    #             item_attributes = item_attributes.to(device)
    #             rating = rating.to(device)
    #             outputs = model(user, item, user_attributes, item_attributes)
    #             print('user: {}, item: {}, user_attributes: {}, item_attributes: {}, rating: {}, predicted rating: {}'.format(user, item, user_attributes, item_attributes, rating, outputs))


if __name__ == '__main__':
    main()
