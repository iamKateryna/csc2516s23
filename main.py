import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


import argparse
import wandb
import random

from wildfire import WildfireDataset
from wildfirebestfeatures import WildfireBestFeaturesDataset
from train import train
from evaluate import evaluate
from custom_loss import weighted_cross_entropy_with_logits_with_masked_class

from constants import OUTPUT_FEATURES, BEST_INPUT_FEATURES

np.random.seed(42)
torch.manual_seed(42)


PROJECT_NAME = 'wildfire-prediction'
CHECKPOINT_PATH = 'checkpoints/'


def prepare_data(dataset_size, features):

    if dataset_size == 'sm':
        train_data = torch.load('dataset/small_train_data.pt')
        val_data = torch.load('dataset/small_eval_data.pt')
        test_data = torch.load('dataset/small_test_data.pt')
    elif dataset_size == 'md':
        train_data = torch.load('dataset/med_train_data.pt')
        val_data = torch.load('dataset/med_eval_data.pt')
        test_data = torch.load('dataset/med_test_data.pt')
    elif dataset_size == 'full':
        train_data = torch.load('dataset/train_data.pt')
        val_data = torch.load('dataset/eval_data.pt')
        test_data = torch.load('dataset/test_data.pt')
    elif dataset_size == 'fourty_p':
        train_data = torch.load('dataset/fourty_p_train_data.pt')
        val_data = torch.load('dataset/fourty_p_eval_data.pt')
        test_data = torch.load('dataset/fourty_p_test_data.pt')
    elif dataset_size == 'sixty_p':
        train_data = torch.load('dataset/sixty_p_train_data.pt')
        val_data = torch.load('dataset/sixty_p_eval_data.pt')
        test_data = torch.load('dataset/sixty_p_test_data.pt')
    else: 
        print('Invalid dataset size value!!')

    if features == 'all':
        train_data = WildfireDataset(train_data, transform = False)
        val_data = WildfireDataset(val_data, transform = False)
        test_data = WildfireDataset(test_data, transform = False)
    elif features == 'best':
        print(BEST_INPUT_FEATURES)
        train_data = WildfireBestFeaturesDataset(train_data, transform = False)
        val_data = WildfireBestFeaturesDataset(val_data, transform = False)
        test_data = WildfireBestFeaturesDataset(test_data, transform = False)
    else: 
        print('Invalid features value!!')

    return train_data, val_data, test_data


def get_model(device, features, encoder_weights):

    if features == 'all':

       model = smp.Unet(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=12,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        activation= 'sigmoid'
       )

    if features == 'best':

       model = smp.Unet(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        activation= 'sigmoid'
       )

    return model.to(device)


def main(args):

    model_path = f'{CHECKPOINT_PATH}/{args.lr}-{args.features}-{args.dataset_size}-model.pt'
    group_name = f'{args.dataset_size}-{args.features}-{args.lr}-{args.encoder_weights}'
    exp_prefix = f'{group_name}-{random.randint(int(1), int(1e4) - 1)}'

    wandb.init(project=PROJECT_NAME, group=group_name, name=exp_prefix)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, val_data, test_data = prepare_data(args.dataset_size, args.features)

    # print(f'Train/val/test sizes: {train_data.__len__()}/{val_data.__len__()}/{test_data.__len__()}')

    number_of_classes = len(OUTPUT_FEATURES)
    model = get_model(device, args.features, args.encoder_weights)
    dataloaders = {
        'train': DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True),
        'val': DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True),
        'test': DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    }

    criterion = weighted_cross_entropy_with_logits_with_masked_class()
    optimizer = torch.optim.AdamW(model.parameters(), lr=10e-5, weight_decay=2)

    best_loss = np.inf
    
    print('------- S H O R T  I N F O ------')
    print(f'lr {args.lr}')
    print(f'features {args.features}')
    print(f'dataset_size {args.dataset_size}')

    # Train and test over n_epochs
    for epoch in tqdm(range(args.num_epochs)):
        print(f'Epoch {epoch+1}')
        train(model, device, dataloaders['train'],
              number_of_classes, criterion, optimizer)
        print('start evaluating')
        val_loss, _, _ = evaluate(
            model, device, dataloaders['val'], number_of_classes, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)

            if wandb.run is not None:
                wandb.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--features', type=str, default='all')
    parser.add_argument('--dataset_size', type=str, default='medium')
    parser.add_argument('--encoder_weights', type=str, default=None)

    args = parser.parse_args()
    main(args)

