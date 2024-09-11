########################################################################################################################
# Script name:      train.py
# Description:      This python file is used as a definition of the training workflow of the model and also functions
#                   as an evaluation of the results measuring metrics like the Mean Absolute Error and the Mean
#                   Relative Error.
#                   The training is also done here, saving a .pth file of the model that can be later loaded for future
#                   use.
#
# Author:           Javier Ureña Santiago
# Date of creation: 19/01/2023
########################################################################################################################
import subprocess
import warnings

warnings.filterwarnings('ignore')
import torch
import numpy as np
from torch.optim import Adam
from tests.test import TestAfterTrain
import torch.optim as optim
import torch.nn as nn
from utils.dataloader import create_dataset
from utils.utils import create_dataloaders, set_seed, start_logging, save_log, bcolors
from utils.models_utils import loadModel, saveModel, evaluation
from utils.config import load_args
from tqdm import tqdm


# Configure learning rate scheduler:
def lr_scheduler(optimizer, args):
    if args['lr_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
    else:
        scheduler = None
    return scheduler


# Normal training step
def step(x, gt, model, optimizer, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(x)
    loss = loss_fn(outputs, gt)
    loss.backward()
    optimizer.step()
    return loss


def warmup(model, train_loader, optimizer, loss_fn, args, max_lr=0.0001, steps=3000):
    print(f"Initializing model warmup for {steps} steps...")
    lrs = np.linspace(0, max_lr, steps + 1)[1:]
    model.train()
    n = 0

    total_epochs = (steps // len(train_loader)) + 1

    with tqdm(total=steps) as pbar:
        for epoch in range(total_epochs):  # Ensure we have enough epochs to cover all steps
            for i, (images, labels, _) in enumerate(train_loader):
                if n >= steps:
                    break  # Stop if the specified number of steps has been reached
                images, labels = images.to(args['device']), labels.to(args['device'])
                _ = step(images, labels, model, optimizer, loss_fn)
                optimizer.param_groups[0]['lr'] = lrs[n]  # Update learning rate
                n += 1  # Increment step counter
                pbar.update(1)  # Update the progress bar

    print("Warmup finalized")


def train(model, train_loader, val_loader, num_epochs, optimizer, scheduler, loss_fn, args, output_batches=True,
          nBatchesOutput=100, patience=10):
    """
    Training workflow of the model. When running it will print the average loss of every 20 batches and after every
    epoch it will print the MAE using the test loader. It will also save the model with the best MAE measured.
    :param model: pytorch model to train
    :param train_loader: train loader
    :param val_loader: test loader
    :param num_epochs: number of epoch
    :param optimizer: optimizer for minimizing the loss
    :param loss_fn: loss function
    """

    device = args['device']

    if args['warmup']:
        warmup(model, train_loader, optimizer, loss_fn, args, max_lr=args['lr'], steps=args['warmup_steps'])

    print('\nModel pre-evaluation...')
    _, _, _, best_err, _ = evaluation(model, val_loader, loss_fn)
    print(f'Pre-evaluation result: = {best_err.item()}\n')

    tries = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f'\nCurrent Learning Rate of the training: {optimizer.param_groups[0]["lr"]}')
        model.train()
        running_loss = 0.0
        train_loss = []

        for i, (images, labels, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            loss = step(images, labels, model, optimizer, loss_fn)  # apply training step
            loss_value = loss.detach().item()
            running_loss += loss_value  # extract the loss value
            train_loss = np.append(train_loss, loss_value)
            if output_batches and i % nBatchesOutput == nBatchesOutput - 1:
                # print every n batches
                print('Epoch %d, batches %d to %d average loss (MSE): %.3f' % (
                    epoch + 1, i - (nBatchesOutput - 1), i, running_loss / (nBatchesOutput - 1)))
                # zero the loss
                running_loss = 0.0

        mae, rmse, mre, total_err, val_loss = evaluation(model, val_loader, loss_fn)
        train_loss = np.average(train_loss)

        print(
            f'{bcolors.OKGREEN}Epoch {epoch + 1}{bcolors.ENDC} '
            f'validation summary:\nMAE: {mae.item()}\nMRE: {mre.item()}'
            f'\nRMSE: {rmse.item()}\nTotal Error: {total_err.item()}\n')

        save_log(args, epoch + 1, train_loss, val_loss, mae, rmse, mre)

        if args['lr_scheduler']:
            print(f'Average validation loss: {val_loss}\n')
            scheduler.step(val_loss)

        if total_err < best_err:
            saveModel(model, args)
            best_err = total_err
            tries = 0
        else:
            tries += 1

        if tries > patience:
            break


if __name__ == "__main__":
    args = load_args('training')
    set_seed(args['seed'])
    epochs = args['epochs']

    dataset = create_dataset(args)
    train_loader, val_loader = create_dataloaders(dataset, args)

    model = loadModel(model_name=args['model'], args=args)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = lr_scheduler(optimizer, args)

    tester = TestAfterTrain(args)

    print('\n---------------------\nTraining model\n---------------------')
    print(f'Model: {args["model"]}'
          f'\nPre-trained: {args["pretrain"]}')
    print('\n---------------------\nDataset information\n---------------------')
    print(f'Training model on dataset: {args["dataset"]}'
          f'\nDataset size: {dataset.__len__()}'
          f'\nNumber of channels: {args["channels"]}'
          f'\nData dimensions: {dataset.__getitem__(0)[0].shape}'
          f'\nData type: {dataset.__getitem__(0)[0].dtype}'
          f'\nRandom seed: {args["seed"]}')
    print('\n---------------------\nTraining parameters\n---------------------')
    print(f'Train size: {len(train_loader.dataset)}'
          f'\nValidation size: {len(val_loader.dataset)}'
          f'\nBatch size: {args["batch_size"]}'
          f'\nLearning rate: {args["lr"]}'
          f'\nNumber of epochs: {epochs}'
          f'\nLearning rate scheduler: {args["lr_scheduler"]}'
          f'\nPatience: {int(epochs * 0.05)}'
          )
    print('---------------------')

    while args['batch_size'] > 0:
        try:
            start_logging(args)
            train(model, train_loader, val_loader, epochs, optimizer, scheduler, loss_fn, args,
                  patience=int(epochs * 0.05),
                  output_batches=False)
            print(
                f'{bcolors.UNDERLINE}{bcolors.BOLD}{bcolors.HEADER}____Training for model {args["model"]} '
                f'on dataset {args["dataset"]} finalized____{bcolors.ENDC}')

            print(
                f'{bcolors.UNDERLINE}{bcolors.BOLD}{bcolors.WARNING}________Testing the model____{bcolors.ENDC}')
            tester.run(model, loss_fn)
            break
        except RuntimeError as e:
            if 'CUDA out of memory' in e.args[0]:
                torch.cuda.empty_cache()
                print(
                    f'CUDA OUT OF MEMORY! Reducing batch size '
                    f'from {args["batch_size"]} to {int(args["batch_size"] / 2)}')
                args['batch_size'] = int(
                    args['batch_size'] / 2)  # Reduce the batch size to half of it so it fits memory
                train_loader, val_loader = create_dataloaders(dataset, args)

            else:
                raise
