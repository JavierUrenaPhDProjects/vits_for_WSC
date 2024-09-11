########################################################################################################################
# Script name:      utils.py
# Description:      This python file has all the useful, self made functions used in the program. These functions
#                   allows the creation of the dataset, pre-processing steps, and data augmentation. Also it defines
#                   some useful functions for data interpretation and system configuration.
#
# Author:           Javier Ureña Santiago
# Date of creation: 19/01/2023
########################################################################################################################

import numpy as np
import os
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import random
import shutil
import pandas as pd


def create_dataloaders(dataset, args):
    """
    Create the dataloaders for training and testing from the dataset
    :param dataset: dataset class
    :param args: arguments
    :return:
        train_loader: train loader
        val_loader:  test loader
    """
    trainPercnt = args['train_percnt']
    batchSize = args['batch_size']
    datasetSize = len(dataset)
    trainsize = int(trainPercnt * datasetSize)
    testsize = datasetSize - trainsize
    train_dataset, test_dataset = random_split(dataset, [trainsize, testsize])
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader


def set_seed(seed=1000):
    """
    Sets the seed for reproducability

    Args:
        seed (int, optional): Input seed. Defaults to 1000.
    """
    if seed:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cuda_available = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if cuda_available:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        pass


def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _, _ in tqdm(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean, std


def start_logging(args):
    model_name, date, dataset, seed = args['model'], args['date'], args['dataset'], args['seed']
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    model_path = os.path.join("trained_models", model_name)
    logs_path = os.path.join(model_path, "logs")
    log_filename = f"{model_name}_{dataset}_{seed}_{date}.csv"
    filepath = os.path.join(logs_path, log_filename)
    losspath = os.path.join(logs_path, 'loss.npy')

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    if os.path.exists(filepath):
        shutil.copy(filepath, os.path.join(logs_path, f"{model_name}_{dataset}_{seed}_{date}_previous.csv"))
        os.remove(filepath)
    if os.path.exists(losspath):
        shutil.copy(losspath, os.path.join(logs_path, 'loss_old.npy'))
        os.remove(losspath)

    log_df = pd.DataFrame({'epoch': [], 'train_loss': [], 'val_loss': [], 'MAE': [], 'RMSE': [], 'MRE': []})
    log_df.to_csv(filepath, index=False)


def get_epoch_log(epoch, train_loss, val_loss, mae, rmse, mre):
    log_dict = {'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'MAE': mae,
                'RMSE': rmse,
                'MRE': mre}
    return log_dict


def save_log(args, epoch, train_loss, val_loss, mae, rmse, mre):
    log_dict = get_epoch_log(epoch, train_loss, val_loss,
                             mae.detach().item(),
                             rmse.detach().item(),
                             mre.detach().item())

    model_name, date, dataset, seed = args['model'], args['date'], args['dataset'], args['seed']

    logs_path = os.path.join("trained_models", model_name, "logs")
    log_filename = f"{model_name}_{dataset}_{seed}_{date}.csv"
    filepath = os.path.join(logs_path, log_filename)

    log_df = pd.read_csv(filepath)
    new_log = pd.DataFrame(log_dict, index=[0])
    log_df = pd.concat([log_df, new_log], ignore_index=True)
    log_df.to_csv(filepath, index=False)
    log_df.to_csv(os.path.join(logs_path, "last_log.csv"), index=False)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
