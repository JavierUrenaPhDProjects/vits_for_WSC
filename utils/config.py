########################################################################################################################
# Script name:      config.py
# Description:      This python file initializes and configures hyperparameters and dataset folders. It is prepared to
#                   be executed in local and remote machine.
#                   Note: Make sure that configurations here attend to your system preferences
#
# Author:           Javier Ureña Santiago
# Date of creation: 19/01/2023
########################################################################################################################

import argparse
import os
import torch
import logging
import nni
from nni.utils import merge_parameter
import datetime


def str2bool(v):
    # Función para interpretar palabras booleanas en consola
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def select_dataset(dataset):
    """
    This function selects the appropiate dataset from the DATASETS folder you might have created.
    :param dataset: string. defines the label you would use to refer to the dataset
    :return: string. the actual folder name
    """
    if dataset == 'yellow_cells':
        folder = 'cell_counting_yellow'
    elif dataset == 'LTCO':
        folder = 'LTCO_cells'
    elif dataset == 'desdet':
        folder = 'DesDet'
    elif dataset == 'cancer':
        folder = 'U2OS_HL60'
    return folder


def select_dtype(data_type):
    if data_type == 'float32':
        dtype = torch.float32
    elif data_type == 'float64' or data_type == 'double':
        dtype = torch.float64
    return dtype


parser = argparse.ArgumentParser('Training arguments')

parser.add_argument('--dataset', default='cancer', type=str)
dataset = parser.parse_known_args()[0].dataset
parser.add_argument('--seed', default=1000, type=int)

# Here you must define the node name of the computers you will run the program in, and the path to the
# folder you store your datasets
local_name = 'localhost'
remote_name = 'remotehost'
local_path = '/home/user/DATASETS/'
remote_path = '/data/DATASETS/'
if local_name in os.uname()[1]:
    dataset_path = os.path.join(local_path, select_dataset(dataset))
    parser.add_argument('--device', default='cpu', type=str)

elif remote_name in os.uname()[1]:
    dataset_path = os.path.join(remote_path, select_dataset(dataset))
    parser.add_argument('--device', default='cuda:0', type=str)

# DATASET PARAMETERS #
parser.add_argument('--dataset_path', default=dataset_path, type=str)
parser.add_argument('--train_path', default=os.path.join(dataset_path, 'train_val'), type=str)
parser.add_argument('--test_path', default=os.path.join(dataset_path, 'test'), type=str)
parser.add_argument('--data_type', default='float32', type=str)
dtype = parser.parse_known_args()[0].data_type
parser.add_argument('--dtype', default=select_dtype(dtype))
parser.add_argument('--channels', default=1 if dataset == 'cancer' else 3)

# TRAINING PARAMETERS #
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--train_percnt', default=0.8, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr_scheduler', default=True, type=str2bool)
parser.add_argument('--date', default=datetime.date.today().strftime("%d-%m-%Y"), type=str)
parser.add_argument('--warmup', default=False, type=str2bool)
parser.add_argument('--warmup_steps', default=5000, type=int)

# MODEL PARAMETERS #
parser.add_argument('--trained_models_path', default='trained_models/', type=str)
parser.add_argument('--model_checkpoint', default='', type=str)
parser.add_argument('--model', default='ResNet50', type=str)
parser.add_argument('--img_size', default=384, type=int)
parser.add_argument('--pretrain', default=False, type=str2bool)

# TESTING PARAMETERS #
# ['ResNet34', 'ResNet50', 'ResNet101', 'transcrowd_gap', 'transcrowd_token', 'vit_base'],
parser.add_argument('--test_models', nargs='+',
                    default=['simple_vit'],
                    help='List of models desired to be tested')
parser.add_argument('--cross_dataset_val', default=False, type=str2bool)

# args = parser.parse_known_args()[0]
args = parser.parse_args()
return_args = parser.parse_args()


def load_args(logger_name):
    logger = logging.getLogger(logger_name)
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    args = vars(merge_parameter(return_args, tuner_params))

    return args
