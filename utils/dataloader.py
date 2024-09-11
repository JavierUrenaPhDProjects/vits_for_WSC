########################################################################################################################
# Script name:      dataloader.py
# Description:      This python file creates the dataset class. The CRCDataset is a compilation of cell images with
#                   the respective cell coordinates. This file allows to create a dataset from it, that can be later
#                   be made into dataloaders for the training and testing.
#
# Author:           Javier Ureña Santiago
# Date of creation: 19/01/2023
########################################################################################################################
import os.path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


def create_dataset(args, test=False):
    if test:
        path = args['test_path']
        file = 'test_files.csv'
    else:
        path = args['train_path']
        file = 'train_files.csv'

    datasetName = args['dataset']
    img_size = (args['img_size'], args['img_size'])
    dtype = args['dtype']

    if datasetName == 'yellow_cells':
        dataset = YellowCellsDataset(datasetpath=path, datafile=file, img_size=img_size, dtype=dtype)
    elif datasetName == 'LTCO':
        dataset = LTCODataset(datasetpath=path, datafile=file, img_size=img_size, dtype=dtype)
    elif datasetName == 'desdet':
        dataset = DesDet_artificial(datasetpath=path, datafile=file, img_size=img_size, dtype=dtype)
    elif datasetName == 'cancer':
        dataset = CancerCells(datasetpath=path, datafile=file, img_size=img_size, dtype=dtype)

    return dataset


class YellowCellsDataset(Dataset):
    def __init__(self, datasetpath='/home/user/DATASETS/cell_counting_yellow/train_val/',
                 datafile='train_files.csv', img_size=(512, 512), dtype=torch.float32):
        self.datasetpath = datasetpath
        self.imagespath = os.path.join(datasetpath, 'images')
        self.img_size = img_size
        self.dataFrame = pd.read_csv(os.path.join(datasetpath, datafile))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(img_size),
             transforms.Normalize(mean=[0.0507577807, 0.0267739408, 0.0015457729],
                                  std=[0.0798312202, 0.0611209497, 0.0061732889]),
             transforms.ConvertImageDtype(dtype=dtype)
             ])
        self.dtype = dtype

    def __len__(self):
        return len(self.dataFrame)

    def load_image(self, imgpath):
        with Image.open(imgpath).convert('RGB') as img:
            og = np.asarray(img.resize(self.img_size), dtype=np.uint8)
            # img = np.asarray(img)
            x = self.transform(img)
            return x, og

    def __getitem__(self, idx):
        imgFile, count, coords = self.dataFrame.loc[idx]
        imgpath = os.path.join(self.imagespath, imgFile)
        x, og = self.load_image(imgpath)
        y = torch.tensor(count).unsqueeze(0)

        return x, y, og


class LTCODataset(Dataset):
    def __init__(self, datasetpath='/home/user/DATASETS/LTCO_cells/train_val/', datafile='train_files.csv',
                 img_size=(384, 384),
                 dtype=torch.float32):
        self.datasetpath = datasetpath
        self.imagespath = os.path.join(datasetpath, 'images')
        self.img_size = img_size
        self.dataFrame = pd.read_csv(os.path.join(datasetpath, datafile))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(img_size),
             transforms.Normalize(mean=[0.0195658896, 0.0196784139, 0.3616961241],
                                  std=[0.0424158573, 0.0424043387, 0.2340543121]),
             transforms.ConvertImageDtype(dtype=dtype)
             ])
        self.dtype = dtype

    def __len__(self):
        return len(self.dataFrame)

    def load_image(self, imgpath):
        with Image.open(imgpath).convert('RGB') as img:
            og = np.asarray(img.resize(self.img_size), dtype=np.uint8)
            # img = np.asarray(img)
            x = self.transform(img)
            return x, og

    def __getitem__(self, idx):
        imgFile, count, coords = self.dataFrame.loc[idx]
        imgpath = os.path.join(self.imagespath, imgFile)
        x, og = self.load_image(imgpath)
        y = torch.tensor(count).unsqueeze(0)
        return x, y, og


class DesDet_artificial(Dataset):
    def __init__(self, datasetpath='/home/user/DATASETS/DesDet/train_val/',
                 datafile='train_files.csv',
                 img_size=(256, 256),
                 dtype=torch.float32):
        self.datasetpath = datasetpath
        # 'images' for original size, 'images_reduced' for images 384x384
        self.imagespath = os.path.join(datasetpath, 'images_reduced')
        self.datafile = os.path.join(datasetpath, datafile)
        self.img_size = img_size
        self.dataFrame = pd.read_csv(self.datafile)
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(img_size),
             transforms.Normalize(mean=[0.3766, 0.5191, 0.2525],
                                  std=[0.0869, 0.0736, 0.0833]),
             transforms.ConvertImageDtype(dtype=dtype)
             ])
        self.dtype = dtype

    def __len__(self):
        return len(self.dataFrame)

    def load_image(self, imgpath):
        with Image.open(imgpath).convert('RGB') as img:
            og = np.asarray(img.resize(self.img_size), dtype=np.uint8)
            # img = np.asarray(img)
            x = self.transform(img)
            return x, og

    def __getitem__(self, idx):
        imgFile, count, coords = self.dataFrame.loc[idx]
        imgpath = os.path.join(self.imagespath, imgFile)
        x, og = self.load_image(imgpath)
        y = torch.tensor(count).unsqueeze(0)
        return x, y, og


class CancerCells(Dataset):
    """
    This is a dataset that contains microscope images from two cell lines, namely, a human osteosarcoma
    cell line (U2OS) and a human leukemia cell line (HL-60). The dataset was originally prepared for the
    cell counting task. It contains 165 labeled images (training: 133, test: 32).
    """

    def __init__(self, datasetpath='/home/user/DATASETS/U2OS_HL60/train_val/',
                 datafile='train_files.csv',
                 img_size=(384, 384),
                 dtype=torch.float32):
        self.datasetpath = datasetpath
        self.imagespath = os.path.join(datasetpath, 'images')
        self.datafile = os.path.join(datasetpath, datafile)
        self.img_size = img_size
        self.dataFrame = pd.read_csv(self.datafile)
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(img_size),
             transforms.Normalize(mean=[0.0281262919],
                                  std=[0.0088649262]),
             transforms.ConvertImageDtype(dtype=dtype)
             ])
        self.dtype = dtype

    def __len__(self):
        return len(self.dataFrame)

    def load_image(self, imgpath):
        with Image.open(imgpath) as img:
            og = np.asarray(img.resize(self.img_size), dtype=np.uint8)
            # img = np.asarray(img)
            x = self.transform(img)
            return x, og

    def __getitem__(self, idx):
        imgFile, count = self.dataFrame.loc[idx]
        imgpath = os.path.join(self.imagespath, imgFile)
        x, og = self.load_image(imgpath)
        y = torch.tensor(count).unsqueeze(0)
        return x, y, og