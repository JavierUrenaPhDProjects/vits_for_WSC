# Script name:      models_utils.py
# Description:      This python file define all functions used to manage the pytorch models, in regards of: loading and
#                   saving checkpoints, performance evaluation, and readability of performance summaries created after
#                   testing.
#
# Author:           Javier Ureña Santiago
# Date of creation: 19/01/2023
########################################################################################################################

from models.ViTs.DeepViT import *
from models.ViTs.ParallelViT import *
from models.ViTs.ViT import *
from models.ViTs.XCiT import *
from models.ViTs.CrossViT import *
from models.resnet import *
from models.CNN import *
from models.TransCrowd import *

import os
import datetime
from tqdm import tqdm


def loadModel(model_name, args):
    model = eval(model_name + f'({args})')
    if args['pretrain']:
        if model_name in args['model_checkpoint']:
            ckpt_file = args['model_checkpoint']
        else:
            ckpt_file = f'last_trained_{model_name}_{args["dataset"]}.pth'

        print(f'Loading pre-trained model of {model_name}. Checkpoint: {ckpt_file}')

        try:
            checkpoint = torch.load(
                f'./trained_models/{model_name}/{ckpt_file}',
                map_location=torch.device(args['device']))
            model.load_state_dict(checkpoint, strict=False)
            model.to(args['device'])
            print(f"Model {model_name} loaded")
        except:
            print(f'File {ckpt_file} not found')
            print(f'Checkpoint for model: {model_name}, trained in dataset: {args["dataset"]} not found!')
            print('The model will be loaded FROM SCRATCH')

    model.to(args['dtype'])
    print(f'Size of the architecture: {sum(p.numel() for p in model.parameters())} parameters')

    model.to(args['device'])
    print("The model will be running on", args['device'], "device")

    return model


def saveModel(model, args):
    print('_____New best model encountered! saving checkpoint_____')
    hostname = os.uname()[1]
    today = datetime.date.today().strftime("%d-%m-%Y")
    name = args['model']
    dataset = args['dataset']
    rand_seed = args['seed']
    if not os.path.exists(f'trained_models/{name}'):
        os.mkdir(f'trained_models/{name}')

    path = f"trained_models/{name}"
    torch.save(model.state_dict(), f"{path}/{name}_{dataset}_{rand_seed}_{hostname}_{today}.pth")
    torch.save(model.state_dict(), f"{path}/last_trained_{name}_{dataset}.pth")


def evaluation(model, val_loader, loss_fn, weights=None):
    """
    Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    and Mean Relative Error (MRE) for a given model and validation data loader.
    Computes a weighted sum of these errors to determine the overall model performance.

    :param model: The model to evaluate
    :param val_loader: DataLoader for the validation dataset
    :param weights: A dictionary containing the weights for MAE, RMSE, and MRE
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    N = len(val_loader.dataset)
    ae, se, re = 0.0, 0.0, 0.0
    total_val_loss = 0.0

    if weights is None:
        """Mean Relative Error is removed from the equation with a weight of 0 (breaks too much)"""
        weights = {'mae': 0.5, 'rmse': 0.5, 'mre': 0}

    with torch.no_grad():
        for images, gts, _ in tqdm(val_loader, colour='red'):
            images, gts = images.to(device), gts.to(device)
            preds = model(images)
            ae += torch.sum(torch.abs(gts - preds))
            se += torch.sum(torch.square(gts - preds))
            re += torch.sum(torch.abs(gts - preds) / (gts + 1e-6))  # Adding a small constant to avoid division by zero
            total_val_loss += loss_fn(preds, gts).item() * images.size(0)
    mae = ae / N
    rmse = torch.sqrt(se / N)
    mre = re / N
    val_loss = total_val_loss / N

    total_error = weights['mae'] * mae + weights['rmse'] * rmse + weights['mre'] * mre

    return mae, rmse, mre, total_error, val_loss


def read_performance_summaries(datasets,
                               test_results_dir='/home/user/Pycharm/vits_for_WSC/tests/test_results',
                               file='performance.csv'):
    results_csvs = {}
    for dir in os.listdir(test_results_dir):
        if dir in datasets:
            if file in os.listdir(os.path.join(test_results_dir, dir)):
                results_csvs[dir] = os.path.join(test_results_dir, dir, file)
    return results_csvs


def get_best_models(df, model_list, args):
    models_dict = {}
    args['pretrain'] = True

    for model_name in model_list:
        model_df = df[df['model_name'] == model_name]
        best_ckpt = model_df[model_df.tot_err == model_df.tot_err.min()].ckpt.values[0]
        print(f'Best checkpoint for model {model_name}: {best_ckpt}')
        args['model_checkpoint'] = best_ckpt
        model = loadModel(model_name, args)
        models_dict[model_name] = model

    return models_dict
