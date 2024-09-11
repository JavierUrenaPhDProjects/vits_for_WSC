########################################################################################################################
# Script name:      test.py
# Description:      This script loads all desired models to be tested, and loads into them all the available checkpoints
#                   in ./trained_models/ for each model. It will evaluate all the models in a validation set, measuring
#                   MAE, RSE and total evaluation time. The results of the examination are saved into a csv file and a
#                   .txt file showing a text table summarizing all results.
#
# Author:           Javier Ureña Santiago
# Date of creation: 01/06/2023
########################################################################################################################
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os.path
import warnings

warnings.filterwarnings('ignore')
from utils.utils import *
from utils.dataloader import create_dataset
from utils.models_utils import loadModel, evaluation, get_best_models, read_performance_summaries
import pandas as pd
import time
import logging
import nni
from nni.utils import merge_parameter
from utils.config import return_args
from fvcore.nn import FlopCountAnalysis
from tabulate import tabulate

logger = logging.getLogger('testing_regression_model')
tuner_params = nni.get_next_parameter()
logger.debug(tuner_params)
args = vars(merge_parameter(return_args, tuner_params))
set_seed(args['seed'])


class test:
    def __init__(self, args):
        self.args = args
        self.model_list = args['test_models']
        self.dataset_name = args['dataset']
        self.dataset = create_dataset(args, test=True)
        self.val_loader = DataLoader(self.dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0,
                                     pin_memory=True)
        self.dataset_dim = self.dataset.__getitem__(0)[0].shape
        self.results_dir = f'test_results/{self.dataset_name}'
        self.csv_path = os.path.join(self.results_dir, 'performance.csv')
        self.txt_path = os.path.join(self.results_dir, 'performance.txt')
        self.df = pd.DataFrame()

    def print_info(self):
        print('\n---------------------\nDataset information\n---------------------')
        print(f'Dataset used: {self.dataset_name}'
              f'\nDataset size: {self.dataset.__len__()}'
              f'\nData dimensions: {self.dataset_dim}')

    def check_dir(self):
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

    def save_results(self):
        self.df.to_csv(self.csv_path, index=False)
        if os.path.exists(self.txt_path):
            print('\nRemoving previous summary file')
            os.remove(self.txt_path)

        with open(self.txt_path, 'a') as f:
            df_string = self.df.to_string()
            f.write(df_string)

    def test_model(self, model):
        t0 = time.time()
        mae, rmse, mre, tot = evaluation(model, self.val_loader)
        tf = time.time() - t0
        return mae.item(), rmse.item(), mre.item(), tot.item(), tf

    def run(self):
        self.print_info()
        self.check_dir()
        dic = {}
        self.args['pretrain'] = True
        for model_name in self.model_list:
            dic['model_name'] = model_name
            ckpts_path = f'../trained_models/{model_name}'
            ckpts = [f for f in os.listdir(ckpts_path) if self.dataset_name in f]
            if ckpts:
                print(f"\n~~~~~~~~Testing architecture: {model_name}~~~~~~~~")
                for ckpt_file in ckpts:
                    self.args['model_checkpoint'] = ckpt_file

                    os.chdir('..')
                    model = loadModel(model_name, self.args)
                    os.chdir('tests')

                    dic['ckpt'] = ckpt_file
                    dic['mae'], dic['rmse'], dic['mre'], dic['tot_err'], dic['eval_time'] = self.test_model(model)
                    print(
                        f"Checkpoint: {ckpt_file} | MAE = {dic['mae']} | MRE = {dic['mre']} | RMSE = {dic['rmse']} |"
                        f" Total Error = {dic['tot_err']} | Evaluation time: {dic['eval_time']}")
                    self.df = pd.concat([self.df, pd.DataFrame(dic, index=[0])], ignore_index=True)
            else:
                print(f"No checkpoints of model {model_name} found for dataset {self.dataset_name}.")

        self.save_results()


class TestAfterTrain:
    def __init__(self, args, save_log=True):
        self.args = args
        self.save_log = save_log
        self.dataset = create_dataset(args, test=True)
        self.test_loader = DataLoader(self.dataset, batch_size=1,
                                      shuffle=False,
                                      num_workers=os.cpu_count(),
                                      pin_memory=True)

        self.results_file = f'{args["model"]}_{args["dataset"]}_{args["seed"]}_{args["date"]}.csv'
        if not os.path.exists('tested_models'):
            os.mkdir('tested_models')
        self.results_dir = self.generate_testlog()

    def generate_testlog(self):
        results_dir = f'tested_models/{self.args["model"]}'
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        return results_dir

    def measure_FLOPS(self, model):
        dummy_input = next(iter(self.test_loader))[0].to(next(model.parameters()).device)
        flops = FlopCountAnalysis(model, dummy_input).total()
        return flops

    def run(self, model, loss_fn):
        mae, rmse, mre, tot_err, loss = evaluation(model, self.test_loader, loss_fn)
        flops = self.measure_FLOPS(model)
        test_dict = {'MAE': mae.detach().item(), 'RMSE': rmse.detach().item(),
                     'MRE': mre.detach().item(), 'Total_Err': tot_err.detach().item(),
                     'Test_Loss': loss, 'FLOPS': flops}
        test_df = pd.DataFrame(test_dict, index=[0])
        print(tabulate(test_df, headers='keys', tablefmt='psql', showindex=False))
        if self.save_log:
            test_df.to_csv(os.path.join(self.results_dir, self.results_file), index=False)
        return test_df


if __name__ == "__main__":
    test = test(args)
    test.run()
