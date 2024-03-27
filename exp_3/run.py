import sys
sys.path.append('C:/code/coldstart_bglp/lstm_pop')
sys.path.append('C:/code/coldstart_bglp')
from train import *



import random
import numpy as np
import torch
import os

# dataset = 'replace-bg'

# dataset = 'arises'
dataset_list = [ 'ohio', 'abc4d', 'ctr3_cgm_only', 'replace-bg']
for seed in [1, 2, 3, 4]:
    for dataset in dataset_list:
        # random_valid = True if dataset == 'replace-bg' else False
        random_valid = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        version = 'coldstart_fl'
        experiment_version = 'exp_3' + '_seed_' +str(seed) if seed != 1 else 'exp_3' # ph 30
        pred_window = 6
        
        # experiment_version = 'exp_3_ph_60'
        # pred_window = 12
        CONF = {
            'data_path':f'C:/code_data/{dataset}/{version}/',
            'log_path': f'D:/code_log/{version}/{experiment_version}/{dataset}',
            'comments': '',

            'meta': 'MAML',
            # 'meta': 'MetaSGD',
            
            'dataset': dataset,
            'n_prev' : 24,
            'pred_window' : pred_window,
            'seq_len': 12,

            'attri_list': [],

            'time_attri_list' : [],  

            'model_name':'lstm', 

            'hidden_dim': 256,

            'lr': 1e-3,  # 1e-4\n",
            'fast_lr': 1e-1, 
            'adaptation_steps': 1,
            'adapt_eval_ratio': 0.5, 


            # 'epochs': 80,
            # 'epochs' : 15000,
            # 'meta_batch_size': 32,
            
            'epochs' : 20000,
            'meta_batch_size': 1,
            'batch_size': 128,
            'print_every':1000,
            'start_print': 5000,

            # 'meta_batch_size': 4,
            # 'batch_size': 128,

            # 'epochs': 2,
            # 'local_epochs': 2,

            'weight_decay': 1e-4,


            'device': 'cuda',

            
            'random_valid': random_valid,
            'random_loop' : 10,
            
            'save_model': True,

            'interval': 5,

        }
        train_maml(CONF)



