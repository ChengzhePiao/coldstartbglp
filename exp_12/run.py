import sys
sys.path.append('C:/code/coldstart_bglp/lstm_pop')
sys.path.append('C:/code/coldstart_bglp')
from train import *



import random
import numpy as np
import torch

dataset_list = ['ohio','abc4d', 'ctr3_cgm_only', 'replace-bg', ]

for seed in [2, 3, 4]:
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
        
        experiment_version = 'exp_12'  + '_seed_' +str(seed)# ph 30
        pred_window = 6
        
        # experiment_version = 'exp_12_ph_60'
        # pred_window = 12
        
        CONF = {
            
            'data_path':f'C:/code_data/{dataset}/{version}/',
            'log_path': f'D:/code_log/{version}/{experiment_version}/{dataset}',
            'comments': '',



            'dataset': dataset,
            'n_prev' : 24,
            
            # 'pred_window' : 6,
            'pred_window': pred_window,
            
            'seq_len': 12,

            'attri_list': [],

            'time_attri_list' : [],  

            'model_name':'xgboost', 

            # 'epochs': 80,
            'epochs' : 30,
            'local_epochs': 500,


            # 'epochs': 2,
            # 'local_epochs': 2,

            'weight_decay': 1e-4,
            'batch_size': 128,

            'device': 'cuda',

            'print_every':2,
            'start_print': 10,
            
            'random_valid': random_valid,
            'random_loop' : 10,
            
            'save_model': True,

            'interval': 5,

        }
        train_scikit_learn(CONF, seed)
