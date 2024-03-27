import sys
sys.path.append('C:/code/coldstart_bglp/lstm_pop')
sys.path.append('C:/code/coldstart_bglp')
from train import *
import xlwt

# dataset_list = ['replace-bg', 'ohio', 'arises', 'ctr3_cgm_only', 'abc4d']
# dataset_list = ['ohio', 'arises']
# dataset_list = [ 'ohio', 'abc4d', 'ctr3_cgm_only', 'replace-bg']
dataset_list = [ 'ohio', 'abc4d', 'ctr3_cgm_only', 'replace-bg']

version = 'coldstart_fl'
data_proc = None
for seed in [1, 2, 3, 4]:  
    
    if seed == 1:
        experiment_version = 'exp_5' 
    else:
        experiment_version = 'exp_5' + '_seed_' +str(seed) # ph 30
        
    pred_window = 6

    # experiment_version = 'exp_1_ph_60'
    # pred_window = 12
    CONF = {
        'dataset_list': dataset_list,
        'data_paths':{dataset:f'C:/code_data/{dataset}/{version}/' for dataset in dataset_list},
        'log_paths': {dataset:f'D:/code_log/{version}/{experiment_version}/{dataset}' for dataset in dataset_list},
        'model_name': 'lstm',
        'comments': '',
        'hidden_dim': 256, 
        'n_prev' : 24,
        'pred_window' : pred_window,
        'seq_len': 12,

        'attri_list': [],

        'time_attri_list' : [],  

        'device': 'cuda:0',
        'weight_decay': 1e-4,
        'batch_size': 128,

        'device': 'cuda',

        'interval' : 5,
        
        'tuning_iter': 30,
        'tuning_local_epochs': 80,
        'tuning_lr' : 1e-5,
        
    }

    print_dataset = {
        'ohio': 'OHIO',
        'replace-bg': 'REPLACE-BG',
        'arises': 'ARISES',
        'ctr3' :'CTR3',
        'ctr3_cgm_only': 'CTR3 (CGM)',
        'abc4d': 'ABC4D',
    }
    print_metric = {
        'rmse' : 'RMSE',
        'mape' : 'MARD',
        'mae' : 'MAE',
        'grmse' : 'gRMSE',
        'time_lag': 'Time Lag',
    }

    population_model = get_model(CONF)
    population_model.to(CONF['device'])

    if data_proc is None:
 
        data_proc = TestDataProcessor(CONF, fine_tuning=True)
    
    test_log = CrossTestLog(CONF)

    update = Update(data_proc, CONF)
    save_dict = {}
    for load_dataset in dataset_list:
    
        name = load_dataset + '_' + CONF['model_name'] + '_' + CONF['comments']
        

        for repeat in range(2):

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(seed)
            metrics_dic = {}
            metrics_dic_after = {}
            
            for pid in data_proc.dataset2test_pid2data_npy[load_dataset].keys():
                if repeat == 0:
                    pid_model = get_model(CONF)
                    pid_model.to(CONF['device'])
                else:
                    pid_model = test_log.load_model(population_model, load_dataset, name)

                pid_model.eval()
                output_dict = update.cross_test_inference(pid_model, pid, load_dataset)
                per_metrics_dic = output_dict['per_metrics_dic']
                metrics_dic = update_metrics_dic(metrics_dic=metrics_dic, per_metrics_dic=per_metrics_dic)


                output_dict = update.inference(pid_model, pid, dataset = load_dataset)
                min_rmse = output_dict['rmse']
                
                best_pid_model = copy.deepcopy(pid_model)
                for tuning_iter in range(CONF['tuning_iter']):
                    pid_model.train()
                    lr = CONF['tuning_lr'] if repeat == 1 else CONF['tuning_lr'] * 100
                    update.update_weights(
                        model=pid_model, local_epochs=CONF['tuning_local_epochs'], pid = pid, lr=lr, dataset = load_dataset)
                    pid_model.eval()

                    output_dict = update.inference(pid_model, pid, dataset = load_dataset)
                    current_rmse = output_dict['rmse']
                    
                    if current_rmse < min_rmse:
                        best_pid_model = copy.deepcopy(pid_model)
                        min_rmse = current_rmse
                sub = '_rand_init_' if repeat == 0 else ''
                test_log.save_model(best_pid_model, load_dataset, name, pid, sub = sub)
                output_dict = update.cross_test_inference(best_pid_model, pid, load_dataset)
                per_metrics_dic = output_dict['per_metrics_dic']
                metrics_dic_after = update_metrics_dic(metrics_dic=metrics_dic_after, per_metrics_dic=per_metrics_dic)
                
            
            if repeat == 1: 
                np.save(f'D:/code_log/{version}/{experiment_version}/fine_tuning_{load_dataset}_before.npy', metrics_dic)
                np.save(f'D:/code_log/{version}/{experiment_version}/fine_tuning_{load_dataset}_after.npy', metrics_dic_after)
            else:
                np.save(f'D:/code_log/{version}/{experiment_version}/fine_tuning_{load_dataset}_pure.npy', metrics_dic_after)
                
            print(repeat, seed, load_dataset, np.mean(metrics_dic['rmse']), np.mean(metrics_dic_after['rmse']))
        
    # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # dataset2sheet = {}
    # sheet = book.add_sheet('Main', cell_overwrite_ok=True)
    # metrics = ['rmse', 'mape', 'mae', 'grmse', 'time_lag']
    # num_mtc = len(metrics)
    # sheet.write(0 , 2 + len(dataset_list), 'ALL')

    # save_dict = {}
    # for r, load_dataset in enumerate(dataset_list):

    #     name = load_dataset + '_' + CONF['model_name'] + '_' + CONF['comments']
    #     population_model = test_log.load_model(population_model, load_dataset, name)
    #     sheet.write(1 + num_mtc * r , 0, print_dataset[load_dataset])
    #     total_metric_dic = {}
    #     unseen_metric_dic = {} # step 1

    #     for c, test_dataset in enumerate(dataset_list):
    #         metrics_dic = {}
    #         pid2prediction = {}
    #         sheet.write(0, 2 + c, print_dataset[test_dataset])

    #         for pid in data_proc.dataset2test_pid2data_npy[test_dataset].keys():
    #             output_dict = update.cross_test_inference(population_model, pid, test_dataset)
    #             per_metrics_dic = output_dict['per_metrics_dic']
    #             metrics_dic = update_metrics_dic(metrics_dic=metrics_dic, per_metrics_dic=per_metrics_dic)
    #             total_metric_dic = update_metrics_dic(metrics_dic=total_metric_dic,  per_metrics_dic=per_metrics_dic)
                
    #             # step 2
    #             if test_dataset != load_dataset:
    #                 unseen_metric_dic = update_metrics_dic(metrics_dic=unseen_metric_dic,  per_metrics_dic=per_metrics_dic)
    #             # end
                   
    #             pid2prediction[pid] = output_dict['pred']
    #         print(f'{load_dataset=},{test_dataset=}')
    #         print_metrics_dic(metrics_dic)
    #         print_str = ""
            
    #         for n, metric in enumerate(metrics):
    #             print_str = f'{np.mean(metrics_dic[metric]):.2f}({np.std(metrics_dic[metric]):.2f})'
    #             sheet.write(1 + num_mtc * r + n, 2 + c, print_str)
    #             sheet.write(1 + num_mtc * r + n, 1, print_metric[metric])
    #             save_dict[(load_dataset, test_dataset, metric)] = (np.mean(metrics_dic[metric]), np.std(metrics_dic[metric]))

    #     for n, metric in enumerate(metrics):
    #         print_str = f'{np.mean(total_metric_dic[metric]):.2f}({np.std(total_metric_dic[metric]):.2f})'
    #         sheet.write(1 + num_mtc * r + n, 2 + len(dataset_list), print_str)
    #         save_dict[(load_dataset, metric)] = (np.mean(total_metric_dic[metric]), np.std(total_metric_dic[metric]))
            
    #         # step 3
    #         save_dict[('unseen_'+load_dataset, metric)] = (np.mean(unseen_metric_dic[metric]), np.std(unseen_metric_dic[metric]))
            

    # np.save(f'D:/code_log/{version}/{experiment_version}/{experiment_version}.npy', save_dict)
    # print(save_dict)

    # import os
    # if not os.path.exists(f'./{experiment_version}'):
    #     experiment_version_pre = '_'.join(experiment_version.split('_')[:2])
    #     book.save(f'{experiment_version_pre}/{experiment_version}.xls')
    # else:
    #     book.save(f'{experiment_version}/{experiment_version}.xls')