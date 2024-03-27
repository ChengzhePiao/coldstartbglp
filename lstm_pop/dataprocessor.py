from util import *

class PopulationDataProcessor():
    def __init__(self, CONF):
        self.CONF = CONF
        n_prev = CONF['n_prev']
        pred_window = CONF['pred_window']
        self.train_pid2data_npy = np.load(
            os.path.join(CONF['data_path'], f'train_pid2data_npy_{n_prev}_{pred_window}.npy'), 
            allow_pickle=True
        )[()]
        self.valid_pid2data_npy = np.load(
            os.path.join(CONF['data_path'], f'valid_pid2data_npy_{n_prev}_{pred_window}.npy'), 
            allow_pickle=True
        )[()]
        self.test_pid2data_npy = np.load(
            os.path.join(CONF['data_path'], f'test_pid2data_npy_{n_prev}_{pred_window}.npy'),  
            allow_pickle=True
        )[()]

        if not 'fl' in CONF or not CONF['fl']:
            self.all_train_data = {}
            for pid in self.train_pid2data_npy:
                for content in self.train_pid2data_npy[pid]:
                    if content not in self.all_train_data:
                        self.all_train_data[content] = self.train_pid2data_npy[pid][content]
                    elif content == 'mean' or content == 'std':
                        continue
                    else:

                        self.all_train_data[content] = np.concatenate([self.all_train_data[content],
                        self.train_pid2data_npy[pid][content]], axis=0) 
            
            
            if len(self.all_train_data) > 0:
                del self.train_pid2data_npy

        self.attri2idx = pd.read_pickle(os.path.join(CONF['data_path'], 'attri2idx.pkl'))
        

    def get_train_batch(self, batch_size, device, seq_len, attri_list, time_attri_list, pid=None, all_data = None):

        data = self.all_train_data if pid is None else self.train_pid2data_npy[pid]
        
        idxs = np.random.choice(data['y'].shape[0], batch_size)

        if all_data is not None and all_data:
            idxs = np.arange(data[f'glucose_level_X'].shape[0])

        G = data[f'glucose_level_X'][idxs, -seq_len:]
        
        temp_data = data['attri_X'][idxs, -seq_len:, :] # select batch

        if len(attri_list) == 0:
            attris = None
        else:
            attris = temp_data[:, :, self.attri2idx.loc[attri_list, 'idx'].to_numpy()] # select features
        
        if len(time_attri_list) == 0:
            time_attris = None
        else:
            time_attris = temp_data[:, :, self.attri2idx.loc[time_attri_list, 'idx'].to_numpy()] # select features

        y = data['y'][idxs]

        G = torch.tensor(G, dtype=torch.float32, device=device)
        if attris is not None:
            attris = torch.tensor(attris, dtype=torch.float32, device=device) 
        if time_attris is not None:
            time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)
        y = torch.tensor(np.expand_dims(y, axis=1), dtype=torch.float32, device=device)

        target_time_list = []
        if len(time_attri_list) != 0:
            for time_attri in time_attri_list:
                time_name = f'target_{time_attri}'
                time_np = np.expand_dims(data[time_name][idxs], axis=1)
                target_time_list.append(time_np)
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        return {'glucose': G, 'attris': attris, 'time_attris': time_attris, 'y':y, 'target_time': target_time}
        
    def get_random_val(self, batch_size, device, seq_len, attri_list, time_attri_list, pid):
        data = self.valid_pid2data_npy[pid]
        mean = self.valid_pid2data_npy[pid]['mean']
        std = self.valid_pid2data_npy[pid]['std']
        idxs = np.random.choice(data['y'].shape[0], batch_size)

        G = data[f'glucose_level_X'][idxs, -seq_len:]
        
        temp_data = data['attri_X'][idxs, -seq_len:, :] # select batch
        if len(attri_list) == 0:
            attris = None
        else:
            attris = temp_data[:, :, self.attri2idx.loc[attri_list, 'idx'].to_numpy()] # select features
        
        if len(time_attri_list) == 0:
            time_attris = None
        else:
            time_attris = temp_data[:, :, self.attri2idx.loc[time_attri_list, 'idx'].to_numpy()] # select features

        y = data['y'][idxs]
        y = y * std + mean
        
        G = torch.tensor(G, dtype=torch.float32, device=device)
        if attris is not None:
            attris = torch.tensor(attris, dtype=torch.float32, device=device)
        if time_attris is not None:
            time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)


        if len(time_attri_list) != 0:
            target_time_list = []
            for time_attri in time_attri_list:
                time_name = f'target_{time_attri}'
                time_np = np.expand_dims(data[time_name][idxs], axis=1)
                target_time_list.append(time_np)
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None
        return {'glucose': G, 'attris': attris, 'time_attris': time_attris, 'y':y, 'target_time': target_time}

    def get_val(self, device, seq_len, attri_list, time_attri_list, pid):
        data = self.valid_pid2data_npy[pid]
        mean = self.valid_pid2data_npy[pid]['mean']
        std = self.valid_pid2data_npy[pid]['std']
        
        G = data[f'glucose_level_X'][:, -seq_len:]
        if len(attri_list) == 0:
            attris = None
        else:
            attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[attri_list, 'idx']]
        if len(time_attri_list) == 0:
            time_attris = None
        else:
            time_attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[time_attri_list, 'idx']]
        y = data['y'] * std + mean

        G = torch.tensor(G, dtype=torch.float32, device=device)
        if attris is not None:
            attris = torch.tensor(attris, dtype=torch.float32, device=device)
        if time_attris is not None:
            time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)

        if len(time_attri_list) != 0:
            target_time_list = []
            for time_attri in time_attri_list:
                time_name = f'target_{time_attri}'
                time_np = np.expand_dims(data[time_name], axis=1)
                target_time_list.append(time_np)
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None
        return {'glucose': G, 'attris': attris, 'time_attris': time_attris, 'y':y, 'target_time': target_time}

    def get_mean_std(self, pid, dataset = None):
        return self.test_pid2data_npy[pid]['mean'],  self.test_pid2data_npy[pid]['std']

    def get_test(self, device, seq_len, attri_list, time_attri_list, pid):
        data = self.test_pid2data_npy[pid]
        mean = self.test_pid2data_npy[pid]['mean']
        std = self.test_pid2data_npy[pid]['std']
        
        G = data[f'glucose_level_X'][:, -seq_len:]
        if len(attri_list) == 0:
            attris = None
        else:
            attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[attri_list, 'idx']]
        if len(time_attri_list) == 0:
            time_attris = None
        else:
            time_attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[time_attri_list, 'idx']]
        
        y = data['y'] * std + mean

        G = torch.tensor(G, dtype=torch.float32, device=device)
        if attris is not None:
            attris = torch.tensor(attris, dtype=torch.float32, device=device)
        if time_attris is not None:
            time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)

        if len(time_attri_list) != 0:
            target_time_list = []
            for time_attri in time_attri_list:
                time_name = f'target_{time_attri}'
                time_np = np.expand_dims(data[time_name], axis=1)
                target_time_list.append(time_np)
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        seq_st_ed = data['seq_st_ed_list']

        return {'glucose': G, 'attris': attris, 'time_attris': time_attris, 'y':y, 'target_time': target_time, 'seq_st_ed': seq_st_ed}


class TestDataProcessor():
    def __init__(self, CONF, fine_tuning=False):
        self.CONF = CONF
        n_prev = CONF['n_prev']
        pred_window = CONF['pred_window']
        self.dataset2test_pid2data_npy = {}
        self.dataset2attri2idx = {}
        for dataset in CONF['dataset_list']:
            self.dataset2test_pid2data_npy[dataset] = np.load(
                os.path.join(CONF['data_paths'][dataset], f'test_pid2data_npy_{n_prev}_{pred_window}.npy'),  
                allow_pickle=True
            )[()]

            self.dataset2attri2idx[dataset] = pd.read_pickle(os.path.join(CONF['data_paths'][dataset], 'attri2idx.pkl'))
        if fine_tuning:
            self.dataset2train_pid2data_npy = {}
            self.dataset2valid_pid2data_npy = {}
            for dataset in CONF['dataset_list']:
                self.dataset2train_pid2data_npy[dataset] = np.load(
                    os.path.join(CONF['data_paths'][dataset], f'train_pid2data_npy_{n_prev}_{pred_window}.npy'), 
                    allow_pickle=True
                )[()]
    
                self.dataset2valid_pid2data_npy[dataset] = np.load(
                    os.path.join(CONF['data_paths'][dataset], f'valid_pid2data_npy_{n_prev}_{pred_window}.npy'), 
                    allow_pickle=True
                )[()]
                
                
    def get_mean_std(self, pid, dataset):
        return self.dataset2test_pid2data_npy[dataset][pid]['mean'],  self.dataset2test_pid2data_npy[dataset][pid]['std']

    def get_test(self, device, seq_len, attri_list, time_attri_list, pid, dataset):
        data = self.dataset2test_pid2data_npy[dataset][pid]
        mean = self.dataset2test_pid2data_npy[dataset][pid]['mean']
        std = self.dataset2test_pid2data_npy[dataset][pid]['std']
        
        G = data[f'glucose_level_X'][:, -seq_len:]
        if len(attri_list) == 0:
            attris = None
        else:
            attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[attri_list, 'idx']]
        if len(time_attri_list) == 0:
            time_attris = None
        else:
            time_attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[time_attri_list, 'idx']]
        
        y = data['y'] * std + mean

        G = torch.tensor(G, dtype=torch.float32, device=device)
        if attris is not None:
            attris = torch.tensor(attris, dtype=torch.float32, device=device)
        if time_attris is not None:
            time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)

        if len(time_attri_list) != 0:
            target_time_list = []
            for time_attri in time_attri_list:
                time_name = f'target_{time_attri}'
                time_np = np.expand_dims(data[time_name], axis=1)
                target_time_list.append(time_np)
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        seq_st_ed = data['seq_st_ed_list']

        return {'glucose': G, 'attris': attris, 'time_attris': time_attris, 'y':y, 'target_time': target_time, 'seq_st_ed': seq_st_ed}
    
    
    def get_train_batch(self, batch_size, device, seq_len, attri_list, time_attri_list, pid, dataset):

        data = self.dataset2train_pid2data_npy[dataset][pid]
        
        idxs = np.random.choice(data['y'].shape[0], batch_size)

        G = data[f'glucose_level_X'][idxs, -seq_len:]
        
        temp_data = data['attri_X'][idxs, -seq_len:, :] # select batch

        if len(attri_list) == 0:
            attris = None
        else:
            attris = temp_data[:, :, self.attri2idx.loc[attri_list, 'idx'].to_numpy()] # select features
        
        if len(time_attri_list) == 0:
            time_attris = None
        else:
            time_attris = temp_data[:, :, self.attri2idx.loc[time_attri_list, 'idx'].to_numpy()] # select features

        y = data['y'][idxs]

        G = torch.tensor(G, dtype=torch.float32, device=device)
        if attris is not None:
            attris = torch.tensor(attris, dtype=torch.float32, device=device) 
        if time_attris is not None:
            time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)
        y = torch.tensor(np.expand_dims(y, axis=1), dtype=torch.float32, device=device)

        target_time_list = []
        if len(time_attri_list) != 0:
            for time_attri in time_attri_list:
                time_name = f'target_{time_attri}'
                time_np = np.expand_dims(data[time_name][idxs], axis=1)
                target_time_list.append(time_np)
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        return {'glucose': G, 'attris': attris, 'time_attris': time_attris, 'y':y, 'target_time': target_time}
    
    def get_val(self, device, seq_len, attri_list, time_attri_list, pid, dataset):
        data = self.dataset2valid_pid2data_npy[dataset][pid]
        mean = self.dataset2valid_pid2data_npy[dataset][pid]['mean']
        std = self.dataset2valid_pid2data_npy[dataset][pid]['std']
        
        G = data[f'glucose_level_X'][:, -seq_len:]
        if len(attri_list) == 0:
            attris = None
        else:
            attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[attri_list, 'idx']]
        if len(time_attri_list) == 0:
            time_attris = None
        else:
            time_attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[time_attri_list, 'idx']]
        y = data['y'] * std + mean

        G = torch.tensor(G, dtype=torch.float32, device=device)
        if attris is not None:
            attris = torch.tensor(attris, dtype=torch.float32, device=device)
        if time_attris is not None:
            time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)

        if len(time_attri_list) != 0:
            target_time_list = []
            for time_attri in time_attri_list:
                time_name = f'target_{time_attri}'
                time_np = np.expand_dims(data[time_name], axis=1)
                target_time_list.append(time_np)
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None
        return {'glucose': G, 'attris': attris, 'time_attris': time_attris, 'y':y, 'target_time': target_time}
