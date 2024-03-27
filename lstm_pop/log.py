from util import *

class Log():
    def __init__(self, name, CONF):
        self.root_dir = os.path.join(CONF['log_path'], name)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.global_epoch = []
        self.tune_epoch = []
        self.CONF = CONF

        self.epoch_vari_importance = []
        self.epoch_part_vari_importance = []
        self.epoch_indexs = []
    
        self.eval_rmse_list = []
        
    def save_prediction(self, pid2prediction, ):
        
        save_path = os.path.join(self.root_dir, 'pid2prediction.npy')
        np.save(save_path, pid2prediction)

    def save_metrics_dic(self, metrics_dic):
        save_path = os.path.join(self.root_dir, 'pid2metrics.npy')
        np.save(save_path, metrics_dic)

    def save_models(self, pid2model):
        if type(pid2model) is dict:

            for pid in pid2model:
                save_path = os.path.join(self.root_dir, f'{pid}_model.pth')
                torch.save(pid2model[pid].state_dict(), save_path)
        else:
            save_path = os.path.join(self.root_dir, f'population_model.pth')
            torch.save(pid2model.state_dict(), save_path)

    def save_scikit_models(self, pid2model):
        if type(pid2model) is dict:
            for pid in pid2model:
                save_path = os.path.join(self.root_dir, f'{pid}_model.joblib')
                dump(pid2model[pid], save_path) 
        else:
            save_path = os.path.join(self.root_dir, f'population_model.joblib')
            dump(pid2model, save_path) 
            print(save_path)
    
    def save_xgboost_models(self, pid2model):
        if type(pid2model) is dict:
            for pid in pid2model:
                save_path = os.path.join(self.root_dir, f'{pid}_model.json')
                pid2model[pid].save_model(save_path)
        else:
            save_path = os.path.join(self.root_dir, f'population_model.json')
            pid2model.save_model(save_path)
            print(save_path)

    def load_xgboost_models(self, pid2model):
        path = self.root_dir[:-14]
        for pid in pid2model:
            load_path = os.path.join(path, f'{pid}_model.json')
            pid2model[pid].load_model(load_path) 
        return pid2model
    
    def load_models(self, pid2model, CONF, pid=None):


        if pid is not None:
            path = self.root_dir
            load_path = os.path.join(path, f'{pid}_model.pth')
            pid2model.load_state_dict(torch.load(load_path))
            pid2model.eval()
            return pid2model
        for pid in pid2model:
            path = self.root_dir[:-14]
            load_path = os.path.join(path, f'{pid}_model.pth')
            pid2model[pid].load_state_dict(torch.load(load_path))
            pid2model[pid].eval()
        return pid2model
    def save_rmse(self, pid2rmse):
        save_path = os.path.join(self.root_dir, 'pid2rmse.npy')
        np.save(save_path, pid2rmse)
    
    def save_mape(self, pid2mape):
        save_path = os.path.join(self.root_dir, 'pid2mape.npy')
        np.save(save_path, pid2mape)

    def save_mae(self, pid2mae):
        save_path = os.path.join(self.root_dir, 'pid2mae.npy')
        np.save(save_path, pid2mae)

    def save_attention(self, attention):
        save_path = os.path.join(self.root_dir, f'{self.current_pid}_{self.t}_attention.npy')
        np.save(save_path, attention)

    def set_pid(self, pid):
        self.current_pid = pid

    def set_time(self, t):
        self.t = t

    def save_global_epochs(self, epoch, rmse, time):
        save_path = os.path.join(self.root_dir, 'global_epoch.npy')

        self.global_epoch.append([epoch, rmse, time])

        np.save(save_path, np.array(self.global_epoch))

    def save_tune_epochs(self, epoch, rmse, time):
        save_path = os.path.join(self.root_dir, 'tune_epoch.npy')

        self.tune_epoch.append([epoch, rmse, time])

        np.save(save_path, np.array(self.tune_epoch))


    def record_eval_rmse(self, rmse, epoch_index):
        self.eval_rmse_list.append(rmse)
        self.epoch_indexs.append(epoch_index)

    def save_epoch_rmse(self):
        save_path = os.path.join(self.root_dir, 'epoch_rmse.pdf')
        save_data_path_1 = os.path.join(self.root_dir, 'epoch_rmse.npy')
        save_data_path_2 = os.path.join(self.root_dir, 'epoch_rmse_idx.npy')
        fig, ax = plt.subplots(figsize = (20, 10))
        np.save(save_data_path_1, np.array(self.eval_rmse_list))
        np.save(save_data_path_2, self.epoch_indexs)
        
        eval_rmse_list = [np.mean(_) for _ in self.eval_rmse_list]
        ax.plot(self.epoch_indexs, eval_rmse_list, )
        ax.set_ylabel('RMSE (mg/dL)', fontsize=24)
        ax.set_xlabel('Epoch', fontsize=24)   # relative to plt.rcParams['font.size']
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        fig.tight_layout()
        plt.savefig(save_path)
        plt.clf()


        
class CrossTestLog():
    def __init__(self, CONF):
        self.root_dir = CONF['log_paths']


        self.CONF = CONF

    def load_model(self, model, dataset, name):
        if type(model) is not dict:
            path = self.root_dir[dataset]
            load_path = os.path.join(path, name, f'population_model.pth')
            model.load_state_dict(torch.load(load_path))
            model.eval()
        return model
    
    def save_model(self, model, dataset, name, pid, sub=''):
        path = self.root_dir[dataset]
        save_path = os.path.join(path, name, f'{pid}_model' + sub + '.pth')
        torch.save(model.state_dict(), save_path)
        
        
    def load_scikit_model(self, model, dataset, name):

        if type(model) is not dict:
            path = self.root_dir[dataset]
            load_path = os.path.join(path, name, f'population_model.joblib')
            model = load(load_path)
        
        return model
    
    def load_xgboost_model(self, model, dataset, name):

        if type(model) is not dict:
            path = self.root_dir[dataset]
            load_path = os.path.join(path, name, f'population_model.json')
            model.load_model(load_path)
        
        return model
    