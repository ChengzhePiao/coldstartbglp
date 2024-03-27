
from util import *



def MAPE_LOSS(pred, y):
    return torch.mean(
            torch.abs((y-pred))/(y+1e-6)
        )
class Update():
    def __init__(self, data_proc, CONF):

        self.data_proc = data_proc
        self.criterion = nn.MSELoss()
        self.CONF = CONF
        self.optimizer = None

    def update_weights(self, model, local_epochs, pid=None, lr = None, dataset=None):

        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=self.CONF['weight_decay'])

        for iter in range(local_epochs):
            batch_loss = []


            return_dict = self.data_proc.get_train_batch(
            self.CONF['batch_size'], self.CONF['device'], 
            self.CONF['seq_len'], self.CONF['attri_list'], self.CONF['time_attri_list'], pid=pid, dataset=dataset)
            G, attris, time_attris, y = return_dict['glucose'], return_dict['attris'], return_dict['time_attris'], return_dict['y']

            model.zero_grad()

            output_dict = model(G, attris, time_attris)

            predicted_norm = output_dict['pred']

            loss = self.criterion(predicted_norm, y) # + 0.001*MAPE_LOSS(predicted_norm, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        
        epoch_loss = np.mean(batch_loss)
        return model, epoch_loss
    
    def fast_adapt(self, batch, learner, loss, adaptation_steps, ratio):
        G, attris, time_attris, y = batch

        # Separate data into adaptation/evalutation sets
        adaptation_indices = np.zeros(G.size(0), dtype=bool)
        adaptation_indices[np.arange(int(G.size(0) * ratio)) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        
        G_adp, G_evl = G[adaptation_indices], G[evaluation_indices]
        y_adp, y_evl = y[adaptation_indices], y[evaluation_indices]

        if attris is not None:
            attris_adp, attris_evl = attris[adaptation_indices], attris[evaluation_indices]
        else:
            attris_adp, attris_evl = None, None
        
        if time_attris is not None:
            time_attris_adp, time_attris_evl = time_attris[adaptation_indices], time_attris[evaluation_indices]
        else:
            time_attris_adp, time_attris_evl = None, None


        # Adapt the model
        for step in range(adaptation_steps):
            train_error = loss(learner(G_adp, attris_adp, time_attris_adp)['pred'], y_adp)
            learner.adapt(train_error)

        # Evaluate the adapted model
        predictions = learner(G_evl, attris_evl, time_attris_evl)['pred']
        valid_error = loss(predictions, y_evl)
        return valid_error
    
    def update_weights_maml(self, model, meta_batch_size, pid=None, lr = None):

        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=self.CONF['weight_decay'])

        optimizer.zero_grad()

        meta_train_error = 0.0

        for iter in range(meta_batch_size):
            learner = model.clone()
            return_dict = self.data_proc.get_train_batch(
            self.CONF['batch_size'], self.CONF['device'], 
            self.CONF['seq_len'], self.CONF['attri_list'], self.CONF['time_attri_list'], pid=pid)
            G, attris, time_attris, y = return_dict['glucose'], return_dict['attris'], return_dict['time_attris'], return_dict['y']

            evaluation_error = self.fast_adapt(
                batch=(G, attris, time_attris, y),
                learner=learner,
                loss=self.criterion,
                adaptation_steps=self.CONF['adaptation_steps'],
                ratio = self.CONF['adapt_eval_ratio'],
            )

            meta_train_error += evaluation_error.item()
            evaluation_error.backward()

        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        
        optimizer.step()
        
        epoch_loss = evaluation_error / meta_batch_size


        return model, epoch_loss



    def inference(self, model, pid, dataset=None):
        
        mean, std = self.data_proc.get_mean_std(pid, dataset)
        model.eval()
        with torch.no_grad():

            return_dict = self.data_proc.get_val(self.CONF['device'], self.CONF['seq_len'], self.CONF['attri_list'], self.CONF['time_attri_list'], pid, dataset)
            G, attris, time_attris, y = return_dict['glucose'], return_dict['attris'], return_dict['time_attris'], return_dict['y']

            temp_batch = 128
            st = 0
            predicted_norm_list = []
            while True:
                ed = st + temp_batch
                # print(st, ed, G[st:ed].shape[0], G.shape[0], )
                attris_batch = attris[st:ed] if attris is not None else attris
                time_attris_batch = time_attris[st:ed] if time_attris is not None else time_attris
                output_dict = model(G[st:ed], attris_batch, time_attris_batch)
                
                predicted_norm_ = output_dict['pred']
                predicted_norm_list.append(predicted_norm_)
                st = ed
                if ed >= G.shape[0]:
                    break
            predicted_norm = torch.cat(predicted_norm_list, dim=0)

            if predicted_norm.shape[0] != G.shape[0]:
                print('error')

            predicted_norm_np = predicted_norm.cpu().numpy()[:]

            predicted_np = predicted_norm_np * std + mean
            rmse = mean_squared_error(y, predicted_np[:,0])**0.5
            mape = mean_absolute_percentage_error(y, predicted_np[:,0]) * 100
            output_dict = {
                'rmse': rmse,
                'mape': mape,
            }
        return output_dict

    def random_inference(self, model, pid):
        
        mean, std = self.data_proc.get_mean_std(pid)
        model.eval()
        with torch.no_grad():

            
            temp_batch = 128
            predicted_norm_list = []
            y_list = []
            for _ in range(self.CONF['random_loop']):
                return_dict = self.data_proc.get_random_val(temp_batch, self.CONF['device'], self.CONF['seq_len'], self.CONF['attri_list'], self.CONF['time_attri_list'], pid)
                G, attris, time_attris, y = return_dict['glucose'], return_dict['attris'], return_dict['time_attris'], return_dict['y']

                output_dict = model(G, attris, time_attris)
                predicted_norm_ = output_dict['pred']
                predicted_norm_list.append(predicted_norm_)
                y_list.append(y)
            predicted_norm = torch.cat(predicted_norm_list, dim=0)
            y = np.concatenate(y_list, axis=0)
            predicted_norm_np = predicted_norm.cpu().numpy()[:]

            predicted_np = predicted_norm_np * std + mean
            rmse = mean_squared_error(y, predicted_np[:,0])**0.5
            mape = mean_absolute_percentage_error(y, predicted_np[:,0]) * 100
            output_dict = {
                'rmse': rmse,
                'mape': mape,
            }
        return output_dict


    def test_inference(self, model, pid, log=None, print_feature_maps=False):
        mean, std = self.data_proc.get_mean_std(pid)
        model.eval()
        with torch.no_grad():
   
            return_dict = self.data_proc.get_test(self.CONF['device'], self.CONF['seq_len'],  self.CONF['attri_list'], self.CONF['time_attri_list'], pid)
            G, attris, time_attris, y, seq_st_ed = return_dict['glucose'], return_dict['attris'], return_dict['time_attris'], return_dict['y'], return_dict['seq_st_ed']

            # model.to(self.CONF['device'])    
            temp_batch = 128
            st = 0
            predicted_norm_list = []
            while True:
                ed = st + temp_batch
                # print(st, ed, G[st:ed].shape[0], G.shape[0], )
                attris_batch = attris[st:ed] if attris is not None else attris
                time_attris_batch = time_attris[st:ed] if time_attris is not None else time_attris
                output_dict = model(G[st:ed], attris_batch, time_attris_batch)

                predicted_norm_ = output_dict['pred']
                predicted_norm_list.append(predicted_norm_)

                st = ed
                if ed >= G.shape[0]:
                    break
            predicted_norm = torch.cat(predicted_norm_list, dim=0)

            if predicted_norm.shape[0] != G.shape[0]:
                print('error')

            predicted_norm_np = predicted_norm.cpu().numpy()[:]
            predicted_np = predicted_norm_np * std + mean
        
        output_dict = {}
        per_metrics_dic ={}
        per_metrics_dic['rmse'] = mean_squared_error(y, predicted_np[:,0])**0.5
        per_metrics_dic['mape'] = mean_absolute_percentage_error(y, predicted_np[:,0]) * 100
        per_metrics_dic['mae'] = mean_absolute_error(y, predicted_np[:,0])
        per_metrics_dic['grmse'] = cal_gmse(y, predicted_np)
        per_metrics_dic['time_lag'] = cal_time_lag(y, predicted_np, seq_st_ed, self.CONF['interval'])
        output_dict['per_metrics_dic'] = per_metrics_dic
        output_dict['pred'] = predicted_np

        return output_dict
    
    def cross_test_inference(self, model, pid, dataset):
        mean, std = self.data_proc.get_mean_std(pid, dataset)
        model.eval()
        with torch.no_grad():
   
            return_dict = self.data_proc.get_test(self.CONF['device'], self.CONF['seq_len'],  self.CONF['attri_list'], self.CONF['time_attri_list'], pid, dataset)
            G, attris, time_attris, y, seq_st_ed = return_dict['glucose'], return_dict['attris'], return_dict['time_attris'], return_dict['y'], return_dict['seq_st_ed']

            # model.to(self.CONF['device'])    
            temp_batch = 128
            st = 0
            predicted_norm_list = []
            while True:
                ed = st + temp_batch
                # print(st, ed, G[st:ed].shape[0], G.shape[0], )
                attris_batch = attris[st:ed] if attris is not None else attris
                time_attris_batch = time_attris[st:ed] if time_attris is not None else time_attris
                output_dict = model(G[st:ed], attris_batch, time_attris_batch)

                predicted_norm_ = output_dict['pred']
                predicted_norm_list.append(predicted_norm_)

                st = ed
                if ed >= G.shape[0]:
                    break
            predicted_norm = torch.cat(predicted_norm_list, dim=0)

            if predicted_norm.shape[0] != G.shape[0]:
                print('error')

            predicted_norm_np = predicted_norm.cpu().numpy()[:]
            predicted_np = predicted_norm_np * std + mean
        
        output_dict = {}
        per_metrics_dic ={}
        per_metrics_dic['rmse'] = mean_squared_error(y, predicted_np[:,0])**0.5
        per_metrics_dic['mape'] = mean_absolute_percentage_error(y, predicted_np[:,0]) * 100
        per_metrics_dic['mae'] = mean_absolute_error(y, predicted_np[:,0])
        per_metrics_dic['grmse'] = cal_gmse(y, predicted_np)
        per_metrics_dic['time_lag'] = cal_time_lag(y, predicted_np, seq_st_ed, self.CONF['interval'])
        output_dict['per_metrics_dic'] = per_metrics_dic
        output_dict['pred'] = predicted_np

        return output_dict


    def cross_test_inference_for_scikit(self, model, pid, dataset):
        mean, std = self.data_proc.get_mean_std(pid, dataset)


        return_dict = self.data_proc.get_test(self.CONF['device'], self.CONF['seq_len'],  self.CONF['attri_list'], self.CONF['time_attri_list'], pid, dataset)
        G, attris, time_attris, y, seq_st_ed = return_dict['glucose'], return_dict['attris'], return_dict['time_attris'], return_dict['y'], return_dict['seq_st_ed']

        # model.to(self.CONF['device'])    
        temp_batch = 128
        st = 0
        predicted_norm_list = []
        while True:
            ed = st + temp_batch
            # print(st, ed, G[st:ed].shape[0], G.shape[0], )
            attris_batch = attris[st:ed] if attris is not None else attris
            time_attris_batch = time_attris[st:ed] if time_attris is not None else time_attris
            output_dict = model(G[st:ed], attris_batch, time_attris_batch)

            predicted_norm_ = output_dict['pred']
            predicted_norm_list.append(predicted_norm_)

            st = ed
            if ed >= G.shape[0]:
                break
        predicted_norm = np.concatenate(predicted_norm_list, axis=0)

        if predicted_norm.shape[0] != G.shape[0]:
            print('error')

        predicted_norm_np = predicted_norm
        predicted_np = predicted_norm_np * std + mean
    
        output_dict = {}
        per_metrics_dic ={}
        if len(predicted_np.shape) == 1:
            predicted_np = predicted_np[:, None]
            
        per_metrics_dic['rmse'] = mean_squared_error(y, predicted_np[:,0])**0.5
        per_metrics_dic['mape'] = mean_absolute_percentage_error(y, predicted_np[:,0]) * 100
        per_metrics_dic['mae'] = mean_absolute_error(y, predicted_np[:,0])
        per_metrics_dic['grmse'] = cal_gmse(y, predicted_np)
        per_metrics_dic['time_lag'] = cal_time_lag(y, predicted_np, seq_st_ed, self.CONF['interval'])
        output_dict['per_metrics_dic'] = per_metrics_dic
        output_dict['pred'] = predicted_np

        return output_dict