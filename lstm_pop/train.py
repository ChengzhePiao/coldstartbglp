from method import *
from update import *
from dataprocessor import *
from log import *
from learn2learn.algorithms.maml import MAML
from learn2learn.algorithms.meta_sgd import MetaSGD
def get_model(CONF):
    
    if CONF['model_name'] == 'lstm':
        return LSTM_NN(hidden_dim=CONF['hidden_dim'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        )
    elif CONF['model_name'] == 'nbeats':
        return NBEATSGenericBlock_NN(CONF['seq_len'], CONF['hidden_dim'])
    elif CONF['model_name'] == 'nhits':
        return NHiTSModule_NN(CONF['seq_len'], CONF['hidden_dim'], CONF['hidden_dim']//8)

    

def create_pid2model(data_proc, CONF, model):
    pid2model = {}
    for pid in data_proc.test_pid2data_npy:
        pid2model[pid] = get_model(CONF)
        pid2model[pid].to(CONF['device'])
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pid2model[pid].load_state_dict(copy.deepcopy(model.state_dict())) 
        # same with pid2model[pid].load_state_dict(model.state_dict()) no difference
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return pid2model


def update_metrics_dic(metrics_dic, per_metrics_dic):
    for mtc in per_metrics_dic:
        if mtc not in metrics_dic:
            metrics_dic[mtc] = []
        metrics_dic[mtc].append(per_metrics_dic[mtc])
    return metrics_dic

def print_metrics_dic(metrics_dic):

    for mtc in metrics_dic:
        print(f'{mtc}, {np.mean(metrics_dic[mtc]):.5f}, {np.std(metrics_dic[mtc]):.5f}')
    
    print(metrics_dic['rmse'])



def train(CONF):
    global_model = get_model(CONF)
    global_model.to(CONF['device'])
    global_model.train()

    best_val_rmse = np.inf
    best_model = None

    data_proc = PopulationDataProcessor(CONF)

    log = Log(CONF['dataset'] + '_' + CONF['model_name'] + '_' + CONF['comments'], CONF)

    if 'rewrite' in CONF and not CONF['rewrite'] and os.path.exists(os.path.join(log.root_dir, 'pid2prediction.npy')):
        print('exists.... don\'t rewrite')
        return
    
    update = Update(data_proc, CONF)
    best_val_rmse = np.inf

    pbar = tqdm(range(CONF['epochs']), desc = 'global_loop')

    for epoch in pbar:

        update.update_weights(
                model=global_model, local_epochs=CONF['local_epochs'], lr=CONF['lr'])

        if epoch % CONF['print_every'] == 0 and epoch >= CONF['start_print']:

            rmse_list = []
            mape_list = []
            for pid in data_proc.valid_pid2data_npy.keys():
                if CONF['random_valid']:
                    output_dict = update.random_inference(global_model, pid)
                else:
                    output_dict = update.inference(global_model, pid)
                rmse, mape = output_dict['rmse'], output_dict['mape']
                rmse_list.append(rmse)
                mape_list.append(mape)

            log.record_eval_rmse(rmse_list, (epoch + 1) * CONF['local_epochs'])
            rmse = np.mean(rmse_list)
            mape = np.mean(mape_list)

            if best_val_rmse > rmse:
                best_val_rmse = rmse
                best_model = copy.deepcopy(global_model)

            best_model = copy.deepcopy(global_model)



    metrics_dic = {}
    pid2prediction = {}
    for pid in data_proc.test_pid2data_npy.keys():
        output_dict = update.test_inference(best_model, pid)
        per_metrics_dic = output_dict['per_metrics_dic']
        metrics_dic = update_metrics_dic(metrics_dic=metrics_dic, per_metrics_dic=per_metrics_dic)
        pid2prediction[pid] = output_dict['pred']
    print(' \n Results after ' +
        str(CONF['epochs'])+' global rounds of training:')
    print_metrics_dic(metrics_dic)
    log.save_prediction(pid2prediction)
    log.save_metrics_dic(metrics_dic)
    log.save_epoch_rmse()
    if CONF['save_model']:
        log.save_models(best_model)
        print('models have been saved...')
    
def average_weights(pid2model):
    """
    Returns the average of the weights.
    """
    pid_list = list(pid2model.keys())

    pid2state_dict = {pid: pid2model[pid].state_dict() for pid in pid_list}

    w_avg = copy.deepcopy(pid2state_dict[pid_list[0]])

    for key in w_avg.keys():
        for pid in pid_list[1:]:
            w_avg[key] += pid2state_dict[pid][key]
        w_avg[key] = torch.div(w_avg[key], len(pid_list))
    return w_avg

def train_fl(CONF):
    global_model = get_model(CONF)
    global_model.to(CONF['device'])
    global_model.train()

    best_val_rmse = np.inf
    

    data_proc = PopulationDataProcessor(CONF)

    pid2model = create_pid2model(data_proc, CONF, global_model)


    log = Log(CONF['dataset'] + '_' + CONF['model_name'] + '_' + CONF['comments'], CONF)

    if 'rewrite' in CONF and not CONF['rewrite'] and os.path.exists(os.path.join(log.root_dir, 'pid2prediction.npy')):
        print('exists.... don\'t rewrite')
        return
    
    update = Update(data_proc, CONF)
    best_val_rmse = np.inf

    pbar = tqdm(range(CONF['epochs']), desc = 'global_loop')
    best_model = None

    for epoch in pbar:

        for pid in pid2model:
            pid2model[pid].load_state_dict(copy.deepcopy(global_model.state_dict()))

        for pid in pid2model:
            update.update_weights(
                    model=pid2model[pid], 
                    local_epochs=CONF['local_epochs'], 
                    lr=CONF['lr'], 
                    pid=pid)

        w_avg = average_weights(pid2model)
        global_model.load_state_dict(w_avg)

        if epoch % CONF['print_every'] == 0 and epoch >= CONF['start_print']:

            rmse_list = []
            mape_list = []
            for pid in data_proc.valid_pid2data_npy.keys():
                if CONF['random_valid']:
                    output_dict = update.random_inference(global_model, pid)
                else:
                    output_dict = update.inference(global_model, pid)
                rmse, mape = output_dict['rmse'], output_dict['mape']
                rmse_list.append(rmse)
                mape_list.append(mape)

            log.record_eval_rmse(rmse_list, (epoch + 1) * CONF['local_epochs'])
            rmse = np.mean(rmse_list)
            mape = np.mean(mape_list)

            if best_val_rmse > rmse:
                best_val_rmse = rmse
                best_model = copy.deepcopy(global_model)

            best_model = copy.deepcopy(global_model)



    metrics_dic = {}
    pid2prediction = {}
    for pid in data_proc.test_pid2data_npy.keys():
        output_dict = update.test_inference(best_model, pid)
        per_metrics_dic = output_dict['per_metrics_dic']
        metrics_dic = update_metrics_dic(metrics_dic=metrics_dic, per_metrics_dic=per_metrics_dic)
        pid2prediction[pid] = output_dict['pred']
    print(' \n Results after ' +
        str(CONF['epochs'])+' global rounds of training:')
    print_metrics_dic(metrics_dic)
    log.save_prediction(pid2prediction)
    log.save_metrics_dic(metrics_dic)
    log.save_epoch_rmse()
    if CONF['save_model']:
        log.save_models(best_model)
        print('models have been saved...')



def train_maml(CONF):
    global_model = get_model(CONF)
    global_model.to(CONF['device'])
    
    if CONF['meta'] == 'MAML':
        global_model = MAML(global_model, lr=CONF['fast_lr'], first_order = False)
    elif CONF['meta'] == 'MetaSGD':
        global_model = MetaSGD(global_model, lr=CONF['fast_lr'], first_order = False)
    else:
        print('error in meta')

    global_model.train()

    best_val_rmse = np.inf
    best_model = None

    data_proc = PopulationDataProcessor(CONF)

    log = Log(CONF['dataset'] + '_' + CONF['model_name'] + '_' + CONF['comments'], CONF)

    if 'rewrite' in CONF and not CONF['rewrite'] and os.path.exists(os.path.join(log.root_dir, 'pid2prediction.npy')):
        print('exists.... don\'t rewrite')
        return
    
    update = Update(data_proc, CONF)
    best_val_rmse = np.inf

    pbar = tqdm(range(CONF['epochs']), desc = 'global_loop')

    for epoch in pbar:

        update.update_weights_maml(
                model=global_model, meta_batch_size=CONF['meta_batch_size'], lr=CONF['lr'])


        # if CONF['keep_last_model']:
        #     best_model = copy.deepcopy(global_model)
        if epoch % CONF['print_every'] == 0 and epoch >= CONF['start_print']:

            rmse_list = []
            mape_list = []
            for pid in data_proc.valid_pid2data_npy.keys():
                if CONF['random_valid']:
                    output_dict = update.random_inference(global_model, pid)
                else:
                    output_dict = update.inference(global_model, pid)
                rmse, mape = output_dict['rmse'], output_dict['mape']
                rmse_list.append(rmse)
                mape_list.append(mape)

            log.record_eval_rmse(rmse_list, (epoch + 1) * CONF['meta_batch_size'])
            rmse = np.mean(rmse_list)
            mape = np.mean(mape_list)

            if best_val_rmse > rmse:
                best_val_rmse = rmse
                best_model = copy.deepcopy(global_model)

            best_model = copy.deepcopy(global_model)



    metrics_dic = {}
    pid2prediction = {}
    for pid in data_proc.test_pid2data_npy.keys():
        output_dict = update.test_inference(best_model, pid)
        per_metrics_dic = output_dict['per_metrics_dic']
        metrics_dic = update_metrics_dic(metrics_dic=metrics_dic, per_metrics_dic=per_metrics_dic)
        pid2prediction[pid] = output_dict['pred']
    print(' \n Results after ' +
        str(CONF['epochs'])+' global rounds of training:')
    print_metrics_dic(metrics_dic)
    log.save_prediction(pid2prediction)
    log.save_metrics_dic(metrics_dic)
    log.save_epoch_rmse()
    if CONF['save_model']:
        log.save_models(best_model)
        print('models have been saved...')

def select_batch_of_pid_model(pid2model, dfl_random_batch, pid, dead_client_list=None):

    all_pid_list = list(pid2model.keys())
    all_pid_list.remove(pid)

    if dead_client_list is not None:
        all_pid_list = list(set(all_pid_list) - set(dead_client_list))

    dfl_random_batch = min(dfl_random_batch - 1, len(all_pid_list))

    selected_pid = list(np.random.choice(all_pid_list, dfl_random_batch , replace=False))
    
    selected_pid = selected_pid + [pid]

    return {pid:pid2model[pid] for pid in selected_pid}

def select_batch_of_pid_model_wf(pid2model, dfl_random_batch, pid, dead_client_list=None):

    all_pid_list = list(pid2model.keys())
    all_pid_list.remove(pid)

    dfl_random_batch = min(dfl_random_batch - 1, len(all_pid_list))

    selected_pid = list(np.random.choice(all_pid_list, dfl_random_batch , replace=False))
    
    if dead_client_list is not None:
        selected_pid = list(set(selected_pid) - set(dead_client_list))
    
    selected_pid = selected_pid + [pid]

    return {pid:pid2model[pid] for pid in selected_pid}

def select_ring_batch_of_pid_model(pid2model, pid, dead_client_list=None):
    all_pid_list = list(pid2model.keys())
    all_pid_list.sort()
    idx = all_pid_list.index(pid)

    pre_idx = idx - 1
    aft_idx = idx + 1 if idx != len(all_pid_list) - 1 else 0

    selected_pid = [pid]
    if dead_client_list is None or all_pid_list[pre_idx] not in dead_client_list:
        selected_pid += [all_pid_list[pre_idx]]
    elif dead_client_list is None or all_pid_list[aft_idx] not in dead_client_list:
        selected_pid += [all_pid_list[aft_idx]]

    return {pid:pid2model[pid] for pid in selected_pid}



def divide_chunks(l, n): 
    # looping till length l
    
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
    
def select_cluster_batch_of_pid_model(pid2model, cluster_size, min_cluster_size, pid, dead_client_list=None):

    all_pid_list = list(pid2model.keys())
    all_pid_list.sort()

    chunks = list(divide_chunks(all_pid_list, cluster_size))
    if len(chunks[-1]) < min_cluster_size and len(chunks) > 1:
        temp = copy.deepcopy(chunks[-1])
        chunks.pop(-1)
        chunks[-1] += temp
        
    selected_pid = []
    for idx, chunk in enumerate(chunks):
        if pid in chunk:
            
            for other_pid in chunk:
                if dead_client_list is None or other_pid not in dead_client_list:
                    selected_pid += [other_pid]
                        
            if pid == chunk[0]:
                if dead_client_list is None or chunks[idx-1][-1] not in dead_client_list:
                    selected_pid += [chunks[idx-1][-1]]
            elif pid == chunk[-1]:
                aft_idx = idx + 1 if idx != len(chunks) - 1 else 0
                if dead_client_list is None or chunks[aft_idx][0] not in dead_client_list:
                    selected_pid += [chunks[aft_idx][0]]               
            break
        
    return {pid:pid2model[pid] for pid in selected_pid}

def train_dfl_random(CONF):
    global_model = get_model(CONF)
    global_model.to(CONF['device'])
    global_model.train()

    best_val_rmse = np.inf
    

    data_proc = PopulationDataProcessor(CONF)

    pid2model = create_pid2model(data_proc, CONF, global_model)


    log = Log(CONF['dataset'] + '_' + CONF['model_name'] + '_' + CONF['comments'], CONF)

    if 'rewrite' in CONF and not CONF['rewrite'] and os.path.exists(os.path.join(log.root_dir, 'pid2prediction.npy')):
        print('exists.... don\'t rewrite')
        return
    
    update = Update(data_proc, CONF)
    best_val_rmse = np.inf

    pbar = tqdm(range(CONF['epochs']), desc = 'global_loop')
    best_model = None
    dead_client_list = None 

    for epoch in pbar:

        # for pid in pid2model:
        #     pid2model[pid].load_state_dict(copy.deepcopy(global_model.state_dict()))
        new_pid2model = {}

        if 'asyn' in CONF['dfl_mode']:
            all_pid_list = list(pid2model.keys())
            dfl_random_batch = int(len(all_pid_list) * CONF['dead_client_ratio'])
            dead_client_list = list(np.random.choice(all_pid_list, dfl_random_batch, replace=False))

        for pid in pid2model:
            if dead_client_list is not None and pid in dead_client_list:
                new_pid2model[pid] = copy.deepcopy(pid2model[pid])
                continue
            
            if 'random' in CONF['dfl_mode'] and 'wf' not in CONF['dfl_mode']:
                selected_pid2model = select_batch_of_pid_model(pid2model, CONF['dfl_random_batch'], pid, dead_client_list)
            elif 'random' in CONF['dfl_mode'] and 'wf' in CONF['dfl_mode']:
                selected_pid2model = select_batch_of_pid_model_wf(pid2model, CONF['dfl_random_batch'], pid, dead_client_list)
            elif 'ring' in CONF['dfl_mode']: 
                selected_pid2model = select_ring_batch_of_pid_model(pid2model, pid, dead_client_list)
            elif 'cluster' in CONF['dfl_mode']:
                selected_pid2model = select_cluster_batch_of_pid_model(pid2model, CONF['cluster_size'], CONF['min_cluster_size'], pid, dead_client_list)
                

            w_avg = average_weights(selected_pid2model)
            new_pid2model[pid] = copy.deepcopy(pid2model[pid])
            new_pid2model[pid].load_state_dict(w_avg)
        pid2model = new_pid2model



        for pid in pid2model:
            if dead_client_list is not None and pid in dead_client_list:
                continue
            update.update_weights(
                    model=pid2model[pid], 
                    local_epochs=CONF['local_epochs'], 
                    lr=CONF['lr'], 
                    pid=pid)
        
        if epoch % CONF['print_every'] == 0 and epoch >= CONF['start_print']:
            w_avg = average_weights(pid2model)
            global_model.load_state_dict(w_avg)
            rmse_list = []
            mape_list = []
            for pid in data_proc.valid_pid2data_npy.keys():
                if CONF['random_valid']:
                    output_dict = update.random_inference(global_model, pid)
                else:
                    output_dict = update.inference(global_model, pid)
                rmse, mape = output_dict['rmse'], output_dict['mape']
                rmse_list.append(rmse)
                mape_list.append(mape)

            log.record_eval_rmse(rmse_list, (epoch + 1) * CONF['local_epochs'])
            rmse = np.mean(rmse_list)
            mape = np.mean(mape_list)

            if best_val_rmse > rmse:
                best_val_rmse = rmse
                best_model = copy.deepcopy(global_model)

            best_model = copy.deepcopy(global_model)



    metrics_dic = {}
    pid2prediction = {}
    for pid in data_proc.test_pid2data_npy.keys():
        output_dict = update.test_inference(best_model, pid)
        per_metrics_dic = output_dict['per_metrics_dic']
        metrics_dic = update_metrics_dic(metrics_dic=metrics_dic, per_metrics_dic=per_metrics_dic)
        pid2prediction[pid] = output_dict['pred']
    print(' \n Results after ' +
        str(CONF['epochs'])+' global rounds of training:')
    print_metrics_dic(metrics_dic)
    log.save_prediction(pid2prediction)
    log.save_metrics_dic(metrics_dic)
    log.save_epoch_rmse()
    if CONF['save_model']:
        log.save_models(best_model)
        print('models have been saved...')

def train_scikit_learn(CONF, seed):
    data_proc = PopulationDataProcessor(CONF)
    log = Log(CONF['dataset'] + '_' + CONF['model_name'] + '_' + CONF['comments'], CONF)
    if CONF['model_name'] == 'lr':
        model = LR(CONF, data_proc)
        log.save_scikit_models(model)
    elif CONF['model_name'] == 'xgboost':
        model = XGB(CONF, data_proc, seed)
        log.save_xgboost_models(model)

    