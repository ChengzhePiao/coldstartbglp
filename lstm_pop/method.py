from util import *

class LSTM_NN(nn.Module):
    def __init__(self, hidden_dim, num_varis):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.feature_dim = hidden_dim // num_varis

        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.rnn = nn.LSTMCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris, time_attris,):

        if attris is not None and time_attris is not None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        elif attris is None and time_attris is None:
            X = torch.unsqueeze(G, dim=2) # B, T , C
        elif attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), time_attris], dim=2) # B, T , C
        elif time_attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        
        B, T, C = X.shape

        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        hc = None

        for t in range(T):
        
            rnn_in = X_emb[:, t].view(B, -1) # B, T, C*F

            hc = self.rnn(rnn_in, hc) 

        pred = self.output(hc[0])

        output_dict = {
            'pred': pred,
        }

        return output_dict
    

class LR():
    def __init__(self, CONF, data_proc=None, is_load_model=False):
        self.CONF = CONF

        self.data_proc = data_proc

        self.model = LinearRegression()
        
        if is_load_model is True:
            return

        # batch_size, device, seq_len, attri_list, time_attri_list, pid=None, all_data = None
        data_set = data_proc.get_train_batch(batch_size=None, device='cpu', seq_len=self.CONF['seq_len'], attri_list=self.CONF['attri_list'], time_attri_list=self.CONF['time_attri_list'],
                                        all_data=True)
        G, attris, time_attris, y_norm, target_time = data_set['glucose'], data_set['attris'], data_set['time_attris'], data_set['y'], data_set['target_time']

        if attris is not None and time_attris is not None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        elif attris is None and time_attris is None:
            X = torch.unsqueeze(G, dim=2) # B, T , C
        elif attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), time_attris], dim=2) # B, T , C
        elif time_attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        
        y_norm = y_norm.detach().cpu().numpy()
        
        X = X.detach().cpu().numpy()

        B, T, C = X.shape
        X = X.transpose(0, 2, 1) # B C T
        # X = X[:, 0, :]
        X = X.reshape(B, -1)
        self.model.fit(X, y_norm)


    def __call__(self, G, attris, time_attris,):

        if attris is not None and time_attris is not None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        elif attris is None and time_attris is None:
            X = torch.unsqueeze(G, dim=2) # B, T , C
        elif attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), time_attris], dim=2) # B, T , C
        elif time_attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        
        B, T, C = X.shape

        X = X.permute([0, 2, 1]) # B C T
        X = X.reshape(B, -1)
        X = X.detach().cpu().numpy()

        pred = self.model.predict(X)

        output_dict = {
            'pred': pred,
        }

        return output_dict
        

class XGB():
    def __init__(self, CONF, data_proc=None, seed=1, is_load_model=False):
        self.CONF = CONF
        self.model = xgb.XGBRegressor(
            # tree_method='gpu_hist', 
            objective="reg:squarederror", 
            random_state=seed, 
            min_child_weight=5, gamma=1, subsample=0.95, colsample_bytree=1.0, max_depth=5, n_estimators=50, learning_rate=0.1)
        if is_load_model is True:
            return

        # batch_size, device, seq_len, attri_list, time_attri_list, pid=None, all_data = None
        data_set = data_proc.get_train_batch(batch_size=None, device='cpu', seq_len=self.CONF['seq_len'], attri_list=self.CONF['attri_list'], time_attri_list=self.CONF['time_attri_list'],
                                        all_data=True)
        G, attris, time_attris, y_norm, target_time = data_set['glucose'], data_set['attris'], data_set['time_attris'], data_set['y'], data_set['target_time']

        if attris is not None and time_attris is not None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        elif attris is None and time_attris is None:
            X = torch.unsqueeze(G, dim=2) # B, T , C
        elif attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), time_attris], dim=2) # B, T , C
        elif time_attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        
        y_norm = y_norm.detach().cpu().numpy()
        
        X = X.detach().cpu().numpy()

        B, T, C = X.shape
        X = X.transpose(0, 2, 1) # B C T
        # X = X[:, 0, :]
        X = X.reshape(B, -1)
        self.model.fit(X, y_norm)

    def save_model(self, path):
        self.model.save_model(path)
    def load_model(self, path):
        self.model.load_model(path)
        
    def __call__(self, G, attris, time_attris,):

        if attris is not None and time_attris is not None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        elif attris is None and time_attris is None:
            X = torch.unsqueeze(G, dim=2) # B, T , C
        elif attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), time_attris], dim=2) # B, T , C
        elif time_attris is None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        
        B, T, C = X.shape

        X = X.permute([0, 2, 1]) # B C T
        X = X.reshape(B, -1)
        X = X.detach().cpu().numpy()

        pred = self.model.predict(X)

        output_dict = {
            'pred': pred,
        }

        return output_dict
    
from pytorch_forecasting.models.nbeats.sub_modules import NBEATSGenericBlock
class NBEATSGenericBlock_NN(nn.Module):
    def __init__(self, time_length,  hidden_dim) :
        super().__init__()
        self.nbeats = NBEATSGenericBlock(                    
            units=hidden_dim,
            thetas_dim=hidden_dim,
            num_block_layers=1,
            dropout = 0.0,
            backcast_length=time_length,
            forecast_length=1,
        ) 

    def forward(self, G, attris, time_attris,):

        X_tar = G
        backcast, forecast = self.nbeats(X_tar)
        

        y = forecast
                
        output_dict = {
            'pred': y,
        }
        return output_dict
    
    
from pytorch_forecasting.models.nhits.sub_modules import NHiTS as NHiTSModule
class NHiTSModule_NN(nn.Module):
    def __init__(self, time_length,  hidden_dim, output_hidden_dim) :
        super().__init__()
        self.nhits = NHiTSModule(
            context_length=time_length,
            prediction_length=1,
            covariate_size=0,
            output_size=[output_hidden_dim],
            static_size=0,
            static_hidden_size=0,
            n_blocks=[1],
            n_layers=1 * [8], # n_blocks
            hidden_size=1 * [8 * [hidden_dim]], # n_blocks
            pooling_sizes=1 * [1],
            downsample_frequencies=1 * [1], # n_blocks 
            pooling_mode="max",
            interpolation_mode='nearest',
            dropout=0.0,
            activation="ReLU",
            initialization="orthogonal",
            batch_normalization=False,
            shared_weights=True,
            naive_level=True,
        )
        self.output = nn.Sequential(
            nn.Linear(
                output_hidden_dim,
                1,
            ),
        )
    def forward(self, G, attris, time_attris,):

        X_tar = torch.unsqueeze(G, dim=2)
        # X_fea = time_attris # B, T , C   
        X_msk = torch.ones_like(X_tar[..., 0], device=X_tar.device)
        forecast, _, _, _ = self.nhits(X_tar, X_msk, None, None, None)
        

        y = self.output(forecast.squeeze(dim=1))

                
        output_dict = {
            'pred': y,
        }
        return output_dict