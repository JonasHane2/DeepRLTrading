import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#nn_activation_function = nn.ReLU()
nn_activation_function = nn.LeakyReLU()
#nn_activation_function = nn.SELU()
out_layer_std = 3e-3


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

def _get_seq_out(seq, shape) -> int:
    """ Returns the output dimension of a convolutional sequence. """
    o = seq(torch.zeros(1, 1, shape))
    return int(np.prod(o.size())) 


# ------------------ State
## ----------------- Convolutional + LSTM
class AConvLSTMDiscrete(nn.Module): #DRQN
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, prev_action_size=None, window_size=1, num_lstm_layers=1, action_bounds=1, dropout=0, kaiming_init=False):
        super(AConvLSTMDiscrete, self).__init__()  
        if prev_action_size is not None:
            self.prev_action_size = prev_action_size
        else: 
            self.prev_action_size = action_space
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), #seems like batch norm functions better after activation than before
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.lstm_layer = nn.LSTM(input_size=conv_out, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear((hidden_size+self.prev_action_size), action_space)
        if kaiming_init:
            self.apply(weights_init)
        self.fc_out.weight.data.normal_(0, out_layer_std)

    def forward(self, x, prev_action=None, hx=None):
        x = self.fc_in(x)
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        self.lstm_layer.flatten_parameters()
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device) 
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device) # Add previous action to feature map
        x = self.fc_out(x)
        return x, hx


## ----------------- Convolutional
class AConvDiscrete(nn.Module): #DQN
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, prev_action_size=None, dropout=0.1, kaiming_init=False):
        super(AConvDiscrete, self).__init__()
        if prev_action_size is not None:
            self.prev_action_size = prev_action_size
        else: 
            self.prev_action_size = action_space
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), #seems like batch norm functions better after activation than before
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.fc_out = nn.Linear((conv_out+self.prev_action_size), action_space)
        if kaiming_init:
            self.apply(weights_init)
        self.fc_out.weight.data.normal_(0, out_layer_std)

    def forward(self, x, prev_action=None) -> torch.Tensor:
        x = self.fc_in(x)
        x = x.unsqueeze(1) #Think you have to do this since there is no batch_first arg for conv nets
        x = self.conv(x)
        x = x.view(x.shape[0], -1) #Add all activation maps to one big activation map
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device) 
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device) # Add previous action to feature map
        x = self.fc_out(x)
        return x


## ----------------- LSTM 
# Seems like performance increases if there is an activation function after lstm layer
class ALSTMDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, prev_action_size=None, n_layers=1, dropout=0.1, kaiming_init=False) -> None:
        super(ALSTMDiscrete, self).__init__()
        if prev_action_size is not None:
            self.prev_action_size = prev_action_size
        else: 
            self.prev_action_size = action_space
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear((hidden_size+self.prev_action_size), action_space)        
        if kaiming_init:
            self.apply(weights_init)
        self.fc_out.weight.data.normal_(0, out_layer_std)

    def forward(self, x, prev_action=None, hx=None):
        x = self.fc_in(x)
        self.lstm_layer.flatten_parameters()
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device) 
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device) # Add previous action to feature map
        x = self.fc_out(x)
        return x, hx


## ----------------- FF 
class FFDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, prev_action_size=None, dropout=0.1, kaiming_init=False) -> None:
        super(FFDiscrete, self).__init__()
        if prev_action_size is not None:
            self.prev_action_size = prev_action_size
        else: 
            self.prev_action_size = action_space
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size*2), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
            nn.Linear(hidden_size*2, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        ) 
        self.fc_out = nn.Linear((hidden_size+self.prev_action_size), action_space)
        if kaiming_init:
            self.apply(weights_init)
        self.fc_out.weight.data.normal_(0, out_layer_std)

    def forward(self, x, prev_action=None) -> torch.Tensor:
        x = self.fc_in(x)
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device)         
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device) # Add previous action to feature map
        x = self.fc_out(x)
        return x


## ----------------- Linear 
class LinearDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, prev_action_size=None, dropout=0.1, kaiming_init=False) -> None:
        super(LinearDiscrete, self).__init__()
        if prev_action_size is not None:
            self.prev_action_size = prev_action_size
        else: 
            self.prev_action_size = action_space
        self.fc_out = nn.Linear((observation_space+self.prev_action_size), action_space)
        if kaiming_init:
            self.apply(weights_init)
        self.fc_out.weight.data.normal_(0, out_layer_std)

    def forward(self, x, prev_action=None) -> torch.Tensor:
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device)         
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device) 
        x = self.fc_out(x)
        return x


# ------------------ State-Action
## ----------------- Conv
class CConvSA(nn.Module):
    def __init__(self, observation_space=8, hidden_size=64, action_space=1, dropout=0.1, kaiming_init=False) -> None:
        super(CConvSA, self).__init__()         
        self.prev_action_size = action_space
        self.fc1 = nn.Linear(observation_space,hidden_size*2)
        self.fc2 = nn.Linear(((hidden_size*2)+action_space),hidden_size)
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), 
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.fc3 = nn.Linear((conv_out+self.prev_action_size),1)
        if kaiming_init:
            self.apply(weights_init)
        self.fc3.weight.data.normal_(0, out_layer_std)
        
    def forward(self, state, action, prev_action=None):
        x = self.fc1(state)
        x = nn_activation_function(x)
        x = torch.cat((x, action), dim=1).to(device)
        x = self.fc2(x)
        x = nn_activation_function(x)
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device)         
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device)
        x = self.fc3(x)
        return x


## ----------------- LSTM 
class CLSTMSA(nn.Module):
    def __init__(self, observation_space=8, hidden_size=64, action_space=1, n_layers=1, dropout=0.1, kaiming_init=False) -> None:
        super(CLSTMSA, self).__init__()         
        self.prev_action_size = action_space
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.lstm_layer = nn.LSTM(input_size=(hidden_size+action_space), hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear((hidden_size+self.prev_action_size), action_space)        
        if kaiming_init:
            self.apply(weights_init)
        self.fc_out.weight.data.normal_(0, out_layer_std)
        
    def forward(self, state, action, prev_action=None, hx=None):
        x = self.fc_in(state)
        x = torch.cat((x, action), dim=1).to(device)
        self.lstm_layer.flatten_parameters()
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device)
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device)
        x = self.fc_out(x)
        return x


## ----------------- FF
class CFFSA(nn.Module):
    def __init__(self, observation_space=8, hidden_size=64, action_space=1, dropout=0.1, kaiming_init=False) -> None:
        super(CFFSA, self).__init__()         
        self.prev_action_size = action_space
        self.fc1 = nn.Linear(observation_space,hidden_size*2)
        self.du1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(((hidden_size*2)+action_space),hidden_size)
        self.du2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear((hidden_size+self.prev_action_size),1)
        if kaiming_init:
            self.apply(weights_init)
        self.fc3.weight.data.normal_(0, out_layer_std)
        
    def forward(self, state, action, prev_action=None):
        x = self.fc1(state)
        x = nn_activation_function(x)
        x = self.du1(x)
        x = torch.cat((x, action), dim=1).to(device)
        x = self.fc2(x)
        x = nn_activation_function(x)
        x = self.du2(x)
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device)         
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device) 
        x = self.fc3(x)
        return x


## ----------------- Linear 
class CLinearSA(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, dropout=0.1, kaiming_init=False) -> None:
        super(CLinearSA, self).__init__()
        self.prev_action_size=action_space
        self.fc_out = nn.Linear((observation_space+(action_space*2)), action_space)
        if kaiming_init:
            self.apply(weights_init)
        self.fc_out.weight.data.normal_(0, out_layer_std)

    def forward(self, state, action, prev_action=None) -> torch.Tensor:
        x = torch.cat((state, action), dim=1).to(device)
        if prev_action is None: 
            prev_action = torch.Tensor(np.zeros((x.shape[0],self.prev_action_size))).to(device)         
        x = torch.cat((x, prev_action.to(device)), dim=1).to(device) 
        x = self.fc_out(x)
        return x
