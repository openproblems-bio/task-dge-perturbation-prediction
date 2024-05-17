import torch
import torch.nn as nn
from helper_classes import LogCoshLoss

dims_dict = {
    'conv': {'heavy': 15624, 'light': 5312, 'initial': 10472},
    'rnn': {
        'linear': {'heavy': 3072, 'light': 9728, 'initial': 29440},
        'input_shape': {'heavy': [22, 5861], 'light': [74, 593], 'initial': [228, 379]}
    }
}

class Conv(nn.Module):
    def __init__(self, scheme):
        super(Conv, self).__init__()
        self.name = 'Conv'
        self.conv_block = nn.Sequential(nn.Conv1d(1, 8, 5, stride=1, padding=0),
                                        nn.Dropout(0.3),
                                        nn.Conv1d(8, 8, 5, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv1d(8, 16, 5, stride=2, padding=0),
                                        nn.Dropout(0.3),
                                        nn.AvgPool1d(11),
                                        nn.Conv1d(16, 8, 3, stride=3, padding=0),
                                        nn.Flatten())
        self.scheme = scheme
        self.linear = nn.Sequential(
                nn.Linear(dims_dict['conv'][self.scheme], 1024),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.Dropout(0.3),
                nn.ReLU())
        # self.head1 = nn.Linear(512, 18211)
        self.head1 = nn.Linear(512, 21265)
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        if y is None:
            out = self.conv_block(x)
            out = self.head1(self.linear(out))
            return out
        else:
            out = self.conv_block(x)
            out = self.head1(self.linear(out))
            loss1 = 0.4*self.loss1(out, y) + 0.3*self.loss2(out, y) + 0.3*self.loss3(out, y)
            yhat = torch.sigmoid(out)
            yy = torch.sigmoid(y)
            loss2 = self.loss4(yhat, yy)
            return 0.8*loss1 + 0.2*loss2


class LSTM(nn.Module):
    def __init__(self, scheme):
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.scheme = scheme
        self.lstm = nn.LSTM(dims_dict['rnn']['input_shape'][self.scheme][1], 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dims_dict['rnn']['linear'][self.scheme], 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU())
        # self.head1 = nn.Linear(512, 18211)
        self.head1 = nn.Linear(512, 21265)
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = dims_dict['rnn']['input_shape'][self.scheme]
        x = x.reshape(x.shape[0],shape1,shape2)
        if y is None:
            out, (hn, cn) = self.lstm(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            return out
        else:
            out, (hn, cn) = self.lstm(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            loss1 = 0.4*self.loss1(out, y) + 0.3*self.loss2(out, y) + 0.3*self.loss3(out, y)
            yhat = torch.sigmoid(out)
            yy = torch.sigmoid(y)
            loss2 = self.loss4(yhat, yy)
            return 0.8*loss1 + 0.2*loss2
        
        
class GRU(nn.Module):
    def __init__(self, scheme):
        super(GRU, self).__init__()
        self.name = 'GRU'
        self.scheme = scheme
        self.gru = nn.GRU(dims_dict['rnn']['input_shape'][self.scheme][1], 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dims_dict['rnn']['linear'][self.scheme], 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU())
        # self.head1 = nn.Linear(512, 18211)
        self.head1 = nn.Linear(512, 21265)
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = dims_dict['rnn']['input_shape'][self.scheme]
        x = x.reshape(x.shape[0],shape1,shape2)
        if y is None:
            out, hn = self.gru(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            return out
        else:
            out, hn = self.gru(x)
            out = out.reshape(out.shape[0],-1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            loss1 = 0.4*self.loss1(out, y) + 0.3*self.loss2(out, y) + 0.3*self.loss3(out, y)
            yhat = torch.sigmoid(out)
            yy = torch.sigmoid(y)
            loss2 = self.loss4(yhat, yy)
            return 0.8*loss1 + 0.2*loss2
