import torch
import torch.nn as nn
from helper_classes import LogCoshLoss
import numpy as np
from closest_sqrt_factor import closest_sqrt_factor
from torchsummary import summary

class Conv(nn.Module):
    def __init__(self, scheme, xshape, yshape):
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
        # omit batch_size from xshape
        conv_block_summ = summary(self.conv_block, xshape[1:], verbose=0)
        # compute the size of the output of the conv_block
        conv_block_output_size = conv_block_summ.summary_list[-1].output_size[1]
        self.linear = nn.Sequential(
                nn.Linear(conv_block_output_size, 1024),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.Dropout(0.3),
                nn.ReLU())
        self.head1 = nn.Linear(512, yshape[1])
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
    def __init__(self, scheme, xshape, yshape):
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.scheme = scheme
        
        # determine input shaping
        # ideally input_shape_0 * input_shape_1 == xsh_prod
        xsh_prod = np.product(xshape[1:])
        input_shape_0 = closest_sqrt_factor(xsh_prod)
        input_shape_1 = xsh_prod // input_shape_0
        self.input_shape = (input_shape_0, input_shape_1)

        # compute linear shape
        # 128 is the hidden size of the LSTM
        # 256 is the hidden size of the last hidden layer?
        linear_shape = input_shape_0 * 128 + 256

        self.lstm = nn.LSTM(input_shape_1, 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(linear_shape, 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU())
        self.head1 = nn.Linear(512, yshape[1])
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = self.input_shape
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
    def __init__(self, scheme, xshape, yshape):
        super(GRU, self).__init__()
        self.name = 'GRU'
        self.scheme = scheme
        
        # determine input shaping
        # ideally input_shape_0 * input_shape_1 == xsh_prod
        xsh_prod = np.product(xshape[1:])
        input_shape_0 = closest_sqrt_factor(xsh_prod)
        input_shape_1 = xsh_prod // input_shape_0
        self.input_shape = (input_shape_0, input_shape_1)

        # compute linear shape
        # 128 is the hidden size of the LSTM
        # 256 is the hidden size of the last hidden layer?
        linear_shape = input_shape_0 * 128 + 256

        self.gru = nn.GRU(input_shape_1, 128, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(linear_shape, 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU())
        self.head1 = nn.Linear(512, yshape[1])
        self.loss1 = nn.MSELoss()
        self.loss2 = LogCoshLoss()
        self.loss3 = nn.L1Loss()
        self.loss4 = nn.BCELoss()
        
    def forward(self, x, y=None):
        shape1, shape2 = self.input_shape
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
