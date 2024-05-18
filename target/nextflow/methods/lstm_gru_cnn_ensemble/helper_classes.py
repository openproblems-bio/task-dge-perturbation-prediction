import torch
import torch.nn as nn

class LogCoshLoss(nn.Module):
    """Loss function for regression tasks"""
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = (y_t - y_prime_t)/3 # divide by 3 to avoid numerical overflow in cosh
        clipped_ey_t = torch.clamp(ey_t, min=-50, max=50)
        return torch.mean(torch.log(torch.cosh(clipped_ey_t + 1e-12)))
    
    
class Dataset:
    """Python class to load the data for training and inference in Pytorch"""
    def __init__(self, data_x, data_y=None):
        super(Dataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        if self.data_y is not None:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]
