import torch
import torch.nn as nn
from no_adaptation_layers import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, batch_size, theta, xs, ys, sigma_x, sigma_y):
        activation = self.encoder(batch_size, theta, xs, ys, sigma_x, sigma_y)
        rho_hat = self.decoder(batch_size, activation, xs, ys)
        return activation, rho_hat