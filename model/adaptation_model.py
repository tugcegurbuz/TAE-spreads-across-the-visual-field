import torch
from adaptation_layers import *
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, batch_size, theta, xs, ys):
        activation_a, activation = self.encoder(batch_size, theta, xs, ys)
        rho_hat = self.decoder(batch_size, activation, xs, ys)
        return activation_a, rho_hat