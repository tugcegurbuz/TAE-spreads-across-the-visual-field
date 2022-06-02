import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    def f(self, batch_size, theta):
        psi = torch.arange(0, 180, 1).view(1, 180)
        psi = psi.repeat(batch_size, 1)
        sigma = 20.87
        f_a = torch.exp(-(
            (theta - psi) ** 2) / (2 * sigma ** 2)
            )
        return f_a.view(batch_size, 1, 180)
    def r(self, batch_size, xs, ys, sigma_x, sigma_y):
        y_, x_ = np.mgrid[0:20, 0:20]
        x = torch.from_numpy(x_).view(1, 20, 20)
        y = torch.from_numpy(y_).view(1, 20, 20)
        r_a = torch.exp(-(
            ((x - xs) / sigma_x ) ** 2 + ((y - ys) / sigma_y) ** 2 )
                           )
        r_a = r_a.repeat(batch_size, 1, 1)
        return r_a.view(batch_size, 400, 1)
    def forward(self, batch_size, theta, xs, ys, sigma_x, sigma_y):
        A_s = self.f(batch_size, theta) * self.r(batch_size, xs, ys, sigma_x, sigma_y)
        return A_s

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    def C(self, A_s,  xs, ys):
        x_raw = xs - .5
        x_raw = x_raw.long()
        ys = ys.long()
        idx = 20 * ys + x_raw
        idx = idx[0, 0, 0]
        c = A_s[:, idx, :] #indexing is [y, x]
        return c
    def thetaHat(self, batch_size, c):
        beta = 260
        psi = torch.arange(0, 180).view(1, 180)
        num = beta * c
        denom = torch.sum(num, 1)
        avg = num / denom.view(batch_size, 1)
        return torch.sum(avg * psi, 1)
    def readout(self, theta_hat):
        rhohat = torch.max(torch.tensor([0.]), 
                          torch.min(torch.tensor([1.]), 
                                    ((theta_hat - 90 + 2.0001) / 2)))
        return rhohat
    def forward(self, batch_size, A_s, xs, ys):
        c = self.C(A_s, xs, ys)
        theta_hat = self.thetaHat(batch_size, c)
        rho_hat = self.readout(theta_hat)
        return rho_hat