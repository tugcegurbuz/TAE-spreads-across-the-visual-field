import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.alpha1 = nn.Parameter(torch.Tensor(1))
        self.alpha2 = nn.Parameter(torch.Tensor(1))
        self.beta1 = nn.Parameter(torch.Tensor(1))
        self.beta2 = nn.Parameter(torch.Tensor(1))
        self.sigma_x = nn.Parameter(torch.ones(1, 20, 20))
        self.sigma_y = nn.Parameter(torch.ones(1, 20, 20))
        nn.init.uniform_(self.alpha1, 0., 1.)
        nn.init.uniform_(self.alpha2, 0., 1.)
        nn.init.uniform_(self.beta1, 0., 1.)
        nn.init.uniform_(self.beta2, 0., 1.)
        nn.init.xavier_uniform_(self.sigma_x, gain=1.0)
        nn.init.xavier_uniform_(self.sigma_y, gain=1.0)
    def fa(self, batch_size):
        psi = torch.arange(0, 180, 1).view(1, 180)
        psi = psi.repeat(batch_size, 1)
        sigma = 20.87
        theta = 105.
        f_a = torch.exp(-(
            (theta - psi) ** 2) / (2 * sigma ** 2)
            )
        return f_a.view(batch_size, 1, 180)
    def ra(self, batch_size, sigma_x, sigma_y):
        y_, x_ = np.mgrid[0:20, 0:20]
        x = torch.from_numpy(x_).view(1, 20, 20)
        y = torch.from_numpy(y_).view(1, 20, 20)
        xs = 10.5
        ys = 10
        r_a = torch.exp(-(
            ((x - xs) / sigma_x ) ** 2 + ((y - ys) / sigma_y) ** 2 )
                           )
        r_a = r_a.repeat(batch_size, 1, 1)
        return r_a.view(batch_size, 400, 1)
    def wf(self, A_a, alpha1, alpha2):
        return 1 - alpha1 * ((A_a) ** alpha2)
    
    def b(self, A_a, beta1, beta2):
        return beta1 * ((A_a) ** beta2)
    def sign(self, batch_size, theta):
        sign = torch.max(torch.tensor([-1.]), 
                          torch.min(torch.tensor([1.]), 
                                    ((torch.tensor([105.]) - theta) / 1e-3)))
        return sign.view(batch_size, 1, 1)
    def fs(self, batch_size, theta, A_a, beta1, beta2):
        psi = torch.arange(0, 180, 1).view(1, 180)
        psi = psi.repeat(batch_size, 1).view(batch_size, 180, 1)
        sigma = 20.87
        f_s = torch.exp(-(
            (theta.view(batch_size, 1, 1) - (self.b(A_a, beta1, beta2).view(batch_size, 180, 400) * self.sign(batch_size, theta)  + psi)) ** 2) / (2 * sigma ** 2)
            )
        return f_s.view(batch_size, 400, 180)
    def rs(self, batch_size, xs, ys, sigma_x, sigma_y):
        y_, x_ = np.mgrid[0:20, 0:20]
        x = torch.from_numpy(x_).view(1, 20, 20)
        y = torch.from_numpy(y_).view(1, 20, 20)
        r_s = torch.exp(-(
            ((x - xs) / sigma_x ) ** 2 + ((y - ys) / sigma_y) ** 2 )
                           )
        return r_s.view(batch_size, 400, 1)
    def forward(self, batch_size, theta, xs, ys):
        A_a = self.fa(batch_size) * self.ra(batch_size, self.sigma_x, self.sigma_y)
        A_s = self.wf(A_a, self.alpha1, self.alpha2) * self.fs(batch_size, theta, A_a, self.beta1, self.beta2) * self.rs(batch_size, xs, ys, self.sigma_x, self.sigma_y)
        return A_a, A_s

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    def C(self, A_s,  xs, ys):
        x_raw = xs - .5
        x_raw = x_raw.long()
        ys = ys.long()
        idx = 20 * ys + x_raw
        idx = idx[0, 0, 0]
        c = A_s[:, idx, :]
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
