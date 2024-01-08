# %%
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time as time
import numpy as np
from gradient_descent_the_ultimate_optimizer import gdtuo
from gradient_descent_the_ultimate_optimizer.gdtuo import Optimizable
import os
import matplotlib.pyplot as plt
import imageio
from IPython.display import Video, Image

class FourierAct(Optimizable):
    def __init__(self, optimizer, rank = 5):

        self.rank = rank
        self.optimizer = optimizer

        self.Acoeffs = torch.randn(self.rank + 1, requires_grad=True)
        self.Bcoeffs = torch.randn(self.rank, requires_grad=True)
        self.period = torch.tensor(2 * math.pi, requires_grad=True)

        self.parameters = {'Acoeffs': self.Acoeffs, 'Bcoeffs': self.Bcoeffs, 'period': self.period}
        self.all_params_with_gradients = [self.parameters['Acoeffs'], self.parameters['Bcoeffs'], self.parameters['period']]
        super().__init__(self.parameters, self.optimizer)

    def __call__(self, x):

        out = self.parameters['Acoeffs'][0] * torch.ones_like(x)

        for i in range (1, self.rank + 1):
            out += self.parameters['Acoeffs'][i] * torch.cos(2 * math.pi * i * x / self.parameters['period']) + self.parameters['Bcoeffs'][i - 1] * torch.sin(2 * math.pi * i * x / self.parameters['period'])

        return out
    
    def step(self):
        self.optimizer.step(self.parameters)

class Fourier_Act_torch(nn.Module):
    def __init__(self, rank = 5):
        super(Fourier_Act_torch, self).__init__()
        self.rank = rank
        self.Acoeffs = nn.Parameter(torch.randn(self.rank + 1, requires_grad=True))
        self.Bcoeffs = nn.Parameter(torch.randn(self.rank, requires_grad=True))
        self.period = nn.Parameter(torch.tensor(2 * math.pi, requires_grad=True))
        self.params = {'Acoeffs': self.Acoeffs, 'Bcoeffs': self.Bcoeffs, 'period': self.period}

    def forward(self, x):
        out = self.Acoeffs[0] * torch.ones_like(x)

        for i in range (1, self.rank + 1):
            out += self.Acoeffs[i] * torch.cos(2 * math.pi * i * x / self.period) + self.Bcoeffs[i - 1] * torch.sin(2 * math.pi * i * x / self.period)

        return out
    
mode = 'torch' # 'torch' or 'gdtuo'

data = torch.linspace(-6, 6, 10000).reshape(-1, 1)
target = F.gelu(data)
target.requires_grad = True

criterion = nn.MSELoss()

def train_fourier_fit_relu(rank=6, mode='torch'):
    if mode == 'gdtuo':
        optimizer = gdtuo.Adam(alpha = 0.05)
        fourier = FourierAct(optimizer, rank)
        fourier.initialize()
    else:
        fourier = Fourier_Act_torch(rank)
        optimizer = torch.optim.Adam(fourier.parameters(), lr=0.05)
    
    if mode == 'gdtuo':
        for i in range(2000):
            fourier.begin()
            fourier.zero_grad()
            output = fourier(data)
            loss = criterion(output, target)
            loss.backward(create_graph=True)
            fourier.step()

        print(loss.item())
        torch.cuda.empty_cache()
        return fourier, fourier.parameters
    else:
        for i in range(2000):
            optimizer.zero_grad()
            output = fourier(data)
            loss = criterion(output, target)
            loss.backward(create_graph=True)
            optimizer.step()

        print(loss.item())
        torch.cuda.empty_cache()
        return fourier, fourier.params



def plot_fourier_fit_relu(fourier_act):
    data = torch.linspace(-4, 4, 10000).reshape(-1, 1)
    target = F.gelu(data)
    fourier_act.cpu()
    output = fourier_act(data)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(data.detach().numpy(), target.detach().numpy(), label='target')
    plt.plot(data.detach().numpy(), output.detach().numpy(), label='output')
    plt.legend()
    plt.show()
