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
import torch

class PolyAct(Optimizable):
        def __init__(self, optimizer, rank = 5):
            self.n = rank
            self.coefs = nn.Parameter(torch.zeros(self.n))
            self.parameters = {'coefs': self.coefs}
            self.optimizer = optimizer
            self.all_params_with_gradients = [self.parameters['coefs']]
            super().__init__(self.parameters, self.optimizer)

        def __call__(self, x):
            out = 0
            for i in range(self.n):
                out += self.parameters['coefs'][i] * x ** i
            return out
        
        def step(self):
            self.optimizer.step(self.parameters)

data = torch.linspace(-4, 4, 10000).reshape(-1, 1)
target = F.relu(data)
target.requires_grad = True


criterion = nn.MSELoss()

def train_poly_fit_relu(rank=4):
    optimizer = gdtuo.Adam(alpha = 0.01)
    poly = PolyAct(optimizer, rank)
    poly.initialize()
    for i in range(1000):
        poly.begin()
        poly.zero_grad()
        output = poly(data)
        loss = criterion(output, target)
        loss.backward(create_graph=True)
        poly.step()

    return poly, poly.parameters['coefs'].clone().detach()

def plot_poly_fit_relu(polyact):
    data = torch.linspace(-4, 4, 10000).reshape(-1, 1)
    target = F.relu(data)
    output = polyact(data)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(data.detach().numpy(), target.detach().numpy(), label='target')
    plt.plot(data.detach().numpy(), output.detach().numpy(), label='output')
    plt.legend()
    plt.show()


print(train_poly_fit_relu())
