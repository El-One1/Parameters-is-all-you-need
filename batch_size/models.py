######################################################################
####################### LIBRAIRIES IMPORTATION #######################
######################################################################

import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

######################################################################
############################# MODELS #################################
######################################################################

# MNIST MLP

class MNIST_MLP(nn.Module):
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_MLP, self).__init__()
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        """Compute a prediction."""
        x = nn.Flatten()(x)
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x