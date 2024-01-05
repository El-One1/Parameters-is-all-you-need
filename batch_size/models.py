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

# SIMPLE CNN

class SIMPLE_CNN(nn.Module):
    def __init__(self):
        super(SIMPLE_CNN, self).__init__()
        
        # Première couche de convolution (adaptée pour une seule chaîne d'entrée)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # 'same' padding
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Deuxième couche de convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # 'same' padding
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couche complètement connectée
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)  # La taille de l'image MNIST est de 28x28 après deux poolings
        self.fc2 = nn.Linear(1024, 10)  # Nombre de classes dans MNIST

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        # Première couche de convolution
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Deuxième couche de convolution
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Aplatir avant la couche complètement connectée
        x = x.view(-1, 64 * 7 * 7)
        
        # Couche complètement connectée
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        
        return out