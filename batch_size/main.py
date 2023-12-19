######################################################################
####################### LIBRAIRIES IMPORTATION #######################
######################################################################

import math
import torch
import torchvision
import torch.nn as nn

import pickle
import os

from gradient_descent_the_ultimate_optimizer import gdtuo

from functions import *
from models import *

######################################################################

####################### HYPERPARAMETERS #######################

# Batch sizes
batch_size_min = 5 # 2**x
batch_size_max = 10 # 2**x

# Target accuracy
target_acc = 0.97

# Number of epochs
epoch = 4000

# Model
model_name = "MNIST_MLP" # ["MNIST_MLP"]

# Optimizer
optimizer_name = "SGD" # ["SGD", "Adam", "RMSprop"]

# Learning rate
lr = 0.1

# Dataset
dataset_name = "MNIST" # ["MNIST", "CIFAR10"]

# Loss function
loss_fn_name = "CrossEntropyLoss" # ["CrossEntropyLoss", "NLLLoss"]

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Normal or hyperoptimization
hyperoptimization = 0 # [0, 1]

# Number of iterations
n_iter = 1

####################### MAIN #######################

# Batch sizes
batch_sizes = [2**x for x in range(batch_size_min, batch_size_max + 1)]

# Initialize the model
if model_name == "MNIST_MLP":
    model = MNIST_MLP(784, 128, 10)
    model.to(device)

# Initialize the optimizer
if not(hyperoptimization):
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Initialize the loss function
if loss_fn_name == "CrossEntropyLoss":
    loss_fn = nn.CrossEntropyLoss()

# Initialize the dataset
if dataset_name == "MNIST":
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    # Keep only n samples for the validation
    n = 1024
    valid_dataset, _ = torch.utils.data.random_split(test_dataset, [n, len(test_dataset) - n])

# Train the model for different batch sizes
mean_steps = [0] * len(batch_sizes)

if hyperoptimization:
    for i in range(iter):
        steps = training_hyperopt(model, train_dataset, test_dataset, optimizer_name, lr, batch_sizes, target_acc, epoch, loss_fn, device)
        mean_steps = [x + y for x, y in zip(mean_steps, steps)]
    mean_steps = [x / iter for x in mean_steps]
else:
    for i in range(iter):
        steps = training(model, train_dataset, test_dataset, optimizer, batch_sizes, target_acc, epoch, loss_fn, device)
        mean_steps = [x + y for x, y in zip(mean_steps, steps)]
    mean_steps = [x / iter for x in mean_steps]

# Save the results
name_file = "batch_size/results/" + model_name + "_" + optimizer_name + "_" + dataset_name + "_" + loss_fn_name + "_" + lr + "_" + str(hyperoptimization) +  "_" + str(batch_size_min) + "_" + str(batch_size_max) + "_" + str(target_acc)

nb_file = 0
for file in os.listdir("batch_size/results/"):
    if name_file in file:
        nb_file += 1

name_file += "_" + str(nb_file) + ".pkl"

with open(name_file, 'wb') as f:
    pickle.dump(mean_steps, f)