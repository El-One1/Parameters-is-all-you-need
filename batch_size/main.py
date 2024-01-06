######################################################################
####################### LIBRAIRIES IMPORTATION #######################
######################################################################

import math
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

import pickle
import os

from gradient_descent_the_ultimate_optimizer import gdtuo

from functions import *
from models import *

######################################################################

####################### HYPERPARAMETERS #######################

# Batch sizes
batch_size_min = 5 # 2**x
batch_size_max = 11 # 2**x

# Target accuracy
target_acc = 0.97

# Number of epochs
epoch = 4000

# Model
model_name = "MNIST_MLP" # ["MNIST_MLP", "SIMPLE_CNN", "RESNET_18"]

# Optimizer
optimizer_name = "SGD" # ["SGD", "Adam", "RMSprop"]

# Learning rate
lr = 0.1

# Momentum
momentum = 0.0

# Dataset
dataset_name = "MNIST" # ["MNIST", "CIFAR10"]

# Loss function
loss_fn_name = "CrossEntropyLoss" # ["CrossEntropyLoss", "NLLLoss"]

# Device
device_nb = 1
device = torch.device("cuda:"+str(device_nb) if torch.cuda.is_available() else "cpu")

# Normal or hyperoptimization
hyperoptimization = 1 # [0, 1]

# Number of iterations
n_iter = 10

####################### MAIN #######################

# Batch sizes
batch_sizes = [2**x for x in range(batch_size_min, batch_size_max + 1)]

# Initialize the model
if model_name == "MNIST_MLP":
    model = MNIST_MLP(784, 128, 10)
    model.to(device)
if model_name == "SIMPLE_CNN":
    model = SIMPLE_CNN()
    model.to(device)
if model_name == "RESNET_18":
    model = resnet18()
    model.to(device)

# Initialize the optimizer
if not(hyperoptimization):
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initialize the loss function
if loss_fn_name == "CrossEntropyLoss":
    loss_fn = nn.CrossEntropyLoss()

# Initialize the dataset
if dataset_name == "MNIST":
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    # Keep only n samples for the validation
    n = 2048
    valid_dataset, _ = torch.utils.data.random_split(test_dataset, [n, len(test_dataset) - n])

if dataset_name == "CIFAR10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    n = 1024
    valid_dataset, _ = torch.utils.data.random_split(test_dataset, [n, len(test_dataset) - n])


# Train the model for different batch sizes
mean_steps = [0] * len(batch_sizes)
mean_times = [0] * len(batch_sizes)

if hyperoptimization:
    for i in range(n_iter):
        steps, times = training_hyperopt(model, train_dataset, valid_dataset, optimizer_name, lr, momentum, batch_sizes, target_acc, epoch, loss_fn, device)
        mean_steps = [x + y for x, y in zip(mean_steps, steps)]
        mean_times = [x + y for x, y in zip(mean_times, times)]
    mean_steps = [x / n_iter for x in mean_steps]
    mean_times = [x / n_iter for x in mean_times]
else:
    for i in range(n_iter):
        steps, times = training(model, train_dataset, valid_dataset, optimizer, batch_sizes, target_acc, epoch, loss_fn, device)
        mean_steps = [x + y for x, y in zip(mean_steps, steps)]
        mean_times = [x + y for x, y in zip(mean_times, times)]
    mean_steps = [x / n_iter for x in mean_steps]
    mean_times = [x / n_iter for x in mean_times]

# Save the results
name_file = "results/" + model_name + "_" + optimizer_name + "_" + dataset_name + "_" + loss_fn_name + "_" + str(lr) + "_" + str(momentum) + "_" + str(hyperoptimization) +  "_" + str(batch_size_min) + "_" + str(batch_size_max) + "_" + str(target_acc)

nb_file = 1
for file in os.listdir("results/"):
    if name_file in file:
        nb_file += 1

name_file += "_" + str(nb_file) + ".pkl"

data = (mean_steps, mean_times)

with open(name_file, 'wb') as f:
    pickle.dump(data, f)