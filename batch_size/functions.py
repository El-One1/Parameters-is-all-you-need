######################################################################
####################### LIBRAIRIES IMPORTATION #######################
######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from gradient_descent_the_ultimate_optimizer import gdtuo

######################################################################
############################ FUNCTIONS ###############################
######################################################################

################ VALIDATION ################

def validation(model, data_loader, device):
    model = copy.deepcopy(model)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for i, (images, labels) in enumerate(data_loader):
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, dim=1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      return correct / total

################# TRAINING #################

def training(model, train_dataset, test_dataset, optimizer, batch_sizes, target_acc, epoch, loss_fn, device):
    steps = []
    # Train the model for different batch sizes
    for batch_size in batch_sizes:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
        # Reset weights of the model
        model.reset_weights()
        model.train()
        N = len(train_loader.dataset)
        # Value to know if we must change the batch size
        new_batch = False
        for i in range(epoch):
            if new_batch:
                break
            loss_epoch = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                #print(batch_idx)
                optimizer.zero_grad()

                data = data.to(device)
                target = target.to(device)
                output = model(data)

                loss = loss_fn(output, target)

                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()

            loss_epoch /= len(train_loader)

            accuracy = validation(model, test_loader, device)
            if accuracy >= target_acc:
                print("Target reached in {} steps".format((i+1)*N/batch_size))
                steps.append(int((i+1)*N/batch_size))
                new_batch = True
                break

            print('Train Epoch: {} \tLoss: {:.6f} \t Val ACC: {:.3f}'.format(
                i, loss_epoch, accuracy))

    return steps

######################################################################
################ FUNCTIONS FOR HYPER OPTIMISATION ####################
######################################################################

################ VALIDATION ################

def validation_hyperopt(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for i, (images, labels) in enumerate(data_loader):
          #print(i)
          images = images.to(device)
          labels = labels.to(device)
          outputs = model.forward(images)
          _, predicted = torch.max(outputs.data, dim=1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      return correct / total
    
################# TRAINING #################
    
def training_hyperopt(model, train_dataset, test_dataset, optimizer_name, lr, batch_sizes, target_acc, epoch, loss_fn, device, lr_hypopt=10e-5):
    steps = []
    # Train the model for different batch sizes
    for batch_size in batch_sizes:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
        # Reset weights of the model
        model.reset_weights()
        if optimizer_name == "SGD":
            optim = gdtuo.SGD(alpha = lr, optimizer=gdtuo.SGD(lr_hypopt))
        mw = gdtuo.ModuleWrapper(model, optimizer=optim)
        N = len(train_loader.dataset)
        # Value to know if we must change the batch size
        new_batch = False
        #print(validation_hyperopt(mw, test_loader, device))
        for i in range(epoch):
            mw.train()

            if new_batch:
                break
            loss_epoch = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                mw.begin()

                mw.zero_grad()

                data = data.to(device)
                target = target.to(device)
                output = model.forward(data)

                loss = loss_fn(output, target)

                loss_epoch += loss.item()

                loss.backward(create_graph=True)
                mw.step()

            loss_epoch /= len(train_loader)

            accuracy = validation_hyperopt(mw, test_loader, device)
            if accuracy >= target_acc:
                print("Target reached in {} steps".format((i+1)*N/batch_size))
                steps.append(int((i+1)*N/batch_size))
                new_batch = True
                break

            print('Train Epoch: {} \tLoss: {:.6f} \t Val ACC: {:.3f}'.format(
                i, loss_epoch, accuracy))

    return steps
