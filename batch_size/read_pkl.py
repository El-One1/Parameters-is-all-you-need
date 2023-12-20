import pickle

model_name = "MNIST_MLP"
optimizer_name = "SGD"
dataset_name = "MNIST"
loss_fn_name = "CrossEntropyLoss"
lr = 0.1
momentum = 0.0
hyperoptimization = 0
batch_size_min = 5
batch_size_max = 5
target_acc = 0.80
nb_file = 0

name_file = "batch_size/results/" + model_name + "_" + optimizer_name + "_" + dataset_name + "_" + loss_fn_name + "_" + str(lr) + "_" + str(momentum) + "_" + str(hyperoptimization) +  "_" + str(batch_size_min) + "_" + str(batch_size_max) + "_" + str(target_acc) + "_" + str(nb_file) + ".pkl"


with open(name_file, 'rb') as f:
    steps = pickle.load(f)

print(steps)
