## 1. Adaptive skip connections : 
In a typical skip connection, implemented as $y = f(x) + x$ (where f is typically going to be some convolutions and activations), the input $x$ and the convolution output $f(x)$ have the same weight. However, we wanted to try giving them learnable weights. The resulting output becomes $y = (1-\alpha)f(x) + \alpha*x$, where $\alpha = \sigma(w_s)$ and $w_s$ is a learnable weight. We had two goals in mind : to see whether this new parameter would improve training performance, and to see whether the evolution of the "skip weight" $w_s$ teaches us anything interesting about skip connections. For example, whether skip connection closer to the first layer have lower weights since it becomes less and less crucial that they pass gradients during the backward pass.

The first goal turned out not to work so we didn't include particularly relevant code in this directory.


## 2. Layerwise Learning rate and exponential learning rates
This folder contains two files. `layerwise_lr_with_exp.ipynb` contains the code to test `SGDExp-SGD`, `SGDLW-SGD` and `SGDPW-SGD` on MNIST or CIFAR10 and to generate the related graphs from the paper. 
The second file, `stability.ipynb` contains the code to compare the training of `SGD-SGD` and `SGDExp-SGD` with various hyper-learning rates, in order to illustrate how more forgiving `SGDExp-SGD` is to "bad" choices of hyper learning rates.

## 3. Learnable activation functions:
This folder contains the learnable activation functions tested on CIFAR-10 and Caltech-256. Learnable activation functions are activations where the parameters are learnt during training. In particular, we present AdaGELU, a learnable GELU, which outperforms GELU on both datasets. We also present AdaReLU in the same spirit to gain insight on learnable activations, though its performance were not particularly good.
The folders also contain trials for Learnable polynoms and Fourier Series as activation functions but these did not show promising results. 

