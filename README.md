# Bayesian Neural Network - Mean Field Variational Inference

## 1) Introduction

## 2) How to run?

Example of run:

python main.py --dataset CIFAR10 --model_name Resnet --N_value 100 --alpha_value 0.001 --nb_epochs 200 --project_name resnet_cifar --lr=0.01 --vgg_type None

- dataset: {CIFAR10, MNIST, BOSTON}
- model_name: {Linear (1 hidden linear layer), Conv (1 hidden conv layer), VGG, Resnet (resnet20)}
- N_value: Number of neurons. (None if resnet or vgg)
- alpha_value: value of coefficient tau (ie alpha)
- nb_epochs: number of epochs
- projet_name: name of the projet in wandb
- lr: learning rate
- vgg_type: {11, 13, 16, None (if not vgg)}

## 3) Details of training

