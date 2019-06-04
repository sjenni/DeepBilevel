# Deep Bilevel Learning [[Project Page]](https://sjenni.github.io/DeepBilevel/) 

This repository contains demo code of our ECCV2018 [paper](https://arxiv.org/abs/1809.01465). It contains code for bilevel training of an Inception network on the CIFAR-10 dataset with noisy labels. 

## Requirements
The code is based on Python 2.7 and tensorflow 1.12.

## How to use it

### 1. Setup

- Set the paths to the data and log directories in **constants.py**.
- Run **init_cifar10.py** to download and convert the CIFAR-10 dataset. This also creates several noisy label files.

### 2. Bilevel training and evaluation 

- To train the Inception network with bilevel training on CIFAR-10 run **experiments_Inception_CIFAR10_bilevel_noise.py**.
- The bilevel algorithm is implemented in the **build_model** method of the **BilevelTrainer**.
