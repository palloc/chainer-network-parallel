#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import pprint
import data
from chainer import function



BATCH_SIZE = 100
N_EPOCH = 20
N_UNITS = 1000
GPU1 = 0
GPU2 = 1


# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size



# Model1 
model = chainer.FunctionSet(l1=F.Linear(784, N_UNITS),
                            l2=F.Linear(N_UNITS, 10000),
                            l3=F.Linear(10000, 100))

# Model2
model2 = chainer.FunctionSet(l1=F.Linear(100, 100),
                             l2=F.Linear(100, 10))

model.to_gpu(GPU1)
model2.to_gpu(GPU2)

    
# Neural net architecture


def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)

    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    h3 = F.relu(model.l3(h2))
    # Change GPU1 --> GPU2
    h3.data = cuda.to_gpu(h3.data, device=GPU2)
    h4 = F.dropout(F.relu(model2.l1(h3)), train=train)
    y = model2.l2(h4)
    
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# Learning loop
for epoch in six.moves.range(1, N_EPOCH + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, BATCH_SIZE):
        x_batch = x_train[perm[i:i + BATCH_SIZE]]
        y_batch = y_train[perm[i:i + BATCH_SIZE]]

        x_batch = cuda.to_gpu(x_batch, device=GPU1)
        y_batch = cuda.to_gpu(y_batch, device=GPU2)

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)

        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, BATCH_SIZE):
        x_batch = x_test[i:i + BATCH_SIZE]
        y_batch = y_test[i:i + BATCH_SIZE]

        x_batch = cuda.to_gpu(x_batch, device=GPU1)
        y_batch = cuda.to_gpu(y_batch, device=GPU2)

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
