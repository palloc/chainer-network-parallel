#!/usr/bin/env python
import numpy as np
import six
import data
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from chainer import function


BATCH_SIZE = 100
N_EPOCH = 20
GPU1 = 0
GPU2 = 1


class NetworkParallel:
    def __init__(self):
        pass

    # Prepare dataset(MNIST)
    def read_dataset(self):
        print('load MNIST dataset')
        mnist = data.load_mnist_data()
        mnist['data'] = mnist['data'].astype(np.float32)
        mnist['data'] /= 255
        mnist['target'] = mnist['target'].astype(np.int32)

        # Number of data
        self.N = 60000
        self.x_train, self.x_test = np.split(mnist['data'],   [self.N])
        self.y_train, self.y_test = np.split(mnist['target'], [self.N])
        # Number of test data
        self.N_test = self.y_test.size

    def model_define(self):
        # Model1
        self.model = chainer.FunctionSet(l1=F.Linear(784, 1000),
                                         l2=F.Linear(1000, 1000),
                                         l3=F.Linear(1000, 100))

        # Model2
        self.model2 = chainer.FunctionSet(l1=F.Linear(100, 1000),
                                          l2=F.Linear(1000, 10))

        # Assign models to GPU
        self.model.to_gpu(GPU1)
        self.model2.to_gpu(GPU2)

    # Neural net architecture
    def forward(self, x_data, y_data, train=True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)

        h1 = F.dropout(F.relu(self.model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(self.model.l2(h1)), train=train)
        self.h3 = F.relu(self.model.l3(h2))

        # Change GPU1 --> GPU2
        self.h3.data = cuda.to_gpu(self.h3.data, device=GPU2)
        h4 = F.dropout(F.relu(self.model2.l1(self.h3)), train=train)
        y = self.model2.l2(h4)

        self.loss, self.acc = F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # Setting learning options and start learning
    def learning(self):
        # Setup optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

        # Learning loop
        for epoch in six.moves.range(1, N_EPOCH + 1):
            print('epoch', epoch)
            # Training
            self.perm = np.random.permutation(self.N)
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, self.N, BATCH_SIZE):
                self.x_batch = self.x_train[self.perm[i:i + BATCH_SIZE]]
                self.y_batch = self.y_train[self.perm[i:i + BATCH_SIZE]]
                # Reshape data for GPU
                self.x_batch = cuda.to_gpu(self.x_batch, device=GPU1)
                self.y_batch = cuda.to_gpu(self.y_batch, device=GPU2)
                # Forward
                self.optimizer.zero_grads()
                self.forward(self.x_batch, self.y_batch)
                # Backward
                self.loss.backward()
                self.optimizer.update()
                # Calc loss and accuracy
                sum_loss += float(cuda.to_cpu(self.loss.data))*len(self.y_batch)
                sum_accuracy += float(cuda.to_cpu(self.acc.data))*len(self.y_batch)

            print('train mean loss={}, accuracy={}'
                  .format(sum_loss/self.N, sum_accuracy/self.N))

            # Evaluation
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, self.N_test, BATCH_SIZE):
                # Forward
                self.forward(self.x_batch, self.y_batch, train=False)
                # Calc loss and accuracy
                sum_loss += float(cuda.to_cpu(self.loss.data))*len(self.y_batch)
                sum_accuracy += float(cuda.to_cpu(self.acc.data))*len(self.y_batch)

            print('test  mean loss={}, accuracy={}'
                  .format(sum_loss/self.N_test, sum_accuracy/self.N_test))


def main():
    test = NetworkParallel()
    test.read_dataset()
    test.model_define()
    test.learning()


if __name__ == '__main__':
    main()
